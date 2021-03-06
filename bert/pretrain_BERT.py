"""Pre-train BERT for language identification using masked language
modeling and sequence pair classification.

"""

import os, argparse, logging, pickle, glob, math
from io import open
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM, BertConfig
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import trange
from CharTokenizer import CharTokenizer
from BertDataset import BertDatasetForMLM, BertDatasetForSPCAndMLM, NO_MASK_LABEL
from Pooler import Pooler
from utils import check_for_unk_train_data, adjust_loss, weighted_avg, count_params, accuracy, get_dataloader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, pooler, tokenizer, optimizer, scheduler, dataset, args, checkpoint_data, extra_mlm_dataset=None):
    """Pretrain a BertModelForMaskedLM using both sentence pair
    classification and MLM.
    
    Args:
    - model: BertModelForMaskedLM
    - pooler: Pooler used for SPC (required unless args.mlm_only)
    - tokenizer: CharTokenizer
    - optimizer
    - scheduler
    - dataset: BertDatasetForMLM if args.mlm_only, otherwise BertDatasetForSPCAndMLM
    - args
    - checkpoint_data: dict
    - (Optional) extra_mlm_dataset: a BertDatasetForMLM. If provided, we add
    a batch from this to every batch of the other dataset. This is
    useful for pre-training on the unlabeled test data, which we can
    not use for sentence pair classification.

    """
    if args.mlm_only:
        assert type(dataset) == BertDatasetForMLM
        assert extra_mlm_dataset is None
    else:
        assert pooler is not None
        assert type(dataset) == BertDatasetForSPCAndMLM
    if extra_mlm_dataset is not None:
        assert len(dataset) == len(extra_mlm_dataset)
        
    # Write header in log
    header = "GlobalStep\tLossMLM\tAccuracyMLM"
    if not args.mlm_only:
        header += "\tLossSPC\tAccuracySPC"
    if extra_mlm_dataset is not None:
        header += "\tLossExtraMLM\tAccuracyExtraMLM"
    header += "\tGradNorm\tWeightNorm"
    with open(args.train_log_path, "w") as f:
        f.write(header + "\n")

    # Make dataloader(s). Note: since BertDatasetForTraining and its
    # subclasses are IterableDatasets (i.e. streams), the loader is an
    # iterable (with no end and no __len__) that we call with iter().
    dataloader = get_dataloader(dataset, args.train_batch_size, args.local_rank)
    train_batch_sampler = iter(dataloader) 
    if extra_mlm_dataset is not None:
        mlm_dataloader = get_dataloader(extra_mlm_dataset, args.train_batch_size, args.local_rank)
        mlm_batch_enum = enumerate(iter(mlm_dataloader))

    # Start training
    logger.info("***** Running training *****")
    model.train()
    for epoch in trange(int(args.num_epochs), desc="Epoch"):
        
        # Some stats for this epoch
        real_batch_sizes = []
        query_mlm_losses = []
        query_mlm_accs = []
        spc_losses = []
        spc_accs = []
        extra_mlm_losses = []
        extra_mlm_accs = []
        grad_norms = []
        
        # Run training for one epoch
        for step in trange(int(args.num_train_steps_per_epoch), desc="Iteration"):
            batch = next(train_batch_sampler)
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack
            if args.mlm_only:
                input_ids_query = batch[0]
                input_mask_query = batch[1]
                segment_ids_query = batch[2]
                lm_label_ids = batch[3]                
            else:    
                input_ids_query = batch[0]
                input_mask_query = batch[1]
                segment_ids_query = batch[2]
                input_ids_cands = batch[3]
                input_mask_cands = batch[4]
                segment_ids_cands = batch[5]
                cand_labels = batch[6]
                lm_label_ids = batch[7]
            real_batch_sizes.append(len(input_ids_query))

            # Call underlying BERT model to get encoding of query
            outputs = model.bert(input_ids=input_ids_query,
                                 attention_mask=input_mask_query,
                                 token_type_ids=segment_ids_query,
                                 position_ids=None)
            query_last_hidden_states = outputs[0] # Last hidden states, shape (batch_size, seq_len, hidden_size)

            # Do MLM on last hidden states obtained using query
            # inputs.
            mlm_pred_scores = model.cls(query_last_hidden_states)

            if not args.mlm_only:
                # Get encodings of query and candidates for SPC
                query_encodings = pooler(query_last_hidden_states)
                all_cand_encodings = []            
                for i in range(2):
                    outputs = model.bert(input_ids=input_ids_cands[:,i,:],
                                         attention_mask=input_mask_cands[:,i,:],
                                         token_type_ids=segment_ids_cands[:,i,:],
                                         position_ids=None)
                    last_hidden_states = outputs[0]
                    encodings = pooler(last_hidden_states)
                    all_cand_encodings.append(encodings.unsqueeze(1))
                cand_encodings = torch.cat(all_cand_encodings, dim=1)            

                # Score candidates using dot(query, candidate)
                spc_scores = torch.bmm(cand_encodings, query_encodings.unsqueeze(2)).squeeze(2)

            if extra_mlm_dataset is not None:
                # Do MLM on extra_mlm_dataset
                mlm_batch_id, mlm_batch = next(mlm_batch_enum)
                # Make sure the training steps are synced
                assert mlm_batch_id == step
                mlm_batch = tuple(t.to(args.device) for t in mlm_batch)
                xinput_ids, xinput_mask, xsegment_ids, xlm_label_ids = mlm_batch
                # Make sure the batch sizes are equal
                assert len(xinput_ids) == len(input_ids_query)
                outputs = model.bert(input_ids=xinput_ids,
                                     attention_mask=xinput_mask,
                                     token_type_ids=xsegment_ids,
                                     position_ids=None)
                extra_last_hidden_states = outputs[0] # Last hidden states, shape (batch_size, seq_len, hidden_size)
                extra_mlm_pred_scores = model.cls(extra_last_hidden_states)
                
            # Compute loss, do backprop. Compute accuracies.
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(mlm_pred_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1))
            query_mlm_losses.append(loss.item())
            if not args.mlm_only:
                spc_loss = loss_fct(spc_scores, cand_labels)
                loss = loss + spc_loss
                spc_losses.append(spc_loss.item())
            if extra_mlm_dataset is not None:
                extra_mlm_loss = loss_fct(extra_mlm_pred_scores.view(-1, model.config.vocab_size), xlm_label_ids.view(-1))
                loss = loss + extra_mlm_loss
                extra_mlm_losses.append(extra_mlm_loss.item())

            # Backprop
            loss = adjust_loss(loss, args)
            loss.backward()
            
            # Compute norm of gradient
            training_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    training_grad_norm += torch.norm(param.grad, p=2).item()
            if not args.mlm_only:
                for param in pooler.parameters():
                    if param.grad is not None:
                        training_grad_norm += torch.norm(param.grad, p=2).item()
            grad_norms.append(training_grad_norm)

            # Compute accuracies
            query_mlm_acc = accuracy(mlm_pred_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1), ignore_label=NO_MASK_LABEL)
            query_mlm_accs.append(query_mlm_acc)
            if not args.mlm_only:
                spc_acc = accuracy(spc_scores, cand_labels)
                spc_accs.append(spc_acc)
            if extra_mlm_dataset is not None:
                extra_mlm_acc = accuracy(extra_mlm_pred_scores.view(-1, model.config.vocab_size), xlm_label_ids.view(-1), ignore_label=NO_MASK_LABEL)
                extra_mlm_accs.append(extra_mlm_acc)

            # Check if we accumulate grad or do an optimization step
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                checkpoint_data["global_step"] += 1
                if checkpoint_data["global_step"] >= checkpoint_data["max_opt_steps"]:
                    break
                
        # Compute stats for this epoch
        last_grad_norm = grad_norms[-1]
        avg_query_mlm_loss = weighted_avg(query_mlm_losses, real_batch_sizes)        
        avg_query_mlm_acc = weighted_avg(query_mlm_accs, real_batch_sizes)
        if not args.mlm_only:
            avg_spc_loss = weighted_avg(spc_losses, real_batch_sizes)
            avg_spc_acc = weighted_avg(spc_accs, real_batch_sizes)
        if extra_mlm_dataset is not None:
            avg_extra_mlm_loss = weighted_avg(extra_mlm_losses, real_batch_sizes)
            avg_extra_mlm_acc = weighted_avg(extra_mlm_accs, real_batch_sizes)

        # Compute norm of model weights
        weight_norm = 0
        for param in model.parameters():
            weight_norm += torch.norm(param.data, p=2).item()
        if not args.mlm_only:
            for param in pooler.parameters():
                weight_norm += torch.norm(param.data, p=2).item()
            
        # Write stats for this epoch in log
        log_data = []
        log_data.append(str(checkpoint_data["global_step"]))
        log_data.append("{:.5f}".format(avg_query_mlm_loss))
        log_data.append("{:.5f}".format(avg_query_mlm_acc))
        if not args.mlm_only:
            log_data.append("{:.5f}".format(avg_spc_loss))
            log_data.append("{:.5f}".format(avg_spc_acc))
        if extra_mlm_dataset is not None:
            log_data.append("{:.5f}".format(avg_extra_mlm_loss))
            log_data.append("{:.5f}".format(avg_extra_mlm_acc))
        log_data.append("{:.5f}".format(last_grad_norm))
        log_data.append("{:.5f}".format(weight_norm))        
        with open(args.train_log_path, "a") as f:
            f.write("\t".join(log_data)+"\n")

        # Save checkpoint
        model_to_save = model.module if hasattr(model, 'module') else model
        pooler_to_save = pooler.module if hasattr(pooler, 'module') else pooler
        checkpoint_data['model_state_dict'] = model_to_save.state_dict()
        if not args.mlm_only:
            checkpoint_data['pooler_state_dict'] = pooler_to_save.state_dict()
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()        
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint_path = os.path.join(args.output_dir, "checkpoint.tar")
        torch.save(checkpoint_data, checkpoint_path)            


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model_or_config_file", 
                        default=None, 
                        type=str, 
                        required=True,
                        help=("Path of configuration file (if starting from scratch) or directory"
                              " containing checkpoint (if resuming) or directory containig a"
                              " pretrained model and tokenizer (if re-training)."))

    # Use for resuming from checkpoint
    parser.add_argument("--resume",
                        action='store_true',
                        help="Resume from checkpoint")
    
    # Required if not resuming
    parser.add_argument("--dir_train_data",
                        type=str,
                        help="Path of a directory containing training files (names must all match <lang>.train)")
    parser.add_argument("--path_vocab",
                        type=str,
                        help="Path of a 2-column TSV file containing the vocab of chars and their frequency.")
    parser.add_argument("--output_dir",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--mlm_only",
                        action="store_true",
                        help=("Use only masked language modeling, no sentence pair classification "
                              " (e.g. if you only have unk.train in your training directory)"))
    parser.add_argument("--avgpool_for_spc",
                        action="store_true",
                        help=("Use average pooling of all last hidden states, rather than just the last hidden state of CLS, to do SPC. "
                              "Note that in either case, the pooled vector passes through a square linear layer and a tanh before the classification layer."))
    parser.add_argument("--sampling_alpha",
                        type=float,
                        default=1.0,
                        help="Dampening factor for relative frequencies used to compute language sampling probabilities")
    parser.add_argument("--weight_relevant",
                        type=float,
                        default=1.0,
                        help="Relative sampling frequency of relevant languages wrt irrelevant languages")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seq_len",
                        default=128,
                        type=int,
                        help="Length of input sequences. Shorter seqs are padded, longer ones are trucated")
    parser.add_argument("--min_freq",
                        default=1,
                        type=int,
                        help="Minimum character frequency. Characters whose frequency is under this threshold will be mapped to <UNK>")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--equal_betas",
                        action='store_true',
                        help="Use beta1=beta2=0.9 for AdamW optimizer.")
    parser.add_argument("--correct_bias",
                        action='store_true',
                        help="Correct bias in AdamW optimizer (correct_bias=False is meant to reproduce BERT behaviour exactly.")
    parser.add_argument("--max_train_steps",
                        default=1000000,
                        type=int,
                        help="Maximum number of training steps to perform. Note: # optimization steps = # train steps / # accumulation steps.")
    parser.add_argument("--num_train_steps_per_epoch",
                        default=1000,
                        type=int,
                        help="Number of training steps that equals one epoch. Note: # optimization steps = # train steps / # accumulation steps.")
    parser.add_argument("--num_warmup_steps",
                        default=10000,
                        type=int,
                        help="Number of optimization steps (i.e. training steps / accumulation steps) to perform linear learning rate warmup for. ")
    parser.add_argument('--grad_accum_steps',
                        type=int,
                        default=1,
                        help="Number of training steps (i.e. batches) to accumualte before performing a backward/update pass.")
    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="Num GPUs to use for training (0 for none, -1 for all available)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    # These args are required if we are not resuming from checkpoint
    if not args.resume:
        assert args.dir_train_data is not None
        assert args.path_vocab is not None
        assert args.output_dir is not None
        
    # Check whether we are starting from scratch, resuming from a checkpoint, or retraining a pretrained model
    from_scratch = (not args.resume) and (not os.path.isdir(args.bert_model_or_config_file))
    retraining = (not args.resume) and (not from_scratch)
    
    # Load config. Load or create checkpoint data.
    if from_scratch:
        logger.info("***** Starting pretraining job from scratch *******")
        config = BertConfig.from_json_file(args.bert_model_or_config_file)
        checkpoint_data = {}
    elif retraining:
        logger.info("***** Starting pretraining job from pre-trained model *******")
        logger.info("Loading pretrained model...")
        model = BertModelForMaskedLM.from_pretrained(args.bert_model_or_config_file)
        config = model.config
        checkpoint_data = {}
    elif args.resume:
        logger.info("***** Resuming pretraining job *******")
        logger.info("Loading checkpoint...")
        checkpoint_path = os.path.join(args.bert_model_or_config_file, "checkpoint.tar")        
        checkpoint_data = torch.load(checkpoint_path)
        # Make sure we haven't already done the maximum number of optimization steps
        if checkpoint_data["global_step"] >= checkpoint_data["max_opt_steps"]:
            msg = "We have already done %d optimization steps." % checkpoint_data["global_step"]
            raise RuntimeError(msg)
        logger.info("Resuming from global step %d" % checkpoint_data["global_step"])
        # Replace args with initial args for this job, except for num_gpus, seed and model directory
        current_num_gpus = args.num_gpus
        current_seed = args.seed
        checkpoint_dir = args.bert_model_or_config_file
        args = deepcopy(checkpoint_data["initial_args"])
        args.num_gpus = current_num_gpus
        args.seed = current_seed
        args.bert_model_or_config_file = checkpoint_dir
        args.resume = True
        logger.info("Args (most have been reloaded from checkpoint): %s" % args)
        # Load config
        config_path = os.path.join(args.bert_model_or_config_file, "config.json")
        config = BertConfig.from_json_file(config_path)        

    # Check args
    assert args.sampling_alpha >= 0 and args.sampling_alpha <= 1
    assert args.weight_relevant > 0
    if args.grad_accum_steps < 1:
        raise ValueError("Invalid grad_accum_steps parameter: {}, should be >= 1".format(
                            args.grad_accum_steps))
    train_paths = glob.glob(os.path.join(args.dir_train_data, "*.train"))
    assert len(train_paths) > 0
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if (not args.resume) and len(os.listdir(args.output_dir)) > 0:
        msg = "Directory %s is not empty" % args.output_dir
        raise ValueError(msg)
    
    # Make or load tokenizer
    if args.resume or retraining:
        logger.info("Loading tokenizer...")
        tokenizer_path = os.path.join(args.bert_model_or_config_file, "tokenizer.pkl")
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    elif from_scratch:
        logger.info("Making tokenizer...")

        assert os.path.exists(args.path_vocab)
        tokenizer = CharTokenizer(args.path_vocab)
        if args.min_freq > 1:
            tokenizer.trim_vocab(args.min_freq)
        # Adapt vocab size in config
        config.vocab_size = len(tokenizer.vocab)

        # Save tokenizer
        fn = os.path.join(args.output_dir, "tokenizer.pkl")
        with open(fn, "wb") as f:
            pickle.dump(tokenizer, f)
    logger.info("Size of vocab: {}".format(len(tokenizer.vocab)))

    # Copy config in output directory
    if not args.resume:
        config_path = os.path.join(args.output_dir, "config.json")
        config.to_json_file(config_path)
        
    # What GPUs do we use?
    if args.num_gpus == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        device_ids = None
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")
        args.n_gpu = args.num_gpus
        if args.n_gpu > 1:
            device_ids = list(range(args.n_gpu))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        args.device, args.n_gpu, bool(args.local_rank != -1)))
    
    # Seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
                    
    # Prepare model 
    if from_scratch or args.resume:
        model = BertForMaskedLM(config)
        if args.resume:
            model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(args.device)

    # Prepare pooler (if we are doing SPC)
    if args.mlm_only:
        pooler = None
    else:
        pooler = Pooler(model.config.hidden_size, cls_only=(not args.avgpool_for_spc))
        if args.resume:
            pooler.load_state_dict(checkpoint_data["pooler_state_dict"])
        pooler.to(args.device)

    # Distributed or parallel?
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed training.")
        model = DDP(model)
        if not args.mlm_only:
            pooler = DDP(pooler) 
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        pooler = torch.nn.DataParallel(pooler, device_ids=device_ids)

    # Log some info on the model
    logger.info("Model config: %s" % repr(model.config))
    logger.info("Nb params: %d" % count_params(model))
    if not args.mlm_only:
        logger.info("Nb params in pooler: %d" % count_params(pooler))        

    # Check if there is unk training data. 
    path_unk = check_for_unk_train_data(train_paths)

    # Get training data
    max_seq_length = args.seq_len + 2 # We add 2 for CLS and SEP
    logger.info("Preparing dataset using data from %s" % args.dir_train_data)
    if args.mlm_only:
        # We only want to do MLM
        train_dataset_spc = None
        train_dataset_mlm = BertDatasetForMLM(train_paths,
                                              tokenizer,
                                              max_seq_length,
                                              sampling_alpha=args.sampling_alpha,
                                              weight_relevant=args.weight_relevant,
                                              encoding="utf-8",
                                              seed=args.seed)

    else:
        # We want do to SLC and MLM. If unk data is present, we remove
        # it from the paths provided to BertLabeledDataset.
        if path_unk is not None:
            train_paths.remove(path_unk)
        train_dataset_spc = BertDatasetForSPCAndMLM(train_paths,
                                                    tokenizer,
                                                    max_seq_length,
                                                    sampling_alpha=args.sampling_alpha,
                                                    weight_relevant=args.weight_relevant,
                                                    encoding="utf-8",
                                                    seed=args.seed)
        if path_unk is None:
            train_dataset_mlm = None
        else:
            # In this case we use a BertDatasetForMLM for the unk
            # data. Both datasets will be of the same size. The latter
            # is used for MLM only.
            train_dataset_mlm = BertDatasetForMLM([path_unk],
                                                  tokenizer,
                                                  max_seq_length,
                                                  sampling_alpha=args.sampling_alpha,
                                                  weight_relevant=args.weight_relevant,
                                                  encoding="utf-8",
                                                  seed=args.seed)
            assert len(train_dataset_spc) == len(train_dataset_mlm)
                    
    # Check length of dataset
    dataset_length = len(train_dataset_spc) if train_dataset_spc is not None else len(train_dataset_mlm)

    # Store optimization steps performed and maximum number of optimization steps 
    if not args.resume:
        checkpoint_data["global_step"] = 0
        checkpoint_data["max_opt_steps"] = args.max_train_steps // args.grad_accum_steps

    # Compute number of optimization steps per epoch
    num_opt_steps_per_epoch = args.num_train_steps_per_epoch // args.grad_accum_steps

    # Compute number of epochs necessary to reach the maximum number of optimization steps
    opt_steps_left = checkpoint_data["max_opt_steps"] - checkpoint_data["global_step"]
    args.num_epochs = math.ceil(opt_steps_left / num_opt_steps_per_epoch)
                    
    # Log some info before training
    logger.info("*** Training info: ***")
    logger.info("Max training steps: %d" % args.max_train_steps)
    logger.info("Gradient accumulation steps: %d" % args.grad_accum_steps)
    logger.info("Max optimization steps: %d" % checkpoint_data["max_opt_steps"])
    if args.resume:
        logger.info("Nb optimization steps done so far: %d" % checkpoint_data["global_step"])
    logger.info("Total dataset size: %d examples" % (dataset_length))
    logger.info("Batch size: %d" % args.train_batch_size)
    logger.info("# training steps/epoch: %d" % (args.num_train_steps_per_epoch))    
    logger.info("# optimization steps/epoch: %d" % (num_opt_steps_per_epoch))
    logger.info("# epochs to do: %d" % (args.num_epochs))
        
    # Prepare optimizer
    logger.info("Preparing optimizer...")
    np_list = list(model.named_parameters())
    if not args.mlm_only:
        np_list += list(pooler.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_params = [
        {'params': [p for n, p in np_list if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in np_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.equal_betas:
        betas = (0.9, 0.9)
    else:
        betas = (0.9, 0.999)
    optimizer = AdamW(opt_params,
                      lr=args.learning_rate,
                      betas=betas,
                      correct_bias=args.correct_bias) # To reproduce BertAdam specific behaviour, use correct_bias=False
    if args.resume:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    # Prepare scheduler
    logger.info("Preparing learning rate scheduler...")
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=checkpoint_data["max_opt_steps"])
    if args.resume:
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        logger.info("Current learning rate: %f" % scheduler.get_last_lr()[0])

    # Save initial training args
    if not args.resume:
        checkpoint_data["initial_args"] = args
    
    # Prepare training log
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    train_log_path = os.path.join(args.output_dir, "%s.train.log" % time_str)        
    args.train_log_path = train_log_path
    
    # Train
    if args.mlm_only:
        train(model, None, tokenizer, optimizer, scheduler, train_dataset_mlm, args, checkpoint_data, extra_mlm_dataset=None)
    else:
        train(model, pooler, tokenizer, optimizer, scheduler, train_dataset_spc, args, checkpoint_data, extra_mlm_dataset=train_dataset_mlm)

if __name__ == "__main__":
    main()
