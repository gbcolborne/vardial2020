""" Train or evaluate classifier. """

import sys, os, argparse, pickle, random, logging, math
from io import open
from copy import deepcopy
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from transformers import BertForMaskedLM, BertConfig
from transformers import AdamW
from tqdm import tqdm, trange
from CharTokenizer import CharTokenizer
from BertDataset import BertDatasetForClassification, BertDatasetForTesting
from BertForLangID import BertForLangID
from utils import adjust_loss, weighted_avg, count_params, accuracy, get_dataloader, get_module
sys.path.append("..")
from comp_utils import ALL_LANGS
from scorer import compute_fscores

DEBUG=False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model, eval_dataset, target_lang_id, args):
    """ Evaluate model on multi-class language identification. 

    Args:
    - model: BertModelForLangID
    - eval_dataset: BertDatasetForTesting
    - args

    Returns: Dict containing scores

    """
    # Get logits (un-normalized class scores) from model
    logits = predict(model, eval_dataset, args)

    # Extract labels
    gold_labels = [x[3] for x in eval_dataset]

    # Compute loss
    logger.info("Logits:")
    logger.info(logits)
    target_logits = logits[target_lang_id]
    logger.info("Target logits:")
    logger.info(target_logits)
    logger.info("Labels:")
    logger.info(gold_labels)
    
    loss_fct = BCEWithLogitsLoss(reduction="mean")
    loss = loss_fct(target_logits, gold_labels)
    scores = {}
    scores["loss"] = loss.item()
    
    # Compute f-scores
    pred_labels = np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()    
    gold_labels = gold_labels.detach().cpu().numpy().tolist()
    pred_labels = [eval_dataset.label_list[i] for i in pred_labels]
    gold_labels = [eval_dataset.label_list[i] for i in gold_labels]
    fscore_dict = compute_fscores(pred_labels, gold_labels, verbose=False)
    scores.update(fscore_dict)
    return scores


def predict(model, eval_dataset, args):
    """ Get predicted scores (un-normalized) for examples in dataset. 

    Args:
    - model: BertForLangID
    - eval_dataset: BertDatasetForTesting
    - args


    Returns: predicted scores, tensor of shape (nb examples, nb classes)

    """
    assert type(eval_dataset) == BertDatasetForTesting
    dataloader = get_dataloader(eval_dataset, args.eval_batch_size, args.local_rank)
    scores = []
    model.eval()
    for step, batch in enumerate(tqdm(dataloader, desc="Prediction")):
        # Unpack batch
        batch = tuple(t.to(args.device) for t in batch)
        input_ids = batch[0]
        input_mask = batch[1]
        segment_ids = batch[2]
        with torch.no_grad():
            logits = model(input_ids, input_mask, segment_ids, cls=None)
        scores.append(logits)
    scores_tensor = torch.cat(scores, dim=0)
    return scores_tensor


def train(model, optimizer, tokenizer, target_lang, args, checkpoint_data, dev_dataset=None):
    """ Train model. 

    Args:
    - model: BertModelForLangID
    - optimizer
    - tokenizer
    - target_lang: language on which we train
    - args
    - checkpoint_data: dict
    - dev_dataset: (optional) BertDatasetForTesting

    Returns: None

    """
    if args.eval_during_training:
        assert dev_dataset is not None
        assert type(dev_dataset) == BertDatasetForTesting
    target_lang_id = model.lang2id[target_lang]

    # Where do we save stuff?
    save_to_dir = args.dir_pretrained_model if args.resume else args.dir_output
    
    # Make or load dataset
    path_data = os.path.join(save_to_dir, "train-set-%s.pkl" % target_lang)
    if args.resume:
        logger.info("Reloading training set for '%s' from %s" % (target_lang, path_data))
        with open(path_data, "rb") as f:
            train_dataset = pickle.load(f)
        logger.info("Dataset size: %d" % len(train_dataset))            
    else:
        logger.info("Making training set for '%s' using data in %s..." % (target_lang, args.dir_data))
        train_dataset = BertDatasetForClassification(args.dir_data,
                                                     target_lang,
                                                     tokenizer,
                                                     args.max_seq_len,
                                                     sampling_alpha=args.sampling_alpha,
                                                     weight_relevant=args.weight_relevant,
                                                     encoding="utf-8",
                                                     seed=args.seed,
                                                     verbose=DEBUG)
        logger.info("Saving training set at %s" % path_data)        
        with open(path_data, "wb") as f:
            pickle.save(train_dataset, f)


    # Compute number of optimization steps we need to do            
    if args.resume:
        global_step = checkpoint_data["global_step"][target_lang]
        nb_opt_steps = checkpoint_data["nb_opt_steps"][target_lang]
        nb_opt_steps_per_epoch = len(train_dataset) // args.train_batch_size // args.grad_accum_steps
        epochs_done = global_step // num_opt_steps_per_epoch
        nb_batches_to_skip = (global_step % num_opt_steps_per_epoch) * args.grad_accum_steps
        logger.info("Resuming training from optimization step %d/%d" % (global_step, nb_opt_steps))
        logger.info("Nb epochs completed so far: %d" % epochs_done)
    else:
        epochs_done = 0
        nb_opt_steps_per_epoch = len(train_dataset) // args.train_batch_size // args.grad_accum_steps
        nb_opt_steps = nb_opt_steps_per_epoch * args.num_train_epochs
        logger.info("Nb optimization steps to do: %d" % nb_opt_steps)
        logger.info("Nb optimization steps per epoch: %d" % nb_opt_steps_per_epoch)
        checkpoint_data["nb_opt_steps"][target_lang] = nb_opt_steps
        checkpoint_data["global_step"][target_lang] = 0
            
    # Make dataloader
    train_dataloader = get_dataloader(train_dataset, args.train_batch_size, args.local_rank)

    # Skip batches if resuming
    if args.resume:
        logger.info("Skipping %d batches already observed this epoch" % nb_batches_to_skip)        
        for _ in nb_batches_to_skip:
            next(train_dataloader)
            
    # Prepare log
    train_log_path = os.path.join(save_to_dir, "train-%s.log" % target_lang)

    # Write header in log
    if not args.resume:
        header = "GlobalStep\tLossLangID\tAccuracyLangID"
        header += "\tGradNorm\tWeightNorm"
        if args.eval_during_training:
            header += "\tDevLoss\tDevF1Track1\tDevF1Track2\tDevF1Track3"
        with open(train_log_path, "w") as f:
            f.write(header + "\n")
            
    # Evaluate model on dev set before training
    if args.resume:
        best_score = checkpoint_data["best_score"]
    elif args.eval_during_training:
        logger.info("Evaluating model on dev set before we start training...")
        dev_scores = evaluate(model, dev_dataset, args)
        best_score = dev_scores[args.score_to_optimize]
        checkpoint_data["best_score"] = best_score
        log_data = []
        log_data.append(str(checkpoint_data["global_step"][target_lang]))
        log_data += ["", "", "", ""]
        log_data.append("{:.5f}".format(dev_scores["loss"]))
        log_data.append("{:.5f}".format(dev_scores["track1"]))
        log_data.append("{:.5f}".format(dev_scores["track2"]))
        log_data.append("{:.5f}".format(dev_scores["track3"]))                            
        with open(train_log_path, "a") as f:
            f.write("\t".join(log_data)+"\n")
        
    # Start training
    logger.info("***** Running training *****")
    for epoch in trange(epochs_done, args.num_epochs, desc="Epoch"):
        model.train()
                  
        # Some stats for this epoch
        real_batch_sizes = []
        lid_losses = []
        lid_accs = []
        grad_norms = []
        
        # Run training for one epoch
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids = batch[0]
            input_mask = batch[1]
            segment_ids = batch[2]
            labels = batch[3]
            lang_ids = batch[4]                
            real_batch_sizes.append(len(input_ids))

            # Call BERT encoder to get encoding of (un-masked) input sequences
            logits = model(input_ids, input_mask, segment_ids, cls=target_lang_id)

            # Compute loss, do backprop. Compute accuracies.


            logger.info("Logits:")
            logger.info(logits)
            target_logits = logits[labels]
            logger.info("Target logits:")
            logger.info(target_logits)
    
            loss_fct = BCEWithLogitsLoss(reduction="mean")
            loss = loss_fct(target_logits, labels)
            lid_losses.append(loss.item())

            # Backprop
            loss = adjust_loss(loss, args)
            loss.backward()
            
            # Compute norm of gradient
            training_grad_norm = 0
            for param in model.parameters()
                if param.grad is not None:
                    training_grad_norm += torch.norm(param.grad, p=2).item()
            grad_norms.append(training_grad_norm)

            # Compute accuracies
            lid_acc = sys.exit()
            lid_accs.append(lid_acc)

            # Check if we accumulate grad or do an optimization step
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                checkpoint_data["global_step"][target_lang] += 1
                if checkpoint_data["global_step"][target_lang] >= checkpoint_data["nb_opt_steps"][target_lang]:
                    break
                
        # Compute stats for this epoch
        last_grad_norm = grad_norms[-1]
        avg_lid_loss = weighted_avg(lid_losses, real_batch_sizes)
        avg_lid_acc = weighted_avg(lid_accs, real_batch_sizes)

        # Compute norm of model weights
        weight_norm = 0
        for param in model.parameters():
            weight_norm += torch.norm(param.data, p=2).item()

        # Evaluate model on dev set
        if args.eval_during_training:
            dev_scores = evaluate(model, dev_dataset, args)            
            
        # Write stats for this epoch in log
        log_data = []
        log_data.append(str(checkpoint_data["global_step"][target_lang]))
        log_data.append("{:.5f}".format(avg_lid_loss))
        log_data.append("{:.5f}".format(avg_lid_acc))
        log_data.append("{:.5f}".format(last_grad_norm))
        log_data.append("{:.5f}".format(weight_norm))
        if args.eval_during_training:
            log_data.append("{:.5f}".format(dev_scores["loss"]))
            log_data.append("{:.5f}".format(dev_scores["track1"]))
            log_data.append("{:.5f}".format(dev_scores["track2"]))
            log_data.append("{:.5f}".format(dev_scores["track3"]))                            
        with open(train_log_path, "a") as f:
            f.write("\t".join(log_data)+"\n")

        # Save best model if score has improved
        if args.eval_during_training:
            current_score = dev_scores[args.score_to_optimize]
            if current_score > best_score:
                best_score = current_score
                checkpoint_data["best_score"] = best_score
                model_to_save = get_module(model)
                checkpoint_data['best_model_state_dict'] = deepcopy(model_to_save.state_dict())

        # Save checkpoint
        model_to_save = get_module(model)
        checkpoint_data['model_state_dict'] = model_to_save.state_dict()
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint_path = os.path.join(save_to_dir, "checkpoint.tar")
        torch.save(checkpoint_data, checkpoint_path)            
    logger.info("Done training classifier for %s" % target_lang)

    # Clean up
    logger.info("Deleting %s" % path_data)
    cmd = ["rm", path_data]
    subprocess.call(cmd)

    return None
        
def main():
    parser = argparse.ArgumentParser()
    # Model and data are required
    parser.add_argument("--dir_pretrained_model",
                        type=str,
                        required=True,
                        help="Dir containing pre-trained model (checkpoint), which may have been fine-tuned already.")

    # Required for certain modes (--do_train, --eval_during_training, --do_eval or --do_pred)
    parser.add_argument("--dir_train",
                        type=str,
                        help=("Dir containing training data (n files named <lang>.train containing unlabeled text)"))
    parser.add_argument("--dir_output",
                        type=str,
                        help="Directory in which model will be written (required if --do_train (but not --resume) or --do_pred)")
    parser.add_argument("--path_dev",
                        type=str,
                        help="Path of 2-column TSV file containing labeled validation examples.")
    parser.add_argument("--path_test",
                        type=str,
                        help="Path of text file containing unlabeled test examples.")

    # Execution modes
    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume training model in --dir_pretrained_model (note: --dir_output will be ignored)")
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Run training")
    parser.add_argument("--eval_during_training",
                        action="store_true",
                        help="Run evaluation on dev set during training")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Evaluate model on dev set")
    parser.add_argument("--do_pred",
                        action="store_true",
                        help="Run prediction on test set")

    # Score to optimize on dev set (by early stopping)
    parser.add_argument("--score_to_optimize",
                        choices=["track1", "track2", "track3"],
                        default="track3",
                        help="Score to optimize on dev set during training (by early stopping).")
    
    # Hyperparameters
    parser.add_argument("--freeze_encoder",
                        action="store_true",
                        help="Freeze weights of pre-trained encoder.")
    parser.add_argument("--add_adapters",
                        action="store_true",
                        help="Add adapter layers between text encoding and classification layers.")
    parser.add_argument("--sampling_alpha",
                        type=float,
                        default=1.0,
                        help="Dampening factor for relative frequencies used to compute language sampling probabilities")
    parser.add_argument("--weight_relevant",
                        type=float,
                        default=1.0,
                        help="Relative sampling frequency of relevant languages wrt irrelevant languages")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--max_seq_len",
                        default=256,
                        type=int,
                        help="Length of input sequences. Shorter seqs are padded, longer ones are trucated")
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
    parser.add_argument("--num_train_epochs"
                        default=3,
                        type=int,
                        help="Number of training epochs.")
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

    # Distributed or parallel?
    if args.local_rank != -1 or args.num_gpus > 1:
        raise NotImplementedError("No distributed or parallel training available at the moment.")
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = 1
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0
    
    # Check execution mode
    assert args.resume or args.do_train or args.do_eval or args.do_pred
    if args.resume:
        assert not args.do_train
        assert not args.do_eval
        assert not args.do_pred
        
    # Load checkpoint. This contains a pre-trained model which may or
    # may not have been fine-tuned for language identification already
    logger.info("Loading checkpoint...")        
    checkpoint_path = os.path.join(args.dir_pretrained_model, "checkpoint.tar")        
    checkpoint_data = torch.load(checkpoint_path)
    if args.resume:
        # Check progress: what language were we training on when
        # we stopped?
        logger.info("Resuming training. Currently training classifier for %s" % checkpoint_data["current_lang"])
        # Replace args with initial args for this job, except for
        # num_gpus, seed and model directory
        current_num_gpus = args.n_gpu
        current_dir_pretrained_model = args.dir_pretrained_model
        args = deepcopy(checkpoint_data["initial_args"])
        args.num_gpus = current_num_gpus
        args.dir_pretrained_model = dir_pretrained_model
        args.resume = True
        logger.info("Args (most have been reloaded from checkpoint): %s" % args)
    else:
        if args.eval_during_training:
            assert args.do_train
        if args.do_train or args.do_pred:
            assert args.dir_output is not None
            if os.path.exists(args.dir_output) and os.path.isdir(args.dir_output) and len(os.listdir(args.dir_output)) > 1:
                msg = "%s already exists and is not empty" % args.dir_output
                raise ValueError(msg)
            if not os.path.exists(args.dir_output):
                os.makedirs(args.dir_output)
        if args.do_train:
            assert args.dir_train is not None
            checkpoint_data["initial_args"] = args

    if args.do_eval or args.eval_during_training:
        assert args.path_dev is not None
        assert os.path.exists(args.path_dev)
    if args.do_pred:
        assert args.path_test is not None
        assert os.path.exists(args.path_test)
    if args.grad_accum_steps < 1:
        raise ValueError("Invalid grad_accum_steps parameter: {}, should be >= 1".format(
                            args.grad_accum_steps))

    
    # Create list of languages we handle
    lang_list = sorted(ALL_LANGS)
        
    
    # Seed RNGs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_path = os.path.join(args.dir_pretrained_model, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Make encoder and model
    logger.info("Making encoder...")
    encoder_config = BertConfig.from_json_file(os.path.join(args.dir_pretrained_model, "config.json"))            
    encoder = BertForMaskedLM(encoder_config)
    logger.info("Making model...")
    model = BertForLangID(encoder, lang_list)
    model.to(args.device)
    
    # Load model weights. First, check if we just have an encoder, or a previously fine-tuned model
    if "classifier.dense.weight" in checkpoint_data["model_state_dict"]:
        if "best_model_state_dict" in checkpoint_data and not args.resume:
            logger.info("Loading model weights from 'best_model_state_dict'") 
            model.load_state_dict(checkpoint_data["best_model_state_dict"])
        else:
            logger.info("Loading model weights from 'model_state_dict'")             
            model.load_state_dict(checkpoint_data["model_state_dict"])            
    else:
        # Model has not previously been fine-tuned, so we only load encoder weights
        assert args.do_train
        logger.info("Loading encoder weights from 'model_state_dict'")                         
        model.encoder.load_state_dict(checkpoint_data["model_state_dict"])
    if (args.do_train or args.resume) and args.freeze_encoder:
        model.freeze_encoder()

    # Log some info on the model
    logger.info("Encoder config: %s" % repr(model.encoder.config))
    logger.info("Model params:")
    for n,p in model.named_parameters():
        msg = "  %s" % n
        if not p.requires_grad:
            msg += " ***FROZEN***"
        logger.info(msg)
    logger.info("Nb model params: %d" % count_params(model))
    logger.info("Nb params in encoder: %d" % count_params(model.encoder))    
    logger.info("Nb params in pooler: %d" % count_params(model.pooler))
    if args.add_adapters:
        logger.info("Nb params in adapters: %d" % count_params(model.adapter))
    logger.info("Nb params in classifier: %d" % count_params(model.classifier))
        

    # Get data
    dev_dataset = None
    if args.do_eval or args.eval_during_training:
        logger.info("Loading validation data from %s..." % args.path_dev)                                
        dev_dataset = BertDatasetForTesting(args.path_dev,
                                            tokenizer,
                                            model.lang2id,
                                            args.max_seq_len,
                                            require_labels=True,
                                            encoding="utf-8",
                                            verbose=DEBUG)
    if args.do_pred:
        logger.info("Loading test data from %s..." % args.path_test)                                
        test_dataset = BertDatasetForTesting(args.path_test,
                                             tokenizer,
                                             model.lang2id,
                                             args.max_seq_len,
                                             require_labels=False,
                                             encoding="utf-8",
                                             verbose=DEBUG)

    # Write config and tokenizer in output directory
    if (not args.resume) and args.do_train:
        path_config = os.path.join(args.dir_output, "config.json")
        model.encoder.config.to_json_file(path_config)
        path_tokenizer = os.path.join(args.dir_output, "tokenizer.pkl")
        with open(path_tokenizer, "wb") as f:
            pickle.dump(tokenizer, f)
            
    # Train
    if args.do_train or args.resume:
        # Add some data to checkpoint
        if not args.resume:
            checkpoint_data["global_step"] = {}
            checkpoint_data["nb_opt_steps"] = {}
        
        # Check which languages are left to process
        langs_left = lang_list[:]
        if args.resume:
            start = langs_left.index(checkpoint_data["current_lang"])
            langs_left = langs_left[start:]
            logger.info("Resuming training of classifier for '%s' (%d/%d)" % (target_lang,
                                                                              start+1,
                                                                              len(lang_list)))
        else:
            logger.info("Starting training of %d classifiers" % len(langs_left))            

        # Log some info before training
        logger.info("*** Training info: ***")
        logger.info("  Num epochs: %d" % args.num_train_epochs)
        logger.info("  Gradient accumulation steps: %d" % args.grad_accum_steps)
        logger.info("  Batch size: %d" % args.train_batch_size)
        if args.eval_during_training:
            logger.info("  Validation dataset size: %d" % len(dev_dataset))

        # Loop over the languages we haven't processed yet.
        for target_lang in langs_left:
            logger.info("Preparing to train classifier for '%s'" % target_lang)
            checkpoint_data["current_lang"] = target_lang
            
            # Prepare optimizer
            logger.info("  Preparing optimizer")
            np_list = list(model.named_parameters())
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
                
            # Load optimizer state if resuming
            if args.resume:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
            # Run training on current target language
            train(model,
                  optimizer,
                  tokenizer,
                  target_lang,
                  args,
                  checkpoint_data,
                  dev_dataset=dev_dataset)
        logger.info("Finised training %d classifiers" % len(lang_list))
        
        # Reload model after training all classifiers
        save_to_dir = args.dir_pretrained_model if args.resume else args.dir_output
        checkpoint_data = torch.load(os.path.join(save_to_dir, "checkpoint.tar"))
        if "best_model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["best_model_state_dict"])
        else:
            model.load_state_dict(checkpoint_data["model_state_dict"])
        
    # Evaluate model on dev set
    if args.do_eval:
        logger.info("*** Running evaluation... ***")
        scores = evaluate(model, dev_dataset, args)
        logger.info("***** Evaluation Results *****")
        for score_name in sorted(scores.keys()):
            logger.info("- %s: %.4f" % (score_name, scores[score_name]))

    # Get model's predictions on test set
    if args.do_pred:
        logger.info("*** Running prediction... ***")        
        logits = predict(model, test_dataset, args)
        pred_class_ids = np.argmax(logits.cpu().numpy(), axis=1)
        pred_labels = [test_dataset.label_list[i] for i in pred_class_ids]
        path_pred = os.path.join(args.dir_output, "pred.txt")
        logger.info("Writing predictions in %s..." % path_pred)
        with open(path_pred, 'w', encoding="utf-8") as f:
            for x in pred_labels:
                f.write("%s\n" % x)
        
if __name__ == "__main__":
    main()
