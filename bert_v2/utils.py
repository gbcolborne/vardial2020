import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from BertDataset import BertDatasetForTesting


def get_dataloader(dataset, batch_size, local_rank):
    """ Get data loader. """
    if isinstance(dataset, BertDatasetForTesting):
        if local_rank == -1:
            sampler = SequentialSampler(dataset)            
        else:
            sampler = DistributedSampler(dataset, shuffle=false)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        # BertDatasetForTraining is an IterableDataset, so we don't use a sampler
        return Dataloader(dataset, batch_size=batch_size)
    

def accuracy(pred_scores, labels, ignore_label=None):
    """ Compute accuracy. """
    ytrue = labels.cpu().numpy()
    assert len(ytrue.shape) == 1    
    ypred = pred_scores.detach().cpu().numpy()
    ypred = np.argmax(ypred, axis=1)
    assert len(ypred.shape) == 1
    assert ypred.shape == ytrue.shape
    if ignore_label is not None:
        keep = np.where(ytrue != ignore_label)
        ytrue = ytrue[keep]
        ypred = ypred[keep]
    nb_correct = np.sum(ypred == ytrue)
    accuracy = nb_correct / ytrue.shape[0]
    return accuracy


def count_params(model):
    """ Count params in model. """
    count = 0
    for p in model.parameters():
         count += torch.prod(torch.tensor(p.size())).item()
    return count


def weighted_avg(vals, weights):
    """ Compute weighted average. """
    vals = np.asarray(vals)
    weights = np.asarray(weights)
    assert len(vals.shape) == 1
    assert vals.shape == weights.shape
    probs = weights / weights.sum()
    return np.sum(vals * probs)    


def adjust_loss(loss, args):
    """ Adapt loss for distributed training or gradient accumulation. """
    if args.n_gpu > 1:
        loss = loss.mean() # mean() to average on multi-gpu.
    if args.grad_accum_steps > 1:
        loss = loss / args.grad_accum_steps
    return loss


def check_for_unk_train_data(train_paths):
    """ Check for a file named `unk.train`, containing unlabeled data. """
    for path in train_paths:
        if os.path.split(path)[-1] == "unk.train":
            return path
    return None

