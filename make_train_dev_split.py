""" Make dev set """

import os, argparse, random, logging
import numpy as np
from comp_utils import map_ULI_langs_to_paths, stream_sents, RELEVANT_LANGS, IRRELEVANT_URALIC_LANGS
from iteround import saferound

parser = argparse.ArgumentParser()
parser.add_argument("--balance-irrelevant", action="store_true",
                    help="Balance irrelevant languages.")
parser.add_argument("--rel_size", type=int, default=6000,
                    help="Number of relevant sentences in dev set")
parser.add_argument("--irr_size", type=int, default=4000,
                    help="Number of irrelevant sentences in dev set")
parser.add_argument("output_dir")
args = parser.parse_args()

# Check args
if os.path.exists(args.output_dir):
    assert os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) == 0
else:
    os.makedirs(args.output_dir)
    
# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.DEBUG)

# Seed RNG
random.seed(91500)

# Get all langs
langs = sorted(map_ULI_langs_to_paths().keys())
rel_langs = sorted(RELEVANT_LANGS)
irr_langs = sorted(set(langs).difference(rel_langs))

# Map langs to IDs
rel_lang2id = dict((l,i) for (i,l) in enumerate(rel_langs))
irr_lang2id = dict((l,i) for (i,l) in enumerate(irr_langs))

# Compute expected distribution of dev set based on training data
logger.info("Computing expected distribution of dev set based on training data...")
lang2freq = {}
for lang in langs:
    logger.info("  " + lang)
    lang2freq[lang] = sum(1 for (sent, url) in stream_sents(lang))
rel_counts = np.array([lang2freq[k] for k in rel_langs], dtype=np.float)
rel_probs = rel_counts / rel_counts.sum()
if args.balance_irrelevant:
    irr_probs = np.ones(len(irr_langs), dtype=float) / len(irr_langs)
else:
    irr_counts = np.array([lang2freq[k] for k in irr_langs], dtype=np.float)
    irr_probs = irr_counts / irr_counts.sum()
# Compute expected count of dev sentences. Use a sum-safe rounding function.
rel_dev_counts = [int(x) for x in saferound(rel_probs * args.rel_size, 0, "largest")]
irr_dev_counts = [int(x) for x in saferound(irr_probs * args.irr_size, 0, "largest")]

def build_example(text, lang):
    return "%s\t%s" % (text, lang)

# Make train/dev split
logger.info("Writing train-dev split --> %s..." % args.output_dir)
path_train = os.path.join(args.output_dir, "train.tsv")
path_dev = os.path.join(args.output_dir, "valid.tsv")
f_train = open(path_train, 'w')
f_dev = open(path_dev, 'w')
logger.info("  ----------------")
logger.info("  lang (train/dev)")
logger.info("  ----------------")
#for lang in langs:
for lang in langs:
    nb_sents = int(lang2freq[lang])
    all_indices = list(range(nb_sents))
    if lang in rel_lang2id:
        nb_dev = rel_dev_counts[rel_lang2id[lang]]
    else:
        nb_dev = irr_dev_counts[irr_lang2id[lang]]
    nb_train = nb_sents - nb_dev
    logger.info("  %s (%d/%d)" % (lang, nb_train, nb_dev))
    dev_empty = nb_dev == 0
    if not dev_empty:
        dev_indices = np.random.choice(np.arange(nb_sents,  dtype=int), size=nb_dev, replace=False)
        sorted_dev_indices = sorted(dev_indices)
        next_dev_ix = sorted_dev_indices.pop(0)
    nb_dev_written = 0
    for i, (text,url) in enumerate(stream_sents(lang)):
        example = build_example(text, lang) + "\n"
        if dev_empty:
            f_train.write(example)
        else:
            if i == next_dev_ix:
                f_dev.write(example)
                nb_dev_written += 1
                if len(sorted_dev_indices) == 0:
                    dev_empty = True
                else:
                    next_dev_ix = sorted_dev_indices.pop(0)
            else:
                f_train.write(example)
    if nb_dev_written != nb_dev:
        raise ValueError("Expected {} but got {}.".format(nb_dev, nb_dev_written))
f_train.close()
f_dev.close()

logger.warning(" !!! YOU SHOULD SHUFFLE THE DATASET, AS IT IS ORDERED BY LANG !!! ")
logger.warning(" !!! YOU SHOULD SHUFFLE THE DATASET, AS IT IS ORDERED BY LANG !!! ")
logger.warning(" !!! YOU SHOULD SHUFFLE THE DATASET, AS IT IS ORDERED BY LANG !!! ")    
                
