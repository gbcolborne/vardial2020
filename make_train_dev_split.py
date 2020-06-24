""" Make dev set """

import os, argparse, random, logging
import numpy as np
from comp_utils import map_ULI_langs_to_paths, stream_sents, RELEVANT_LANGS, IRRELEVANT_URALIC_LANGS
from iteround import saferound

parser = argparse.ArgumentParser()
parser.add_argument("rel_size", type=int, 
                    help="Number of relevant sentences in dev set")
parser.add_argument("con_size", type=int, 
                    help="Number of confounding (i.e. irrelevan Uralic) sentences in dev set")
parser.add_argument("irr_size", type=int, 
                    help="Number of irrelevant sentences in dev set")
parser.add_argument("--balance-rel", action="store_true",
                    help="Balance the 29 relevant languages.")
parser.add_argument("--balance-con", action="store_true",
                    help="Balance the 3 confounding languages (i.e. irrelevant Uralic)")
parser.add_argument("--balance-irr", action="store_true",
                    help="Balance the 146 irrelevant languages (excluding Uralic confounders)")
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
con_langs = sorted(IRRELEVANT_URALIC_LANGS)
irr_langs = sorted(set(langs).difference(rel_langs + con_langs))

# Map langs to IDs
rel_lang2id = dict((l,i) for (i,l) in enumerate(rel_langs))
con_lang2id = dict((l,i) for (i,l) in enumerate(con_langs))
irr_lang2id = dict((l,i) for (i,l) in enumerate(irr_langs))

# Compute expected distribution of dev set based on training data
logger.info("Computing expected distribution of dev set based on training data...")
lang2freq = {}
for lang in langs:
    logger.info("  " + lang)
    lang2freq[lang] = sum(1 for (sent, url) in stream_sents(lang))
if args.balance_rel:
    rel_probs = np.ones(len(rel_langs), dtype=float) / len(rel_langs)
else:
    rel_counts = np.array([lang2freq[k] for k in rel_langs], dtype=np.float)
    rel_probs = rel_counts / rel_counts.sum()
if args.balance_con:
    con_probs = np.ones(len(con_langs), dtype=float) / len(con_langs)
else:
    con_counts = np.array([lang2freq[k] for k in con_langs], dtype=np.float)
    con_probs = con_counts / con_counts.sum()
if args.balance_irr:
    irr_probs = np.ones(len(irr_langs), dtype=float) / len(irr_langs)
else:
    irr_counts = np.array([lang2freq[k] for k in irr_langs], dtype=np.float)
    irr_probs = irr_counts / irr_counts.sum()

# Compute expected count of dev sentences. Use a sum-safe rounding function.
rel_dev_counts = [int(x) for x in saferound(rel_probs * args.rel_size, 0, "largest")]
con_dev_counts = [int(x) for x in saferound(con_probs * args.con_size, 0, "largest")]
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

    # Is this a relevant, confounding or irrelevant lang?
    if lang in rel_lang2id:
        nb_dev = rel_dev_counts[rel_lang2id[lang]]
    elif lang in con_lang2id:
        nb_dev = con_dev_counts[con_lang2id[lang]]
    elif lang in irr_lang2id:
        nb_dev = irr_dev_counts[irr_lang2id[lang]]
    nb_train = nb_sents - nb_dev
    logger.info("  %s (%d/%d)" % (lang, nb_train, nb_dev))

    # Check if we have to make a dev
    dev_empty = nb_dev == 0
    if not dev_empty:
        # Sample dev indices
        dev_indices = np.random.choice(np.arange(nb_sents,  dtype=int), size=nb_dev, replace=False)
        sorted_dev_indices = sorted(dev_indices)
        next_dev_ix = sorted_dev_indices.pop(0)

    # Write split
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
                
