""" Make dev set """

import os, argparse, random, logging
import numpy as np
from comp_utils import map_ULI_langs_to_paths, stream_sents, RELEVANT_LANGS

REL_SUBSET_SIZE = 1000  # Nb sets in the relevant portion of the dev set
IRR_SUBSET_SIZE = 1000  # Nb sets in the irrelevant portion of the dev set

parser = argparse.ArgumentParser()
parser.add_argument("--balance-irrelevant", "-b", action="store_true",
                    help="Balance irrelevant languages.")
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

# Compute class frequencies
lang2freq = {}
logger.info("Computing class frequencies...")
for lang in langs:
    logger.info("  " + lang)
    lang2freq[lang] = sum(1 for (sent, url) in stream_sents(lang))

# Compute sampling probabilities (separately for each of the 2 groups)
rel_counts = np.array([lang2freq[k] for k in rel_langs], dtype=np.float32)
rel_probs = rel_counts / rel_counts.sum()
if args.balance_irrelevant:
    irr_probs = np.ones(len(irr_langs), dtype=np.float32) / len(irr_langs)
else:
    irr_counts = np.array([lang2freq[k] for k in irr_langs], dtype=np.float32)
    irr_probs = irr_counts / irr_counts.sum()
for subsize in [500, 1000, 2500, 5000, 10000, 25000, 50000]:
    rel_cum_probs = rel_probs * subsize
    print(rel_cum_probs)
    irr_cum_probs = irr_probs * subsize
    print(irr_cum_probs)
    rel_excluded = (rel_cum_probs < 0.5).sum()
    irr_excluded = (irr_cum_probs < 0.5).sum()
    print("{}\t{}\t{}".format(subsize, rel_excluded, irr_excluded))
sys.exit()

# Make train-dev split
lang2dev = dict([(lang,[]) for lang in langs])
logger.info("Sampling dev set...")
for i in range(MAX_SENTS):
    # Sample a language
    lang = random.choice(langs)
    # Sample a sentence
    sent_id = random.randint(0,lang2size[lang]-1)
    lang2dev[lang].append(sent_id)
        
    # Now loop over data again and write split
    data = []
    logger.info("Writing train-dev split --> %s..." % args.output_dir)
    path_train = os.path.join(args.output_dir, "train.tsv")
    path_dev = os.path.join(args.output_dir, "valid.tsv")
    f_train = open(path_train, 'w')
    f_dev = open(path_dev, 'w')
    for lang in langs:
        logger.info("  " + lang)
        dev_ids = sorted(lang2dev[lang])
        for i,(sent, url) in enumerate(stream_sents(lang)):
            # Remove any tabs from text, as data is written in tsv
            sent = sent.replace("\t", " ")
            # Write
            example = "%s\t%s\n" % (sent,lang)
            if len(dev_ids) and i == dev_ids[0]:
                f_dev.write(example)
                dev_ids.pop(0)
            else:
                f_train.write(example)
    f_train.close()
    f_dev.close()

    logger.warning(" !!! YOU SHOULD SHUFFLE THE DATASET, AS IT IS ORDERED BY LANG !!! ")
    logger.warning(" !!! YOU SHOULD SHUFFLE THE DATASET, AS IT IS ORDERED BY LANG !!! ")
    logger.warning(" !!! YOU SHOULD SHUFFLE THE DATASET, AS IT IS ORDERED BY LANG !!! ")    
                
