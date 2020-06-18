""" Make dev set """

import os, argparse, random, logging
from comp_utils import map_ULI_langs_to_paths, stream_sents


SENTS_PER_LANG = 5  # Used for "balanced-by-lang" dev set
MAX_SENTS = 1000    # Used for all other dev set structures

parser = argparse.ArgumentParser()
parser.add_argument("--structure", "-s",
                    choices=["unbalanced", "balanced-by-lang", "balanced-by-group", "double-balanced"],
                    default="unbalanced",
                    help="How to structure the dev set.")
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

# Build dev set
if args.structure == "unbalanced":
    # Do 2 passes over the training files to avoid storing all the
    # sentences. Start by counting the number of sentences per
    # language
    lang2size = {}
    logger.info("Counting sentences...")
    for lang in langs:
        logger.info("  " + lang)
        lang2size[lang] = sum(1 for (sent, url) in stream_sents(lang))

    # Now do random sampling
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
                
