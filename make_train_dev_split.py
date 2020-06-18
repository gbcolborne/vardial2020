""" Make dev set """

import argparse, random, logging
from comp_utils import map_ULI_langs_to_paths, stream_sents


SENTS_PER_LANG = 5  # Used for "balanced-by-lang" dev set
MAX_SENTS = 1000    # Used for all other dev set structures

parser = argparse.ArgumentParser()
parser.add_argument("--structure", "-s",
                    choices=["unbalanced", "balanced-by-lang", "balanced-by-group", "double-balanced"],
                    default="unbalanced",
                    help="How to structure the dev set.")
args = parser.parse_args()

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
    logger.info("Counting sents...")
    for lang in langs:
        logger.info("  " + lang)
        lang2size[lang] = sum(1 for sent in stream_sents(lang))

    # Now do random sampling
    lang2samples = {}
    logger.info("Sampling sentences...")
    for i in range(MAX_SENTS):
        # Sample a language
        lang = random.choice(langs)
        if lang not in lang2samples:
            lang2samples[lang] = []

        # Sample a sentence
        sent_id = random.randint(0,lang2size[lang]-1)
        lang2samples[lang].append(sent_id)
        
    # Retrieve sampled sentences
    data = []
    logger.info("Retrieving sampled sentences...")
    for lang, sent_ids in lang2samples.items():
        sorted_sent_ids = sorted(sent_ids)
        for i,sent in enumerate(stream_sents(lang)):
            if i == sorted_sent_ids[0]:
                data.append((sent, lang))
                logger.info("  %d" % len(data))                
                sorted_sent_ids.pop(0)
                if len(sorted_sent_ids) == 0:
                    # Stop streaming sentences, as we have all our samples
                    break

    for (x,y) in data:
        logger.info("%s\t%s" % (y,x))
