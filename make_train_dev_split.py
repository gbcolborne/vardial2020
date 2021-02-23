""" Make train/dev/test split. """

import os, argparse, logging
import numpy as np
from comp_utils import stream_sents, ALL_LANGS, RELEVANT_LANGS, IRRELEVANT_LANGS, IRRELEVANT_URALIC_LANGS


def log_stats(numbers, logger):
    logger.info("Count: {}".format(len(numbers)))
    logger.info("Min: {}".format(min(numbers)))
    logger.info("Mean: {}".format(np.mean(numbers)))
    logger.info("Max: {}".format(max(numbers)))
    logger.info("Sum: {}".format(sum(numbers)))    
    logger.info("Median: {}".format(np.median(numbers)))    
    logger.info("# zeros: {}".format(sum(1 for x in numbers if x==0)))

    
def compute_sampling_probs(data_dir, alpha=1.0, rel_weight=1.0, logger=None):
    assert alpha >= 0 and alpha <= 1
    lang2prob = {}
    # We compute the sampling probabilities of the relevant and
    # irrelevant languages independently.
    rel_langs = sorted(RELEVANT_LANGS)
    irr_langs = sorted(IRRELEVANT_LANGS)
    if logger:
        logger.info("Computing sampling probabilities for relevant languages...")
    rel_probs = compute_sampling_probs_for_subgroup(rel_langs, data_dir, alpha=alpha, logger=logger)
    if logger:
        logger.info("Computing sampling probabilities for irrelevant languages...")
    irr_probs = compute_sampling_probs_for_subgroup(irr_langs, data_dir, alpha=alpha, logger=logger)
    # Weight the distribution of relevant languages, then renormalize
    rel_probs = rel_probs * rel_weight
    sum_of_both = rel_probs.sum() + irr_probs.sum()
    rel_probs = rel_probs / sum_of_both
    irr_probs = irr_probs / sum_of_both
    for lang, prob in zip(rel_langs, rel_probs):
        lang2prob[lang] = prob
    for lang, prob in zip(irr_langs, irr_probs):
        lang2prob[lang] = prob
    if logger:
        title = "Stats on sampling probabilities for relevant languages"            
        log_title_with_border(title, logger)
        log_stats(rel_probs, logger)
        title = "Stats on sampling probabilities for irrelevant languages"            
        log_title_with_border(title, logger)
        log_stats(irr_probs, logger)            
    return lang2prob


def compute_sampling_probs_for_subgroup(lang_list, data_dir, alpha=1.0, logger=None):
    assert alpha >= 0 and alpha <= 1
    if len(lang_list) == 1:
        return [1]
    lang2freq = {}
    for lang in lang_list:
        if logger: 
            logger.info("  %s" % lang)
        lang2freq[lang] = sum(1 for (sent, text_id, url) in stream_sents(lang,
                                                                         data_dir,
                                                                         input_format="text-only"))
    counts = np.array([lang2freq[k] for k in lang_list], dtype=np.float)
    probs = counts / counts.sum()
    probs_damp = probs ** alpha
    probs = probs_damp / probs_damp.sum()
    return probs


def log_title_with_border(title, logger):
    title = "--- %s ---" % title
    line = "-" * len(title)
    logger.info(line)
    logger.info(title)
    logger.info(line)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_alpha", type=float, default=1.0,
                        help="Frequency dampening factor used for computing language sampling probabilities")
    parser.add_argument("--weight_relevant", type=float, default=1.0,
                        help=("Relative sampling frequency of relevant languages wrt irrelevant languages."
                              " Default is 1, which produces a balanced mix of relevant and irrelevant."))
    parser.add_argument("dev_size", type=int, 
                        help="Number of examples in dev set (must be greater than 0)")
    parser.add_argument("test_size", type=int,
                        help="Number of examples in test set (can be 0)")
    parser.add_argument("input_dir", help=("Path of directory containing training data (n files named <lang>.train,"
                                           " containing unlabeled text only (no labels, URLS or text IDs)"))
    parser.add_argument("output_dir")
    args = parser.parse_args()
    
    # Check args
    assert args.dev_size > 0
    assert args.test_size >= 0
    assert args.sampling_alpha >= 0 and args.sampling_alpha <= 1
    assert not os.path.exists(args.output_dir)
    os.makedirs(args.output_dir)
    outdir_train = os.path.join(args.output_dir, "Training")
    outdir_test = os.path.join(args.output_dir, "Test")        
    os.makedirs(outdir_train)
    os.makedirs(outdir_test)
        
    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.DEBUG)

    # We expect that the input dir contains n files called lang.train,
    # which contain unlabeled text (without labels, URLS or text IDs)
    filenames = [n for n in os.listdir(args.input_dir) if n[-6:] == ".train"]
    logger.info("Nb training files found: %d" % len(filenames))
    for n in filenames:
        lang = n[:-6]
        assert lang in ALL_LANGS
    
    # Seed RNG
    np.random.seed(91500)
    
    # Get language sampling probabilities
    lang2prob = compute_sampling_probs(args.input_dir,
                                       alpha=args.sampling_alpha,
                                       rel_weight=args.weight_relevant,
                                       logger=logger)

    # Sample languages and count
    all_langs = sorted(ALL_LANGS)
    sampling_probs = [lang2prob[k] for k in all_langs]
    dev_sample = np.random.choice(np.arange(len(all_langs)),
                              size=args.dev_size,
                              replace=True,
                              p=sampling_probs)
    dev_counts = [0 for k in all_langs]
    for lang_id in dev_sample:
        dev_counts[lang_id] += 1
    if args.test_size > 0:
        test_sample = np.random.choice(np.arange(len(all_langs)),
                                       size=args.test_size,
                                       replace=True,
                                       p=sampling_probs)
        test_counts = [0 for k in all_langs]            
        for lang_id in test_sample:
            test_counts[lang_id] += 1
        
    # Print stats on distributions of the dev and test sets. Show min,
    # max, mean and median. Then do the same for RELEVANT, CONFOUNDING
    # AND IRRELEVANT.
    title = "Stats on # dev samples (all languages)"
    log_title_with_border(title, logger)
    log_stats(dev_counts, logger)
    title = "Stats on # dev samples (relevant languages)"
    log_title_with_border(title, logger)
    rel_counts = [dev_counts[i] for i in range(len(dev_counts)) if all_langs[i] in RELEVANT_LANGS]
    log_stats(rel_counts, logger)
    title = "Stats on # dev samples (irrelevant languages)"
    log_title_with_border(title, logger)
    irr_counts = [dev_counts[i] for i in range(len(dev_counts)) if all_langs[i] in IRRELEVANT_LANGS]    
    log_stats(irr_counts, logger)
    title = "Stats on # dev samples (irrelevant Uralic languages)"
    log_title_with_border(title, logger)
    con_counts = [dev_counts[i] for i in range(len(dev_counts)) if all_langs[i] in IRRELEVANT_URALIC_LANGS]        
    log_stats(con_counts, logger)
    if args.test_size > 0:
        title = "Stats on # test samples (all languages)"
        log_title_with_border(title, logger)
        log_stats(test_counts, logger)
        title = "Stats on # test samples (relevant languages)"
        log_title_with_border(title, logger)
        rel_counts = [test_counts[i] for i in range(len(test_counts)) if all_langs[i] in RELEVANT_LANGS]
        log_stats(rel_counts, logger)
        title = "Stats on # test samples (irrelevant languages)"
        log_title_with_border(title, logger)
        irr_counts = [test_counts[i] for i in range(len(test_counts)) if all_langs[i] in IRRELEVANT_LANGS]    
        log_stats(irr_counts, logger)
        title = "Stats on # test samples (irrelevant Uralic languages)"
        log_title_with_border(title, logger)
        con_counts = [test_counts[i] for i in range(len(test_counts)) if all_langs[i] in IRRELEVANT_URALIC_LANGS]        
        log_stats(con_counts, logger)
    
    
    # Write training data in separate, unlabeled text files. Store dev
    # and test examples (to shuffle later, to avoid writing them in
    # order of language)
    dev_set = []
    test_set = []
    logger.info("Writing training data in %s..." % (outdir_train))    
    for lang_id, lang in enumerate(all_langs):
        logger.info("  %s" % lang)
        # Get number of examples
        nb_examples = sum(1 for (sent, text_id, url) in stream_sents(lang, args.input_dir, input_format="text-only"))
        
        # Sample dev and test indices
        indices = np.arange(nb_examples)
        np.random.shuffle(indices)
        nb_dev = dev_counts[lang_id]
        nb_test = test_counts[lang_id]
        dev_indices = set(indices[:nb_dev])
        test_indices = set(indices[nb_dev:nb_dev+nb_test])
        
        # Stream sents, write training examples, store others
        outpath = os.path.join(outdir_train, "%s.train" % (lang))
        with open(outpath, 'w') as outfile:
            for ix, (sent, text_id, url) in enumerate(stream_sents(lang,
                                                                   args.input_dir,
                                                                   input_format="text-only")):
                if ix in dev_indices:
                    dev_set.append((sent, lang))
                elif ix in test_indices:
                    test_set.append((sent, lang))
                else:
                    outfile.write(sent + "\n")

    # Shuffle and write dev and test sets
    logger.info("Writing test data in %s..." % (outdir_test))
    np.random.shuffle(dev_set)
    ptexts = os.path.join(outdir_test, "dev.txt")
    plabels = os.path.join(outdir_test, "dev-gold-labels.txt")    
    ptuples = os.path.join(outdir_test, "dev-labeled.tsv")
    with open(ptexts, 'w') as ftexts, open(plabels, 'w') as flabels, open(ptuples, 'w') as ftuples:
        for (text, lang) in dev_set:
            ftexts.write(text + "\n")
            flabels.write(lang + "\n")
            ftuples.write("%s\t%s\n" % (text, lang))
    if len(test_set):
        np.random.shuffle(test_set)
        ptexts = os.path.join(outdir_test, "test.txt")
        plabels = os.path.join(outdir_test, "test-gold-labels.txt")    
        ptuples = os.path.join(outdir_test, "test-labeled.tsv")
        with open(ptexts, 'w') as ftexts, open(plabels, 'w') as flabels, open(ptuples, 'w') as ftuples:
            for (text, lang) in test_set:
                ftexts.write(text + "\n")
                flabels.write(lang + "\n")
                ftuples.write("%s\t%s\n" % (text, lang))
        
if __name__ == "__main__":
    main()
