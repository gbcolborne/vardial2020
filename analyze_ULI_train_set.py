""" Analyze ULI training data language by language. """

import argparse
from collections import defaultdict, Counter
import statistics as stats
from url_utils import get_netloc, get_toplevel_domains
from comp_utils import map_ULI_langs_to_paths, stream_sents, IRRELEVANT_URALIC_LANGS, RELEVANT_LANGS


def print_title_with_border(title):
    title = "--- %s ---" % title
    line = "-" * len(title)
    print("\n%s\n%s\n%s\n" % (line, title, line))


def print_stats(vals, max_thresholds=None):
    print("- Count: {}".format(len(vals)))
    print("- Min: {}".format(min(vals)))
    print("- Max: {}".format(max(vals)))
    print("- Mean: {}".format(sum(vals)/len(vals)))
    print("- Median: {}".format(stats.median(vals)))
    if max_thresholds:
        for threshold in max_thresholds:
            print_count_gt_threshold(vals, threshold)

    
def print_count_gt_threshold(vals, threshold):
    count = sum(1 for x in vals if x > threshold)
    pct = 100 * count / len(vals)
    print("- # texts with length > {}: {} ({:.2f}%)".format(threshold, count, pct))


def analyze_corpus_sizes(langs):
    corpus_sizes = []
    print()
    for i, lang in enumerate(langs):            
        print("{}/{}. {}".format(i+1, len(langs), lang))
        nb_sents = sum(1 for (text, url) in stream_sents(lang))
        corpus_sizes.append(nb_sents)
    size_fd = Counter(corpus_sizes)
    print_title_with_border("Corpus size (freq)")    
    for (size, count) in sorted(size_fd.items(), key=lambda x:x[0], reverse=True):
        print("- {} ({})".format(size, count))
    print_title_with_border("Summary of corpus sizes")    
    print_stats(list(corpus_sizes))    

    
def analyze_text_lengths(langs):
    text_lengths = []
    max_length_thresholds = [64, 128, 256, 512]
    for i, lang in enumerate(langs):            
        lengths = [len(text) for (text, url) in stream_sents(lang)]
        title = "{}/{}. {}".format(i+1, len(langs), lang)
        print_title_with_border(title)
        print_stats(lengths, max_thresholds=max_length_thresholds)        
        text_lengths.append(lengths)

    # Print some summary stats
    all_text_lengths = []
    for x in text_lengths:
        all_text_lengths += x
    print_title_with_border("Summary")    
    print_stats(all_text_lengths)
    for threshold in [64,128,256,512]:
        print_count_gt_threshold(all_text_lengths, threshold)
    

def analyze_words_chars_urls(langs):
    vocab_sizes = []
    alphabet_sizes = []
    spu_ratios = []
    for i, lang in enumerate(langs):
        char2freq = defaultdict(int)
        word2freq = defaultdict(int)
        uniq_urls = set()
        nb_sents = 0
        for (text, url) in stream_sents(lang):
            nb_sents += 1
            if url:
                uniq_urls.add(url)
            # Whitespace-tokenize text
            for word in text.split(" "):
                word2freq[word] += 1
                for char in word:
                    char2freq[char] += 1
        if len(uniq_urls):
            spu_ratio = nb_sents/len(uniq_urls)
            spu_ratios.append(spu_ratio)
        nb_tokens = sum(word2freq.values())
        vocab_size = len(word2freq)
        vocab_sizes.append(vocab_size)        
        alphabet_size = len(char2freq)
        alphabet_sizes.append(alphabet_size)
        print("\n--- {}/{}. {} ---".format(i+1, len(langs), lang))            
        print("Nb tokens: {}".format(nb_tokens))
        print("Vocab size: {}".format(vocab_size))
        print("Alphabet size: {}".format(alphabet_size))
        if len(uniq_urls):
            print("Nb unique URLs: {}".format(len(uniq_urls)))
            print("Sents/URL ratio: {:f}".format(spu_ratio))

    # Print some summary stats
    to_analyze = [("Vocab sizes", vocab_sizes), ("Alphabet sizes", alphabet_sizes)]
    if len(spu_ratios):
        to_analyze.append(("Sents/URL ratios", spu_ratios))
    for (statname, vals) in to_analyze:
        print_title_with_border(statname) 
        print_stats(vals)
    return

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("langs", choices=["french", "irrelevant-uralic", "relevant", "irrelevant", "all"])
    parser.add_argument("analysis", choices=["corpus-sizes", "text-lengths", "words-chars-urls"])
    args = parser.parse_args()

    # Get all languages and their path
    lang2path = map_ULI_langs_to_paths()

    # Check which languages we are processing
    if args.langs == "all":
        langs = list(lang2path.keys())
    elif args.langs == "relevant":
        langs = list(RELEVANT_LANGS)
    elif args.langs == "irrelevant":
        all_langs = set(lang2path.keys())
        langs = all_langs.difference(RELEVANT_LANGS)
    elif args.langs == "irrelevant-uralic":
        langs = list(IRRELEVANT_URALIC_LANGS)
    elif args.langs == "french":
        langs = ['fra']
    print("\nLangs ({}): {}".format(len(langs), langs))

    # Run
    if args.analysis == "corpus-sizes":
        analyze_corpus_sizes(langs)
    elif args.analysis == "text-lengths":
        analyze_text_lengths(langs)
    elif args.analysis == "words-chars-urls":
        analyze_words_chars_urls(langs)
    print("\n\n")
    return


if __name__ == "__main__":
    main()
