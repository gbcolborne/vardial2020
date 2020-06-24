""" Analyze ULI training data language by language. """

import argparse
from collections import defaultdict, Counter
import statistics as stats
from url_utils import get_netloc, get_toplevel_domains
from comp_utils import map_ULI_langs_to_paths, stream_sents, IRRELEVANT_URALIC_LANGS, RELEVANT_LANGS

parser = argparse.ArgumentParser()
parser.add_argument("langs", choices=["french", "irrelevant-uralic", "relevant", "irrelevant", "all"])
parser.add_argument("--corpus_sizes_only", action="store_true")
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

corpus_sizes = []
alphabet_sizes = []
vocab_sizes = []
spu_ratios = []    
for i, lang in enumerate(langs):
    if args.corpus_sizes_only:
        print("{}/{}. {}".format(i+1, len(langs), lang))
        nb_sents = sum(1 for (text, url) in stream_sents(lang))
        corpus_sizes.append(nb_sents)
        continue
    print("\n---({}) {}---".format(i+1, lang))
    char2freq = defaultdict(int)
    word2freq = defaultdict(int)
    if lang in RELEVANT_LANGS:
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
    corpus_sizes.append(nb_sents)

    if lang in RELEVANT_LANGS:
        spu_ratio = nb_sents/len(uniq_urls)
        spu_ratios.append(spu_ratio)

    nb_tokens = sum(word2freq.values())
    vocab_size = len(word2freq)
    alphabet_size = len(char2freq)
    alphabet_sizes.append(alphabet_size)
    vocab_sizes.append(vocab_size)
    print("Nb sents: {}".format(nb_sents))
    print("Nb tokens: {}".format(nb_tokens))
    print("Vocab size: {}".format(vocab_size))
    print("Alphabet size: {}".format(alphabet_size))
    if lang in RELEVANT_LANGS:
        print("Nb unique URLs: {}".format(len(uniq_urls)))
        print("Sents/URL ratio: {:f}".format(spu_ratio))


# Print some summary stats
if args.corpus_sizes_only:
    print("\n\nCorpus size (freq):")
    size_fd = Counter(corpus_sizes)
    for (size, count) in sorted(size_fd.items(), key=lambda x:x[0], reverse=True):
        print("- {} ({})".format(size, count))
    print("\n\nSummary:")
    print("- Min: {}".format(min(corpus_sizes)))
    print("- Max: {}".format(max(corpus_sizes)))
    print("- Mean: {}".format(sum(corpus_sizes)/len(corpus_sizes)))
    print("- Median: {}".format(stats.median(corpus_sizes)))
else:
    title = '------ SUMMARY ------'
    print("\n")
    print("-"*len(title))
    print(title)
    print("-"*len(title))
    to_analyze = [("Vocab size", vocab_sizes), ("Alphabet size", alphabet_sizes)]
    if args.langs in ["all", "relevant"]:
        to_analyze.append(("Sents/URL ratio", spu_ratios))
    for (statname, vals) in to_analyze:
        print("\n----- %s -----" % statname)
        print("Min: {}".format(min(vals)))
        print("Max: {}".format(max(vals)))
        print("Mean: {}".format(sum(vals)/len(vals)))
        print("Median: {}".format(stats.median(vals)))
    print("\n\n")
