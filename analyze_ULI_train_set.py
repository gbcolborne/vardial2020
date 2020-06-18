""" Analyze ULI training data language by language. """

import argparse
from collections import defaultdict
import statistics as stats
from url_utils import get_netloc, get_toplevel_domains
from comp_utils import RELEVANT_LANGS, map_ULI_langs_to_paths, extract_text_and_url 

parser = argparse.ArgumentParser()
parser.add_argument("--include_irrelevant", action='store_true', help="Include irrelevant languages (takes much longer)")
parser.add_argument("--french_only", action='store_true', help="Analyze French only")
args = parser.parse_args()

lang2path = map_ULI_langs_to_paths()

# Check which languages we are processing
if args.french_only:
    langs = ['fra']
elif args.include_irrelevant:
    langs = list(lang2path.keys())
else:
    langs = list(RELEVANT_LANGS)

alphabet_sizes = []
vocab_sizes = []
spu_ratios = []    
for i, lang in enumerate(langs):
    print("\n---({}) {}---".format(i+1, lang))
    char2freq = defaultdict(int)
    word2freq = defaultdict(int)
    if lang in RELEVANT_LANGS:
        uniq_urls = set()
    with open(lang2path[lang]) as f:
        nb_sents = 0
        for line in f:
            nb_sents += 1
            text, url = extract_text_and_url(line, lang)
            if url:
                uniq_urls.add(url)
            # Whitespace-tokenize text
            for word in text.split(" "):
                word2freq[word] += 1
                for char in word:
                    char2freq[char] += 1
    print("Nb sents: {}".format(nb_sents))
    if lang in RELEVANT_LANGS:
        spu_ratio = nb_sents/len(uniq_urls)
        spu_ratios.append(spu_ratio)
        print("Nb unique URLs: {}".format(len(uniq_urls)))
        print("Sents/URL ratio: {:f}".format(spu_ratio))

    nb_tokens = sum(word2freq.values())
    vocab_size = len(word2freq)
    alphabet_size = len(char2freq)
    alphabet_sizes.append(alphabet_size)
    vocab_sizes.append(vocab_size)
    print("Nb tokens: {}".format(nb_tokens))
    print("Vocab size: {}".format(vocab_size))
    print("Alphabet size: {}".format(alphabet_size))
    ## Print all characters in reverse order of frequency
    #for (c,f) in sorted(char2freq.items(), key=lambda x:x[1], reverse=True):
    #    print("  {} ({})".format(c, f))


# Print some summary stats
for (statname, vals) in [("Vocab size", vocab_sizes),
                         ("Alphabet size", alphabet_sizes),
                         ("Sents/URL ratio", spu_ratios)]:
    print("\n----- %s -----" % statname)
    print("Min: {}".format(min(vals)))
    print("Max: {}".format(max(vals)))
    print("Mean: {}".format(sum(vals)/len(vals)))
    print("Median: {}".format(stats.median(vals)))
