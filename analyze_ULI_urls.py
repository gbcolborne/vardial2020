""" Analyze source URLs of the relevant language sentences. """

from collections import Counter
from url_utils import get_netloc, get_toplevel_domains
from comp_utils import RELEVANT_LANGS, map_ULI_langs_to_paths

lang2path = map_ULI_langs_to_paths()
lang2urls = {}
for lang in RELEVANT_LANGS:
    print(lang)
    lang2urls[lang] = set()
    with open(lang2path[lang]) as f:
        for line in f:
            # Last space separated token is the source URL
            cut = line.rstrip().rfind(" ")
            text = line[:cut]
            url = line[cut+1:]
            assert url.startswith("http")
            lang2urls[lang].add(url)

# Map URLs to langs. Do same for top-level domains (generic and country code). 
url2langs = {}
netloc2langs = {}
domain2langs = {}
suffix2langs = {}
for lang, urls in lang2urls.items():
    for url in urls:
        netloc = get_netloc(url)
        domain, suffix = get_toplevel_domains(url)
        for key, dct in [(url, url2langs),
                         (netloc, netloc2langs),
                         (domain, domain2langs),
                         (suffix, suffix2langs)]:
            if key not in dct:
                dct[key] = []
            dct[key].append(lang)

# Show some results
for keyname, dct in [("netloc", netloc2langs),
                     ("domain", domain2langs),
                     ("suffix", suffix2langs)]:
    print("\n--- %s ---" % keyname)
    for i, (key, langs) in enumerate(sorted(dct.items(), key=lambda x:len(x[1]), reverse=True)):
        lang_fd = Counter(langs)
        lang_str = ", ".join("%s (%d)" % (l,f) for l,f in sorted(lang_fd.items(),key=lambda x:x[1], reverse=True))
        print(" %d. %s: %s" % (i+1, key, lang_str))
