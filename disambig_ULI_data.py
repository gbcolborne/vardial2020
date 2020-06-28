"""Disambiguate ULI training data."""

import os, argparse, subprocess
from comp_utils import map_ULI_langs_to_paths, stream_sents, data_to_string, string_to_data, RELEVANT_LANGS, IRRELEVANT_URALIC_LANGS
from itertools import combinations

def disambig(dir_in, dir_out, max_length=None):
    # Check args
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    lang2path = map_ULI_langs_to_paths(dir_in)
    lang2filename = {x:os.path.split(y)[-1] for (x,y) in lang2path.items()}
    langs = sorted(lang2path.keys())
    
    # Write labeled data. Store class frequencies
    path_tmp = os.path.join(dir_out, "data.labeled.tmp")
    path_lang_fd = os.path.join(dir_out, "data.lang_fd.tsv")
    lang_fd = None
    if not os.path.exists(path_tmp):
        f = open(path_tmp, 'w')
        line_count = 0
        lang_fd = {}
        for i, lang in enumerate(langs):
            print("{}/{}. {}".format(i+1, len(langs), lang))
            
            # Apply length cutoff and deduplicate
            uniq_sents = set()
            data = []
            for (text, text_id, url) in stream_sents(lang, dir_in):
                if max_length is not None:
                    text = text[:max_length]
                if text not in uniq_sents:
                    uniq_sents.add(text)
                    data.append((text,text_id,url))
            for (text, text_id, url) in data: 
                line = data_to_string(text, lang, "custom", url=url, text_id=text_id, label=lang)
                f.write(line)
                line_count += 1
            lang_fd[lang] = len(data)
        f.close()
        with open(path_lang_fd, 'w') as f:
            for (lang, freq) in lang_fd.items():
                f.write("%s\t%d\n" % (lang, freq))

    # Sort labeled dataset in alphabetical order of texts
    path_sorted = os.path.join(dir_out, "data.sorted.tmp")
    if not os.path.exists(path_sorted):
        cmd = ["sort", path_tmp]
        print("\nSorting %d texts... " % line_count)
        with open(path_sorted, 'w') as outfile:
            subprocess.run(cmd, stdout=outfile)
        print("Done.")

    # Check if we skipped labeling and sorting
    if lang_fd is None:
        lang_fd = {}
        line_count = 0
        with open(path_lang_fd) as f:
            for line in f:
                elems = line.strip().split("\t")
                lang = elems[0]
                freq = int(elems[1])
                lang_fd[lang] = freq
                line_count += freq
                
    # Read in sorted dataset, look for duplicate texts, write disambiguated dataset
    lang2outfile = {lang:open(os.path.join(dir_out, lang2filename[lang]), 'w') for lang in langs}
    prev_text = None
    prev_info = []
    lines_processed = 0
    confusion = {}
    print("\nDisambiguating... ")    
    with open(path_sorted) as f_in:
        for i, line in enumerate(f_in):
            if not len(line.strip()):
                continue
            (text, text_id, url, lang) = string_to_data(line, "custom", lang=None)
            if text == prev_text:                
                prev_info.append((lang, text_id, url))
            else:
                if prev_text is not None:
                    # Disambiguate previous text and write to output file for the language we picked
                    ix = None
                    min_lang_freq = 1e10
                    for j, (x, y, z) in enumerate(prev_info):
                        freq = lang_fd[x]
                        if freq < min_lang_freq:
                            min_lang_freq = freq
                            ix = j
                    (slang, stext_id, surl) = prev_info[ix]
                    output = data_to_string(text, slang, "source", url=surl, text_id=stext_id, label=None)
                    lang2outfile[slang].write(output)
                    # Store confusion counts
                    for (x,y) in combinations([x for (x,y,z) in prev_info], 2):
                        if (x,y) not in confusion:
                            confusion[(x,y)] = 0
                        confusion[(x,y)] += 1
                prev_text = text
                prev_info = [(lang, text_id, url)]
            if (i+1) % 1000000 == 0:
                pct = 100 * (i+1) / line_count
                print("# texts processed: %d/%d (%.1f%%)" % (i+1, line_count, pct))
    print("# texts processed: %d/%d" % (line_count, line_count))
    for (lang, outfile) in lang2outfile.items():
        outfile.close()

    # Clean up.
    for path in [path_tmp, path_sorted, path_lang_fd]:
        subprocess.run(["rm", path])

    # Print some stats on pairwise confusion
    print("\n\nConfusion frequencies:")
    if not len(confusion):
        print("(none)")
    for ((lang1,lang2),freq) in sorted(confusion.items(), key=lambda x:x[1], reverse=True):
        msg = "- (%s, %s): %d" % (lang1, lang2, freq)
        extra = []
        for x in [lang1, lang2]:
            if x in RELEVANT_LANGS:
                extra.append("%s is relevant" % x)
            elif x in IRRELEVANT_URALIC_LANGS:
                extra.append("%s is confounding" % x)
        if len(extra):
            msg += " " * 10
            msg += ", ".join(extra)
        print(msg)
    print()
    return
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path of input directory (containing training data)")    
    parser.add_argument("output_dir", help="Path of output directory")
    parser.add_argument("--max_text_length", type=int)
    args = parser.parse_args()
    disambig(args.input_dir, args.output_dir, max_length=args.max_text_length)
    return
    
    
if __name__ == "__main__":
    main()
