"""Disambiguate ULI training data."""

import os, argparse, subprocess
from comp_utils import map_ULI_langs_to_paths, stream_sents, data_to_string, string_to_data, RELEVANT_LANGS, IRRELEVANT_URALIC_LANGS
from itertools import combinations

def disambig(dir_out, max_length=None):
    # Check args
    if os.path.exists(dir_out):
        assert os.path.isdir(dir_out) and len(os.listdir(dir_out)) == 0
    else:
        os.makedirs(dir_out)

    lang2path = map_ULI_langs_to_paths()
    lang2filename = {x:os.path.split(y)[-1] for (x,y) in lang2path.items()}
    langs = sorted(lang2path.keys())
    
    # Write labeled data. Store class frequencies
    path_tmp = os.path.join(dir_out, "data.labeled.tmp")
    f = open(path_tmp, 'w')
    line_count = 0
    lang_fd = {}
    for i, lang in enumerate(langs):
        print("{}/{}. {}".format(i+1, len(langs), lang))

        # Apply length cutoff and deduplicate
        uniq_sents = set()
        data = []
        for (text, text_id, url) in stream_sents(lang):
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

    # Sort labeled dataset in alphabetical order of texts
    path_sorted = os.path.join(dir_out, "data.sorted.tmp")
    cmd = ["shuf", path_tmp]
    print("\nSorting %d texts... " % line_count)
    with open(path_sorted, 'w') as outfile:
        subprocess.run(cmd, stdout=outfile)
    print("Done.")
    cmd = ["rm", path_tmp]
    subprocess.run(cmd)

    # Read in sorted dataset, look for duplicate texts, write disambiguated dataset
    lang2outfile = {lang:open(os.path.join(dir_out, lang2filename[lang]), 'w') for lang in langs}
    prev_text = None
    prev_labels = []
    lines_processed = 0
    confusion = {}
    print("\nDisambiguating... ")    
    with open(path_sorted) as f_in:
        for i, line in enumerate(f_in):
            (text, text_id, url, lang) = string_to_data(line, "custom", lang=None)
            
            if text == prev_text:
                prev_labels.append(lang)
            else:
                if prev_text is not None:
                    # Disambiguate previous text and write to output file for the language we picked
                    sampled_lang = sorted(prev_labels, key=lang_fd.get, reverse=False)[0]
                    output = data_to_string(text, sampled_lang, "source", url=url, text_id=text_id, label=None)
                    lang2outfile[lang].write(output)
                    # Store confusion counts
                    for (x,y) in combinations(prev_labels, 2):
                        if (x,y) not in combinations:
                            combinations[(x,y)] = 0
                        confusion[(x,y)] += 1
                prev_text = text
                prev_labels = [lang]
            if (i+1) % 1000000 == 0:
                pct = 100 * (i+1) / line_count
                print("# texts processed: %d/%d (%.1f%%)" % (i+1, line_count, pct))
    print("# texts processed: %d/%d" % (line_count, line_count))
    for (lang, outfile) in lang2outfile.items():
        outfile.close()
    cmd = ["rm", path_sorted]
    subprocess.run(cmd)

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
    parser.add_argument("output_dir", help="Path of output directory")
    parser.add_argument("--max_text_length", type=int)
    args = parser.parse_args()
    disambig(args.output_dir, max_length=args.max_text_length)
    return
    
    
if __name__ == "__main__":
    main()
