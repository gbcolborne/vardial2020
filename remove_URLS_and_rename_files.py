import os, argparse
from io import open
from comp_utils import ALL_LANGS, get_path_for_lang, string_to_data

parser = argparse.ArgumentParser()
parser.add_argument("dir_input", help="path of directory containing the source package of training data")
parser.add_argument("dir_output")
args = parser.parse_args()

if os.path.exists(args.dir_output) and os.path.isdir(args.dir_output) and len(os.listdir(args.dir_output)) > 0:
    raise RuntimeError("There is already stuff in %s" % (args.dir_output))
if not os.path.exists(args.dir_output):
    os.makedirs(args.dir_output)

for lang in ALL_LANGS:
    inpath = get_path_for_lang(lang, args.dir_input)
    outpath = os.path.join(args.dir_output, ("%s.train" % lang))
    with open(inpath, 'r', encoding="utf-8") as infile, open(outpath, 'w', encoding="utf-8") as outfile:    
        for line in infile:
            data = string_to_data(line, "source", lang=lang)
            text, text_id, url, lang = data
            outfile.write(text + "\n")

