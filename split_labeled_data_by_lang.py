""" Split labeled data by language. """

import os, argparse
from comp_utils import ALL_LANGS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path of directory containing files named train.tsv and valid.tsv, and optionally test.tsv")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    # Check args
    assert "train.tsv" in os.listdir(args.input_dir) and "valid.tsv" in os.listdir(args.input_dir)
    if os.path.exists(args.output_dir):
        assert os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) == 0
    else:
        os.makedirs(args.output_dir)
    parts = ["train", "valid"]
    if "test.tsv" in os.listdir(args.input_dir):
        parts.append("test")

    # Process data
    for part in parts:
        lang2out = {lang:open(os.path.join(args.output_dir, "%s.%s" % (lang, part)), 'w') for lang in ALL_LANGS}
        fni = os.path.join(args.input_dir, "%s.tsv" % part)
        with open(fni) as fi:
            for line in fi:
                elems = line.strip().split("\t")
                text = elems[0] 
                lang = elems[1] 
                assert lang in ALL_LANGS
                lang2out[lang].write(text + "\n")
        for outfile in lang2out.values():
            outfile.close()
    
if __name__ == "__main__":
    main()
