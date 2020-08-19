""" Compute vocab of training data """

import argparse, os
from io import open

def line_to_data(line):
    """Takes a line from a dataset (labeled or unlabeled), and returns the
    text and label.

    """
    elems = line.strip().split("\t")
    assert len(elems) in [1,2]
    text = None
    label = None
    if len(elems) == 1:
        text = elems[0]
    if len(elems) == 2:
        text = elems[0]
        label = elems[1]
    return (text, label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_train", help="path of directory containing training files (one text per line, with optional label, separated by tab)")
    parser.add_argument("path_out", help="Path of output file")
    args = parser.parse_args()
    char2freq = {}
    for filename in sorted(os.listdir(args.dir_train)):
        print(filename)
        path = os.path.join(args.dir_train, filename)
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                text, label = line_to_data(line)
                for char in text:
                    if char not in char2freq:
                        char2freq[char] = 0
                    char2freq[char] += 1
                if (i+1) % 100000 == 0:
                    print("Nb lines processed: %d" % (i+1))
    srtd = sorted(char2freq.items(), key=lambda x:x[1], reverse=True)
    with open(args.path_out, 'w') as outfile:
        for char, freq in srtd:
            outfile.write("%s\t%d\n" % (char, freq))
    return
        
if __name__ == "__main__":
    main()
