""" Checks fine-tuning dataset. """

import os, argparse
from io import open


def check_test_set(dir_data, train_langs, part):
    assert part in ["valid", "test"]
    # Store test texts, mapped to their language
    path_test = os.path.join(dir_data, "%s.tsv" % part)
    lang2test = map_labels_to_texts(path_test)

    # One language at a time, compare training texts to test texts
    print("Checking %s for overlap with training data..." % path_test)
    nb_langs_with_errors = 0
    for lang in sorted(train_langs):
        if lang not in lang2test:
            print("  %s: no test texts for this language" % lang)
            continue
        path_train = os.path.join(dir_data, "%s.train" % lang)
        test_texts = set(lang2test[lang])
        nb_errors = 0
        for (text, _) in stream_data(path_train):
            if text in test_texts:
                nb_errors += 1
        if nb_errors > 0:
            print("  %s: %d ERROR%s FOUND" % (lang, nb_errors, "S" if nb_errors > 1 else ""))
            nb_langs_with_errors += 1
        else:
            print("  %s: OK" % lang)
    print("Nb training languages for which we found at least 1 error: %d" % nb_langs_with_errors)
            

def map_labels_to_texts(path):
    label2texts = {}
    for (text, label) in stream_data(path):
        assert label is not None
        if label not in label2texts:
            label2texts[label] = []
        label2texts[label].append(text)
    return label2texts
        
    
def stream_data(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            elems = line.strip().split("\t")
            if len(elems) > 2:
                msg = "Too many columns"
                raise RuntimeError(msg)
            if len(elems) == 0:
                continue
            text = elems[0]
            if len(elems) == 2:
                label = elems[1]
            else:
                label = None
            yield (text, label)

            
def print_stats(values):
    assert type(values) == list
    print("  min: %d" % min(values))
    print("  max: %d" % max(values))
    print("  mean: %.2f" % (sum(values)/len(values)))
    return


def get_text_lengths(path):
    lengths = []
    for (text, label) in stream_data(path):
        lengths.append(len(text))
    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path of directory containing fine-tuning data")
    args = parser.parse_args()

    # Check file names
    filenames = os.listdir(args.path)
    assert "valid.tsv" in filenames
    train_langs = []
    for fn in filenames:
        if fn not in ["valid.tsv", "test.tsv", "vocab.txt"]:
            if fn[3:] != ".train":
                msg = "Unrecognized file name '%s'" % fn
                raise RuntimeError(msg) 
            lang = fn[:3]
            train_langs.append(lang)
    print("Nb training files: %d" % len(train_langs))
    print("Validation set present: %s" % str("valid.tsv" in filenames))
    print("Test set present: %s" % str("test.tsv" in filenames))    
    print("Unlabeled data (unk.train) present: %s" % str("unk" in train_langs))

    # Check length of texts in valid and test sets
    print("Stats on lengths of texts in validation set:")
    print_stats(get_text_lengths(os.path.join(args.path, "valid.tsv")))
    if "test.tsv" in filenames:
        print("Stats on lengths of texts in test set:")        
        print_stats(get_text_lengths(os.path.join(args.path, "test.tsv")))

    # Make sure no validation or test texts are in the training data
    check_test_set(args.path, train_langs, "valid")
    if "test.tsv" in filenames:
        check_test_set(args.path, train_langs, "test")

        
if __name__ == "__main__":
    main()
