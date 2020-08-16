import argparse
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.metrics import confusion_matrix
from comp_utils import ALL_LANGS, RELEVANT_LANGS


def load_labels(path):
    """Load labels (i.e. ISO language codes) from file. This file can
    contain a single column of labels, or 2 columns (separated by a tab),
    containing the example and label. Empty lines are skipped.

    """
    with open(path) as f:
        labels = []
        for line in f.readlines():
            elems = line.strip().split("\t")
            label = None
            if len(elems) == 0:
                continue
            if len(elems) == 1:
                label = elems[0]
            if len(elems) == 2:
                label = elems[1]
            assert label is not None
            assert label in ALL_LANGS
            labels.append(label)
        return labels

    
def print_title_with_border(title):
    title = "--- %s ---" % title
    line = "-" * len(title)
    print("\n%s\n%s\n%s\n" % (line, title, line))


def compute_fscores(pred, gold, verbose=False):
    """Compute f-scores for all three tracks.

    Args:
    - pred: list of predicted labels (i.e. ISO language codes)
    - gold: list of gold labels (i.e. ISO language codes)

    Returns: Dict mapping track names to the f-score for that track

    """
    # For tracks 1 and 2, we only consider sentences for which either
    # the predicted label or the gold label is a relevant Uralic
    # language
    pred_filtered = []
    gold_filtered = []
    for (p, g) in zip(pred, gold):
        if p in RELEVANT_LANGS or g in RELEVANT_LANGS:
            pred_filtered.append(p)
            gold_filtered.append(g)

    # For track 1, the score is the average (macro) f1-score
    # over the 29 relevant Uralic languages. 
    fscore1 = compute_macro_fscore(pred_filtered,
                                   gold_filtered,
                                   track=1,
                                   verbose=False)
    if verbose:
        title = "Results (Track 1)"
        print_title_with_border(title)
        print("- Average (macro) F1-score: %.4f" % fscore1)
            
    # For track 2, the score is the micro-averaged f1-score over
    # sentences. Precision is computed wrt to cases where either the
    # predicted label or the gold label is a relevant language. Recall
    # is computed wrt to cases where the gold label is a relevant
    # language.
    nb_correct = sum(1 for i in range(len(gold_filtered)) if pred_filtered[i] == gold_filtered[i])
    nb_relevant_gold = sum(1 for i in range(len(gold_filtered)) if gold_filtered[i] in RELEVANT_LANGS)
    p = nb_correct / len(pred_filtered)
    r = nb_correct / nb_relevant_gold
    if p == 0 and r == 0:
        fscore2 = 0
    else:
        fscore2 = 2 * p * r / (p + r)

    if verbose:
        title = "Results (Track 2)"
        print_title_with_border(title)
        print("- Precision: %.4f" % p)
        print("- Recall: %.4f" % r)
        print("- F1-score: %.4f" % fscore2)

    # For track 3, the score is the average (macro) f1-score
    # over all 178 languages.
    fscore3 = compute_macro_fscore(pred, gold, track=3, verbose=False)
    if verbose:
        title = "Results (Track 3)"
        print_title_with_border(title)
        print("- Average (macro) F1-score: %.4f" % fscore3)
    return {"track1":fscore1, "track2":fscore2, "track3":fscore3}


def compute_macro_fscore(pred, gold, track=1, verbose=False):
    """ Compute macro-averaged f-score for track 1 or 3. 

    Args:
    - pred: list of predicted labels (i.e. ISO language codes)
    - gold: list of gold labels (i.e. ISO language codes)
    - track: 1 or 3
    
    Returns: f-score (Float)

    """
    assert track in [1,3]
    # Get binary confusion matrix for each label
    if track == 1:
        labels = sorted(RELEVANT_LANGS)
    else:
        labels = sorted(ALL_LANGS)
    conf = mcm(gold, pred, sample_weight=None, labels=labels, samplewise=False)
    f1scores = []
    for i in range(len(labels)):
        # Get sufficient statistics from confusion matrix for this label
        tp = conf[i,1,1]                        
        fp = conf[i,0,1]
        fn = conf[i,1,0]
        nb_pred = tp + fp
        nb_gold = tp + fn
        # Compute f1-score for this label
        f1score = None
        if nb_gold == 0:
            if nb_pred == 0:
                # If nb_pred is 0 and nb_gold is 0, then both
                # recall and precision are undefined. In this
                # case, f1-score is 1.
                f1score = 1.0
            else:
                assert nb_pred > 0
                # If nb_pred is strictly positive but nb_gold is
                # 0, then recall is undefined, and precision is
                # 0. In this case, f1-score is 0.
                f1score = 0.0
        else:
            assert nb_gold > 0
            if nb_pred == 0:
                # If nb_pred is 0 but nb_gold is strictly
                # positive, then recall is 0, and precision is
                # undefined. In this case, f-score is 0.
                f1score = 0.0
            else:
                assert nb_pred > 0
                precision = tp / nb_pred
                recall = tp / nb_gold
                if (precision + recall) > 0:
                    f1score = 2 * precision * recall / (precision + recall)
                else:
                    f1score = 0
        f1scores.append(f1score)
        if verbose:
            print("\nStats for label '%s':" % labels[i])
            print("  # gold: %d" % nb_gold)
            print("  # pred: %d" % nb_pred)
            print("  # true pos: %d" % tp)
            print("  F1-score: %.4f" % f1score)
    macro_avg = sum(f1scores) / len(f1scores)
    return macro_avg
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_pred", help="Path of predicted labels, i.e. ISO language codes (one per line)")
    parser.add_argument("path_gold", help="Path of gold labels, i.e. ISO language codes (one per line)")
    parser.add_argument("--show_confusion", help="Show some information on the confusion matrix.")
    args = parser.parse_args()
    pred = load_labels(args.path_pred)
    gold = load_labels(args.path_gold)
    fscore_dict = compute_fscores(pred, gold, verbose=True)
    print("\n\n")

    if args.show_confusion:
        
    
        # Show confusion matrix for relevant languages
        labels = sorted(ALL_LANGS)
        label2id = {x:i for (i,x) in enumerate(labels)}
        conf = confusion_matrix(gold, pred, labels=labels)
        print("Most frequent prediction for the relevant languages:")
        for label in sorted(RELEVANT_LANGS):
            label_id = label2id[label]
            srt = np.argsort(conf[label_id])
            argmax = srt[-1]
            most_confused = labels[argmax]
            if most_confused == label:
                print("- {}: {}".format(label, most_confused))
            else:
                if most_confused in RELEVANT_LANGS:
                    group = "relevant"
                else:
                    group = "irrelevant"
                print("- {}: {} ({})".format(label, most_confused, group))
    
                            
        # Now show confusion matrix for irrelevant languages
        print()
        print("Most frequent prediction for the irrelevant languages:")
        irrelevant_langs = ALL_LANGS.difference(RELEVANT_LANGS)
        for label in sorted(irrelevant_langs):
            label_id = label2id[label]
            srt = np.argsort(conf[label_id])
            argmax = srt[-1]
            most_confused = labels[argmax]
            if most_confused == label:
                print("- {}: {}".format(label, most_confused))
            else:
                if most_confused in RELEVANT_LANGS:
                    group = "relevant"
                else:
                    group = "irrelevant"
                print("- {}: {} ({})".format(label, most_confused, group))

    
if __name__ == "__main__":
    main()
