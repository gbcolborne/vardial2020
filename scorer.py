import argparse
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import multilabel_confusion_matrix as mcm
from comp_utils import map_ULI_langs_to_paths, RELEVANT_LANGS

def load_labels(path, lang2id):
    with open(path) as f:
        return [lang2id[x.strip()] for x in f.readlines()]

def print_title_with_border(title):
    title = "--- %s ---" % title
    line = "-" * len(title)
    print("\n%s\n%s\n%s\n" % (line, title, line))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_pred", help="Path of predicted labels (one per line)")
    parser.add_argument("path_gold", help="Path of gold labels (one per line)")
    parser.add_argument("track", type=int, choices=[1,2,3], 
                        help=("Eval track (1 is 'relevant langs as equals', "
                              "2 is 'relevant sents as equals', "
                              "and 3 in 'all 178 langs as equals')"))
    args = parser.parse_args()

    # Load languages
    all_langs = sorted(map_ULI_langs_to_paths().keys())
    rel_langs = set(RELEVANT_LANGS)
    
    # Map all languages to an integer class ID
    lang2id = {x:i for i,x in enumerate(all_langs)}
    rel_lang_ids = set(lang2id[x] for x in rel_langs)
    
    # Load predicted and gold labels (converted to integer class IDs)
    pred = load_labels(args.path_pred, lang2id)
    gold = load_labels(args.path_gold, lang2id)
    
    # For tracks 1 and 2, we only consider sentences for which either
    # the predicted label or the gold label is a relevant Uralic
    # language
    if args.track in [1,2]:
        p_tmp = []
        g_tmp = []
        for (p, g) in zip(pred, gold):
            if p in rel_lang_ids or g in rel_lang_ids:
                p_tmp.append(p)
                g_tmp.append(g)
        pred = p_tmp[:]
        gold = g_tmp[:]


    print("\nPred:")
    print(pred)
    print("Gold:")
    print(gold)
    print("\n\n")
    if args.track == 2:
        # For track 2, the score is the micro-averaged f1-score
        # over sentences.
        p,r,f,_ = prfs(gold, pred, beta=1.0, labels=None, average="micro", zero_division=0)
        title = "Results (track 2)"
        print_title_with_border(title)
        print("Precision: {:.4f}".format(p))
        print("Recall: {:.4f}".format(r))
        print("F1-score: {:.4f}\n".format(f))
    elif args.track in [1,3]:
        # For tracks 1 and 3, the score is the average (macro)
        # f1-score over a set of languages. For track 1, these are the
        # 29 relevant Uralic languages. For track 3, these are all 178
        # languages.
        if args.track == 1:
            labels = sorted(rel_lang_ids)
        elif args.track == 3:
            labels = sorted(land2id.keys())
        conf = mcm(gold, pred, sample_weight=None, labels=labels, samplewise=False)
        f1scores = []
        for i in range(len(labels)):
            tn = conf[i,0,0]
            tp = conf[i,1,1]                        
            fp = conf[i,0,1]
            fn = conf[i,1,0]
            nb_pred = tp + fp
            nb_gold = tp + fn
            precision = tp / nb_pred
            recall = tp / nb_gold
            f1score = 2 * precision * recall / (precision + recall)
            f1scores.append(f1score)
            print("\nLabel: %s" % all_langs[labels[i]])
            print("  tn: %d" % tn)
            print("  tp: %d" % tp)
            print("  fp: %d" % fp)
            print("  fn: %d" % fn)
            print("  nb_pred: %d" % nb_pred)
            print("  nb_gold: %d" % nb_gold)
            print("  precision: %f" % precision)
            print("  recall: %f" % recall)
            print("  f1score: %f" % f1score)
        macro_f1score = sum(f1scores) / len(f1scores)
        title = "Results (track %d)" % args.track
        print_title_with_border(title)
        print("Average (macro) F1-score: {:.4f}".format(macro_f1score))
            

        
if __name__ == "__main__":
    main()
