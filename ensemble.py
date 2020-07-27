import os, argparse, random
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("dir_pred_files", 
                    type=str, 
                    help="Path of directory containing multiple prediction files.")
parser.add_argument("path_best_guess", 
                    type=str, 
                    help="Path of prediction file considered the most likely to be right. Used to break ties.")
parser.add_argument("path_output", 
                    type=str, 
                    help="Path of file in which ensemble predictions will be written.")
args = parser.parse_args()

# Check args
paths = [os.path.join(args.dir_pred_files, fn) for fn in os.listdir(args.dir_pred_files)]
best_guess_ix = paths.index(args.path_best_guess)
assert best_guess_ix >= 0
assert not os.path.exists(args.path_output)

# Load predictions
pred = []
for path in paths:
    with open(path) as f:
        pred.append([line.strip() for line in f])

# Analyze variance
hist = {}
for i in range(len(pred[0])):
    preds_i = []
    for j in range(len(pred)):
        preds_i.append(pred[j][i])
    nb_uniq_preds = len(set(preds_i))
    if nb_uniq_preds not in hist:
        hist[nb_uniq_preds] = 0
    hist[nb_uniq_preds] += 1
print("Histogram: # different predictions per example:")
n = sum(hist.values())
for k in sorted(hist.keys()):
    pct = 100*hist[k]/n
    print("- {}: {} ({:.1f}%)".format(k,hist[k],pct))

# Seed RNG (for breaking ties)
random.seed(91500)

# Take plurality vote
nb_ties = 0
nb_random_choices = 0
ensemble_pred = []
for i in range(len(pred[0])):
    # Get predictions for this example
    preds_i = []
    for j in range(len(pred)):
        preds_i.append(pred[j][i])
    # Count votes
    pred_fd = Counter(preds_i)
    max_freq = max(pred_fd.values())
    best_preds = [p for p,f in pred_fd.items() if f==max_freq]
    if len(best_preds) > 1:
        nb_ties += 1
        # Break tie
        if preds_i[best_guess_ix] in best_preds:
            ensemble_pred.append(preds_i[best_guess_ix])
        else:
            ensemble_pred.append(random.choice(best_preds))
            nb_random_choices += 1
    elif len(best_preds) == 1:
        ensemble_pred.append(best_preds[0])
    else:
        raise RuntimeError("count_max_freq should be strictly positive...")
pct_ties = 100 * nb_ties / sum(hist.values())
print("# ties: {} ({:.1f}%)".format(nb_ties, pct_ties))
pct_random_choices = 100 * nb_random_choices / nb_ties
print("# ties broken randomly (i.e. where 'best guess' was not among the most frequent predictions): {} ({:.1f}%)".format(nb_random_choices, pct_random_choices))

# Write predictions
with open(args.path_output, 'w') as f:
    f.write("\n".join(ensemble_pred))
