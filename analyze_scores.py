""" Analyze influence of hparams on scores. """

import os, argparse, glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dir_models")
parser.add_argument("track", type=int, choices=[1,2,3])
args = parser.parse_args()

USE_BEST_SCORE = False # If True, a model's score is its best score
                       # across epochs. Otherwise, we use the score at
                       # the end of training.

score_name = "DevF1Track" + str(args.track)
model_dirs = os.listdir(args.dir_models)
all_settings = []
all_scores = []
for d in model_dirs:
    # Parse directory name, which encodes hparam settings
    settings = {}
    parts = d.split("_")
    for part in parts:
        subparts = part.split("=")
        if len(subparts) == 2:
            hname, hval = subparts
            settings[hname] = hval
    # Get training log containing scores
    pattern = os.path.join(args.dir_models, d) + "/2020*"
    path_log = glob.glob(pattern)[0]
    with open(path_log) as f:
        # Read header and find column index of the score we are interested in
        header = f.readline().strip()
        col_names = header.split("\t")
        score_col_ix = col_names.index(score_name)
        scores = []
        for line in f:
            cols = line.strip().split("\t")
            scores.append(float(cols[score_col_ix]))
    if USE_BEST_SCORE:
        score = max(scores)
    else:
        score = scores[-1]
    all_settings.append(settings)
    all_scores.append(score)


# Analyze scores wrt hparam settings
hparam_names = list(all_settings[0].keys())
for hname in hparam_names:
    hval_to_scores = {}
    for i in range(len(all_settings)):
        hval = all_settings[i][hname]
        if hval not in hval_to_scores:
            hval_to_scores[hval] = []
        hval_to_scores[hval].append(all_scores[i])
    print("\nHyperparameter: %s" % hname)
    for val in sorted(hval_to_scores.keys()):
        print("- %s" % val)
        print("  - mean score: %f" % np.mean(hval_to_scores[val])) 
        print("  - max score: %f" % max(hval_to_scores[val])) 
