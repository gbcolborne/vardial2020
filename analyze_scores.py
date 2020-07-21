""" Analyze influence of hparams on scores. """

import os, argparse, glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dir_models")
parser.add_argument("track", type=int, choices=[1,2,3])
args = parser.parse_args()

score_name = "DevF1Track" + str(args.track)
model_dirs = os.listdir(args.dir_models)
all_settings = []
all_max_scores = []
all_last_scores = []
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
    max_score = max(scores)
    last_score = scores[-1]
    all_settings.append(settings)
    all_max_scores.append(max_score)
    all_last_scores.append(last_score)

# Analyze scores wrt hparam settings
hparam_names = list(all_settings[0].keys())
for hname in hparam_names:
    hval_to_max_scores = {}
    hval_to_last_scores = {}
    for i in range(len(all_settings)):
        hval = all_settings[i][hname]
        if hval not in hval_to_max_scores:
            hval_to_max_scores[hval] = []
            hval_to_last_scores[hval] = []        
        hval_to_max_scores[hval].append(all_max_scores[i])
        hval_to_last_scores[hval].append(all_last_scores[i])
    print("\nHyperparameter: %s" % hname)
    for val in sorted(hval_to_max_scores.keys()):
        print("- %s" % val)
        print("  - mean best score: %f" % np.mean(hval_to_max_scores[val])) 
        print("  - max best score: %f" % max(hval_to_max_scores[val])) 
        print("  - mean last score: %f" % np.mean(hval_to_last_scores[val])) 
        print("  - max last score: %f" % max(hval_to_last_scores[val])) 
