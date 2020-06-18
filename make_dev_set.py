""" Make dev set """

import argparse
from comp_utils import map_ULI_langs_to_paths

parser = argparse.ArgumentParser()
parser.add_argument("--structure", "-s",
                    choices=["unbalanced", "balanced-by-lang", "balanced-by-group", "double-balanced"],
                    default="unbalanced",
                    help="How to structure the dev set.")
args = parser.parse_args()

# Loop over training data files (one per language)
lang2path = map_ULI_langs_to_paths()
for lang,path in lang2path.items():
    print(lang)
