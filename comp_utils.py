""" Utilities related to the VarDial 2020 Evaluation campaign. """

import os
from collections import OrderedDict

# Relevant langs for ULI task
RELEVANT_LANGS = set("fit fkv izh kca koi kpv krl liv lud mdf mhr mns mrj myv nio olo sjd sjk sju sma sme smj smn sms udm vep vot vro yrk".split(" "))

# Path of directory containing ULI training data
DIR_ULI_TRAIN = "train_data/ULI2020_training"

def map_ULI_langs_to_paths():
    """ For ULI task, map languages to path of file containing the corresponding training data. 

    Args:

    Returns: OrderedDict

    """
    lang2path = OrderedDict()
    for filename in os.listdir(DIR_ULI_TRAIN):
        # 3 first chars are the language code
        lang = filename[:3]
        lang2path[lang] = os.path.join(DIR_ULI_TRAIN, filename)
    return lang2path

