""" Utilities related to the VarDial 2020 Evaluation campaign. """

import os, glob
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


def get_path_for_lang(lang):
    """ Infer path of training data for language. """
    pattern = "%s/%s*" % (DIR_ULI_TRAIN, lang)
    paths = glob.glob(pattern)
    path = paths[0]
    return path


def extract_text_and_url(line, lang):
    """Extract text from a line extracted from one of the training files. """
    if lang in RELEVANT_LANGS:
        # Last space separated token is the source URL
        cut = line.rstrip().rfind(" ")
        text = line[:cut]
        url = line[cut+1:]
        assert url.startswith("http")
        return (text, url)
    else:
        line_number, text = line.split("\t")
        text = text.strip()
        # Make sure we can convert the line number to an int
        int(line_number)
        return (text, None)


def stream_sents(lang):
    """Stream sentences (along with their URL) from training data. Skip empty lines."""
    path = get_path_for_lang(lang)
    with open(path) as f:
        for line in f:
            text, url = extract_text_and_url(line, lang)
            if len(text):
                yield (text, url)
                


  

