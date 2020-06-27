""" Utilities related to the VarDial 2020 Evaluation campaign. """

import os, glob
from collections import OrderedDict

# Relevant langs for ULI task
RELEVANT_LANGS = set(['fit', 'fkv', 'izh', 'kca', 'koi', 'kpv', 'krl',
                      'liv', 'lud', 'mdf', 'mhr', 'mns', 'mrj', 'myv',
                      'nio', 'olo', 'sjd', 'sjk', 'sju', 'sma', 'sme',
                      'smj', 'smn', 'sms', 'udm', 'vep', 'vot', 'vro',
                      'yrk'])

# Irrelevant Uralic languages for ULI task
IRRELEVANT_URALIC_LANGS = set(['ekk', 'fin', 'hun'])

# Path of directory containing ULI training data
DIR_ULI_TRAIN = "data/ULI_training_data/ULI2020_training"


def format_example(text, lang, url=None, text_id=None, label=None):
    """ Given a text, return a string corresponding to the format used in the ULI training data package, unless label is provided, in which we use a custom format. """
    # A few checks to make sure we have the data necessary to produce
    # format used in ULI training data package
    if lang in RELEVANT_LANGS:
        assert url is not None
    else:
        assert text_id is not None
        assert type(text_id) == int

    # Execute
    if label is None:
        # Use format of ULI training data package
        if lang in RELEVANT_LANGS:
            return "%s %s\n" % (text, url)
        else:
            return "%d\t%s\n" % (text_id, text)
    else:
        # Use custom format for labeled data
        if lang in RELEVANT_LANGS:
            return "%s\t%s\t%s\n" % (text, label, url)
        else:
            return "%s\t%s\t%s\n" % (text, label, text_id)
        

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


def extract_text_id_and_url(line, lang):
    """Extract text from a line extracted from one of the training files. """
    if lang in RELEVANT_LANGS:
        # Last space separated token is the source URL
        cut = line.rstrip().rfind(" ")
        text = line[:cut]
        url = line[cut+1:]
        assert url.startswith("http")
        return (text, None, url)
    else:
        line_number, text = line.split("\t")
        text = text.strip()
        text_id = int(line_number)
        return (text, text_id, None)


def stream_sents(lang):
    """Stream sentences (along with their URL) from training data. Skip empty lines."""
    path = get_path_for_lang(lang)
    with open(path) as f:
        for line in f:
            text, text_id, url = extract_text_id_and_url(line, lang)
            if len(text):
                yield (text, text_id, url)
                


  

