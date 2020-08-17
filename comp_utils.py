""" Utilities related to the VarDial 2020 Evaluation campaign. """

import os, glob
from collections import OrderedDict

# Irrelevant langs for ULI task (including 3 confounding Uralic laguages)
IRRELEVANT_LANGS = set(['afr','als','amh','ara','asm','azj','bak','bar',
                        'bcl','bel','ben','bos','bpy','bre','bul','cat',
                        'ceb','ces','che','chv','cmn','cos','cym','dan',
                        'deu','diq','div','ekk','ell','eng','epo','eus',
                        'ext','fao','fin','fra','fry','gle','glg','glv',
                        'gom','grn','gsw','guj','hat','heb','hif','hin',
                        'hrv','hsb','hun','ido','ilo','ina','ind','isl',
                        'ita','jav','jpn','kal','kan','kat','kaz','kir',
                        'kor','krc','ksh','lat','lav','lim','lit','lmo',
                        'ltz','lug','lus','mal','mar','min','mkd','mlg',
                        'mlt','mon','mri','mwl','mzn','nds','nep','new',
                        'nld','nno','nob','nso','oci','ori','oss','pam',
                        'pan','pes','pfl','pms','pnb','pol','por','pus',
                        'que','roh','ron','rus','sah','scn','sco','sgs',
                        'sin','slk','slv','sna','som','sot','spa','srd',
                        'srp','sun','swa','swe','tam','tat','tel','tgk',
                        'tgl','tha','tso','tuk','tur','uig','ukr','urd',
                        'uzn','vec','vie','vls','vol','wln','wuu','xho',
                        'xmf','yid','zea','zsm','zul'])

# Relevant langs for ULI task
RELEVANT_LANGS = set(['fit', 'fkv', 'izh', 'kca', 'koi', 'kpv', 'krl',
                      'liv', 'lud', 'mdf', 'mhr', 'mns', 'mrj', 'myv',
                      'nio', 'olo', 'sjd', 'sjk', 'sju', 'sma', 'sme',
                      'smj', 'smn', 'sms', 'udm', 'vep', 'vot', 'vro',
                      'yrk'])

# Irrelevant Uralic languages for ULI task
IRRELEVANT_URALIC_LANGS = set(['ekk', 'fin', 'hun'])

# All langs for ULI task
ALL_LANGS = IRRELEVANT_LANGS.union(RELEVANT_LANGS)

# Formats used for writing/parsing data
DATA_FORMATS = ["source", "custom"]


def data_to_string(text, lang, frmt, url=None, text_id=None, label=None):
    """ Convert data to string.

    Args:
    - text
    - lang
    - frmt: format
    - url
    - text_id
    - label

    """
    assert frmt in DATA_FORMATS
    if lang in RELEVANT_LANGS:
        assert url is not None
    else:
        assert text_id is not None
        assert type(text_id) == int

    # Execute
    if frmt == "source":
        # This is the format used in the source package containing the
        # ULI training data
        if lang in RELEVANT_LANGS:
            return "%s %s\n" % (text, url)
        else:
            return "%d\t%s\n" % (text_id, text)
        
    if frmt == "custom":
        # Custom format for labeled data
        if lang in RELEVANT_LANGS:
            return "%s\t%s\t%s\n" % (text, label, url)
        else:
            return "%s\t%s\t%s\n" % (text, label, text_id)

        
def string_to_data(string, frmt, lang=None):
    """ Parse a line from the dataset.

    Args:
    - string
    - frmt: format
    - lang (required if frmt is `source`)

    Returns: (text, text_id, url, lang)

    """
    assert frmt in DATA_FORMATS
    if frmt == "source":
        # This is the format used in the source package containing the
        # ULI training data
        assert lang is not None
        if lang in RELEVANT_LANGS:
            # Last space separated token is the source URL
            string = string.strip()
            cut = string.rfind(" ")
            text = string[:cut]
            url = string[cut+1:]
            assert url.startswith("http")
            return (text, None, url, lang)
        else:
            text_id, text = string.split("\t")
            text = text.strip()
            text_id = int(text_id)
            return (text, text_id, None, lang)
        
    if frmt == "custom":
        # Custom format for labeled data
        elems = string.strip().split("\t")
        assert len(elems) == 3
        text = elems[0]
        lang = elems[1]
        if lang in RELEVANT_LANGS:
            url = elems[2]
            text_id = None
        else:
            text_id = int(elems[2])
            url = None
        return (text, text_id, url, lang)

    
def map_ULI_langs_to_paths(dir_training_data):
    """ For ULI task, map languages to path of file containing the corresponding training data. 

    Args:
    - dir_training_data

    Returns: OrderedDict

    """
    lang2path = OrderedDict()
    for filename in os.listdir(dir_training_data):
        # 3 first chars are the language code
        lang = filename[:3]
        lang2path[lang] = os.path.join(dir_training_data, filename)
    return lang2path


def get_path_for_lang(lang, dir_training_data):
    """ Infer path of training data for language. """
    pattern = "%s/%s*" % (dir_training_data, lang)
    paths = glob.glob(pattern)
    path = paths[0]
    return path


def extract_text_id_and_url(line, lang):
    """Extract text from a line extracted from one of the training files. """
    text, text_id, url, lang = string_to_data(line, "source", lang=lang)
    return (text, text_id, url)


def stream_sents(lang, dir_training_data, input_format="source"):
    """Stream sentences (along with their URL) from training data. Skip empty lines."""
    assert input_format in ["source", "text-only"]
    path = get_path_for_lang(lang, dir_training_data)
    with open(path) as f:
        for line in f:
            if input_format == "source":
                text, text_id, url = extract_text_id_and_url(line, lang)
            elif input_format == "text-only":
                text = line.strip()
                text_id = None
                url = None
            if len(text):
                yield (text, text_id, url)
                


  

