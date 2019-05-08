#!/usr/bin/env python3
# @Zijian Wang

import os
import pickle

# model parameter
BATCH_SIZE = 128
EMBEDDING_OUTPUT_SIZE = 128
EMBEDDING_INPUT_SIZE = 3035
EMBEDDING_OUTPUT_SIZE_ASCII = 16
EMBEDDING_INPUT_SIZE_ASCII = 128
LSTM_LAYER = 2
LSTM_LAYER_DES = 2
LSTM_HIDDEN_SIZE = 256
LSTM_OUTPUT_SIZE = 128
LINEAR_OUTPUT_SIZE = 2
VISION_OUTPUT_SIZE = 2048
USERNAME_LEN = 30
SCREENNAME_LEN = 16
DES_LEN = 200

# model dump parameter
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'full_model': ['https://nlp.stanford.edu/~zijwang/m3inference/full_model.mdl',
                   'https://blablablab.si.umich.edu/projects/m3/models/full_model.mdl'],
    'text_model': ['https://nlp.stanford.edu/~zijwang/m3inference/text_model.mdl',
                   'https://blablablab.si.umich.edu/projects/m3/models/text_model.mdl']
}

PRETRAINED_MODEL_MD5_MAP = {
    'full_model': '7dd11b9d89d7fd209e3baa0058baa4a1',
    'text_model': 'c9a9fbd953b3ad5d84e792c3c50392ad'
}

# unicode parameter
UNICODE_CATS = 'Cc,Zs,Po,Sc,Ps,Pe,Sm,Pd,Nd,Lu,Sk,Pc,Ll,So,Lo,Pi,Cf,No,Pf,Lt,Lm,Mn,Cn,Me,Mc,Nl,Zl,Zp,Cs,Co'.split(",")

# language parameter
LANGS = ['en', 'cs', 'fr', 'nl', 'ar', 'ro', 'bs', 'da', 'it', 'pt', 'no', 'es', 'hr', 'tr', 'de', 'fi', 'el', 'he',
         'ru', 'bg', 'hu', 'sk', 'et', 'pl', 'lv', 'sl', 'lt', 'ga', 'eu', 'mt', 'cy', 'rm', 'is', 'un']
LANGS = {k: v for v, k in enumerate(LANGS)}
UNKNOWN_LANG = 'un'

# embedding parameter
EMBEDDING_INPUT_SIZE_LANGS = len(LANGS) + 1
EMBEDDING_OUTPUT_SIZE_LANGS = 8
EMB = pickle.load(open(os.path.join(os.path.dirname(__file__), "data", "emb.pkl"), "rb"))

PRED_CATS = {
    'gender': ['male', 'female'],
    'age': ['<=18', '19-29', '30-39', '>=40'],
    'org': ['non-org', 'is-org']
}

# Default profile image used on Twitter when no image has been specificed
TW_DEFAULT_PROFILE_IMG = os.path.join(os.path.dirname(__file__), "data", "tw_default_profile.png")
