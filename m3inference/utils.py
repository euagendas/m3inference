#!/usr/bin/env python3
# @Zijian Wang

from random import shuffle

import hashlib
import logging
import numpy as np
import pycld2 as cld2
import random
import re
import requests
import shutil
import tempfile
from torch.nn.utils.rnn import *
from tqdm import tqdm

from .consts import *

logger = logging.getLogger(__name__)


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def pack_wrapper(sents, lengths):
    lengths_sorted, idx_sorted = lengths.sort(descending=True)
    sents_sorted = sents[idx_sorted]
    packed = pack_padded_sequence(sents_sorted, lengths_sorted, batch_first=True)
    return packed, idx_sorted


def unpack_wrapper(sents, idx_unsort):
    h, _ = pad_packed_sequence(sents, batch_first=True)
    h = torch.zeros_like(h).scatter_(0, idx_unsort.unsqueeze(1).unsqueeze(1).expand(-1, h.shape[1], h.shape[2]), h)
    return h


def get_lang(sent):
    lang = cld2.detect(''.join([i for i in sent if i.isprintable()]), bestEffort=True)[2][0][1]
    return UNKNOWN_LANG if lang not in LANGS else lang


def normalize_url(sent):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '\u20CC', sent)


def normalize_space(sent):
    return sent.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def fetch_pretrained_model(model_name, model_path):
    # Edited from https://github.com/huggingface/pytorch-pretrained-BERT/blob/68a889ee43916380f26a3c995e1638af41d75066/pytorch_pretrained_bert/file_utils.py
    # TODO: check whether the license from huggingface works with ours
    assert model_name in PRETRAINED_MODEL_ARCHIVE_MAP
    model_urls = PRETRAINED_MODEL_ARCHIVE_MAP[model_name]
    shuffle(model_urls)
    download_flag = False
    for idx, model_url in enumerate(model_urls):
        try:
            temp_file = tempfile.NamedTemporaryFile()
            logger.info(f'{model_path} not found in cache, downloading from {model_url} to {temp_file.name}')

            req = requests.get(model_url, stream=True)
            content_length = req.headers.get('Content-Length')
            total = int(content_length) if content_length is not None else None
            progress = tqdm(unit="KB", total=round(total / 1024))
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(1)
                    temp_file.write(chunk)
            progress.close()
            temp_file.flush()
            temp_file.seek(0)
            download_flag = True
        except Exception as e:
            logger.warning(f'Download from {idx + 1}/{len(model_urls)} mirror failed with an exception of\n{str(e)}')
            try:
                temp_file.close()
            except Exception as e_file:
                logger.warning(f'temp_file failed with an exception of \n{str(e_file)}')
            continue

        if not download_flag:
            logging.warning(f'Download from all mirrors failed. Please retry.')
            return

        logger.info(f'Model {model_name} was downloaded to a tmp file.')
        logger.info(f'Copying tmp file to {model_path}.')
        with open(model_path, 'wb') as cache_file:
            shutil.copyfileobj(temp_file, cache_file)
        logger.info(f'Copied tmp model file to {model_path}.')
        temp_file.close()

        if download_flag and check_file_md5(model_name, model_path):
            break


def check_file_md5(model_name, model_path):
    assert model_name in PRETRAINED_MODEL_MD5_MAP
    logger.info(f'Checking MD5 for model {model_name} at {model_path}')
    correct_md5 = PRETRAINED_MODEL_MD5_MAP[model_name]
    downloaded_md5 = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
    if correct_md5 == downloaded_md5:
        logger.info('MD5s match.')
        return True
    else:
        logger.error('MD5s mismatch. Consider clean your tmp dir (default: `./m3_tmp`) and retry,'
                     ' or download from the link in our github repo.')
        return False
