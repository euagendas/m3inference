#!/usr/bin/env python3
# @Zijian Wang

import json
from collections import *
from os.path import expanduser

import pandas as pd
import torch.nn as nn
from torch.utils.data import *

from .consts import *
from .dataset import M3InferenceDataset
from .full_model import M3InferenceModel
from .text_model import M3InferenceTextModel
from .utils import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class M3Inference:
    '''
    M3 model wrapper
    '''

    def __init__(self, model_dir=expanduser("~/m3/models/"), pretrained=True, use_full_model=True, use_cuda=True,
                 parallel=False, seed=0):
        '''
        :param model_dir: the dir to cache/read cacahed model dump
        :param pretrained: whether to load pretrained weight
        :param use_full_model: whether to use the full m3 model (it is not recommended to set `use_full_model` to False unless you do not have profile images)
        :param use_cuda: whether to run on a GPU (effective only when there is a GPU)
        :param parallel: when to use DataParallel to infer on multiple GPUs (effective only when `use_cuda=True` and there are multiple available GPUs).
        :param seed: set random seed for `random`, `numpy.random`, and `torch`

        '''
        if seed is not None:
            set_seed(seed)
        self.device = torch.device('cpu') if not use_cuda or not torch.cuda.is_available() else torch.device('cuda')
        self.parallel = parallel
        self.use_full_model = use_full_model
        self.model_type = 'full_model' if self.use_full_model else 'text_model'
        self.model_dir = model_dir

        logger.info('Version 1.0.3')
        logger.info(f'Running on {self.device.type}.')

        if not pretrained:
            logger.info(f'No pretrained model will be loaded.')
            return

        if self.use_full_model:
            logger.info(f'Will use full M3 model.')
            self.model = M3InferenceModel(device=self.device)
            self.load_pretrained_model()
        else:
            logger.info(f'Will use text model. Note that as M3 was optimized to work well with both image and text data, \
                                    it is not recommended to use text only model unless you do not have the profile image.')
            self.model = M3InferenceTextModel(device=self.device)
            self.load_pretrained_model()

        if self.device.type == 'cuda' and self.parallel:
            dev_count = torch.cuda.device_count()
            if dev_count > 1:
                logger.info(f"Model to be paralleled on {dev_count} GPUs.")
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def load_pretrained_model(self):
        if not os.path.isdir(self.model_dir):
            logger.info(f'Dir {self.model_dir} does not exist. Creating now.')
            os.makedirs(self.model_dir)
            logger.info(f'Dir {self.model_dir} created.')

        model_path = os.path.join(self.model_dir, f'{self.model_type}.mdl')

        if not os.path.isfile(model_path):
            logger.info(f'Model {self.model_type} does not exist at {model_path}. Try to download it now.')
            if self.model_type in PRETRAINED_MODEL_ARCHIVE_MAP:
                fetch_pretrained_model(self.model_type, model_path)
                self.load_model_weight(model_path)
            else:
                logger.info(f"Model {self.model_type} is not in out pretrained model list. \
                            Consider {list(PRETRAINED_MODEL_ARCHIVE_MAP.keys())}.")
        else:
            logger.info(f'Model {self.model_type} exists at {model_path}.')
            self.load_model_weight(model_path, need_check=True)

    def load_model_weight(self, model_path, need_check=False):
        if need_check:
            check_file_md5(self.model_type, model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f'Loaded pretrained weight at {model_path}')

    def infer(self, data_or_datapath, output_format='json', batch_size=16, num_workers=4):
        """
        Predict attributes
        :param data_or_datapath: a list of jsons or the path to the json file. For each json entry, the following keys are expected: `id`, `name`, `screen_name`, `description`, `lang`, `img_path` (required when using the full model)
        :param output_format: `json` (with `id` as key and predictions as nested values) or `dataframe` (pandas dataframe)
        :param batch_size: batch_size for dataloader
        :param num_workers: number of workers for dataloader
        :return: an object in `output_format` format
        """
        assert output_format in ['json', 'dataframe']
        if isinstance(data_or_datapath, str):

            # jsonl file path, if not working, raise an error
            data = []
            with open(data_or_datapath) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            # json object
            data = data_or_datapath
        # prediction
        dataloader = DataLoader(M3InferenceDataset(data, use_img=self.use_full_model), batch_size,
                                num_workers=num_workers, pin_memory=True)
        y_pred = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting...'):
                batch = [i.to(self.device) for i in batch]
                pred = self.model(batch)
                y_pred.append([_pred.detach().cpu().numpy() for _pred in pred])

        if output_format == 'json':
            return self.format_json_output(data, y_pred)
        else:
            return self.format_dataframe_output(data, y_pred)

    @classmethod
    def format_json_output(cls, data, y_pred):

        # merge batches to reformat the result
        y_pred = [[b[c][i] for c in range(3)] for b in y_pred for i in range(len(b[0]))]

        # construct output json
        pred_joined = OrderedDict()
        for raw_data, pred in zip(data, y_pred):
            _id = raw_data['id']
            if _id in pred_joined:
                logger.warning(f'ID {_id} already exists. Please double-check the input data. Skipping for now...')
                continue
            nested_pred = {}
            for (pred_cat, pred_types), pred_per_cat in zip(PRED_CATS.items(), pred):
                nested_pred[pred_cat] = {k: round(float(v), 4) for k, v in zip(pred_types, pred_per_cat)}
            pred_joined[_id] = nested_pred
        return pred_joined

    @classmethod
    def format_dataframe_output(cls, data, y_pred):
        y_pred = np.vstack([np.hstack(i) for i in y_pred])

        # construct output df
        columns = [f'{pred_cat}_{v}' for pred_cat, values in PRED_CATS.items() for v in values]
        df = pd.DataFrame(y_pred)
        df.columns = columns
        df['id'] = [i['id'] for i in data]
        df = df[['id'] + columns]

        if len(set(df['id'])) != len(df['id']):
            logger.warning('There are duplicated ids in the dataframe. Please double-check the input data.')
        return df
