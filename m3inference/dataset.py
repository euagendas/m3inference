#!/usr/bin/env python3
# @Zijian Wang

import unicodedata

from PIL import Image
from torch.utils.data import *
from torchvision import transforms

from .utils import *

logger = logging.getLogger(__name__)


class M3InferenceDataset(Dataset):

    def __init__(self, data: list, use_img=True):

        self.tensor_trans = transforms.ToTensor()
        self.use_img = use_img
        self.data = []
        for entry in data:
            entry = DotDict(entry)
            if use_img:
                self.data.append([entry.id, entry.lang, normalize_space(str(entry.name)),
                                  normalize_space(str(entry.screen_name)),
                                  normalize_url(normalize_space(str(entry.description))), entry.img_path])
            else:
                self.data.append([entry.id, entry.lang, normalize_space(str(entry.name)),
                                  normalize_space(str(entry.screen_name)),
                                  normalize_url(normalize_space(str(entry.description)))])

        logger.info(f'{len(self.data)} data entries loaded.')

    def __getitem__(self, idx):
        data = self.data[idx]
        return self._preprocess_data(data)

    def _preprocess_data(self, data):
        if self.use_img:
            _id, lang, username, screenname, des, img_path = data
            # image
            fig = self._image_loader(img_path)
        else:
            _id, lang, username, screenname, des = data

        # text
        lang_tensor = LANGS[lang]

        username_tensor = [0] * USERNAME_LEN
        if username.strip(" ") == "":
            username_tensor[0] = EMB["<empty>"]
            username_len = 1
        else:
            if len(username) > USERNAME_LEN:
                username = username[:USERNAME_LEN]
            username_len = len(username)
            username_tensor[:username_len] = [EMB.get(i, len(EMB) + 1) for i in username]

        screenname_tensor = [0] * SCREENNAME_LEN
        if screenname.strip(" ") == "":
            screenname_tensor[0] = 32
            screenname_len = 1
        else:
            if len(screenname) > SCREENNAME_LEN:
                screenname = screenname[:SCREENNAME_LEN]
            screenname_len = len(screenname)
            screenname_tensor[:screenname_len] = [ord(i) for i in screenname]

        des_tensor = [0] * DES_LEN
        if des.strip(" ") == "":
            des_tensor[0] = EMB["<empty>"]
            des_len = 1
        else:
            if len(des) > DES_LEN:
                des = des[:DES_LEN]
            des_len = len(des)
            des_tensor[:des_len] = [EMB.get(i, EMB[unicodedata.category(i)]) for i in des]

        if self.use_img:
            return lang_tensor, torch.LongTensor(username_tensor), username_len, torch.LongTensor(
                screenname_tensor), screenname_len, torch.LongTensor(des_tensor), des_len, fig
        else:
            return lang_tensor, torch.LongTensor(username_tensor), username_len, torch.LongTensor(
                screenname_tensor), screenname_len, torch.LongTensor(des_tensor), des_len

    def __len__(self):
        return len(self.data)

    def _image_loader(self, image_name):
        image = Image.open(image_name)
        return self.tensor_trans(image)


if __name__ == "__main__":
    # full
    data = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data.json")))
    dataloader = DataLoader(M3InferenceDataset(data), batch_size=2)
    for i in dataloader:
        break

    # text
    data = json.load(open(os.path.join(os.path.dirname(__file__), "..", "text_data.json")))
    dataloader = DataLoader(M3InferenceDataset(data, use_img=False), batch_size=2)
    for i in dataloader:
        break
