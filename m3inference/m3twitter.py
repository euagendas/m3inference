#!/usr/bin/env python3
# @Scott Hale

import html
import json
import logging
import os
import re
import urllib.request
from os.path import expanduser

from .consts import UNKNOWN_LANG, TW_DEFAULT_PROFILE_IMG
from .m3inference import M3Inference
from .preprocess import download_resize_img
from .utils import get_lang

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class M3Twitter(M3Inference):
    _RE_IMG = re.compile('<img class="photo" src="https://pbs.twimg.com/profile_images/(.*?)".*?>(.*?)</a>', re.DOTALL)
    _RE_BIO = re.compile('<p class="note">(.*?)</p>', re.DOTALL)
    _TAG_RE = re.compile(r'<[^>]+>')
    # _SCREEN_NAME=re.compile('<span class="nickname">@(.*?)</span>')
    _NAME_SCREEN_NAME = re.compile(r'<title>(.*?) \(@(.*?)\)')

    def __init__(self, cache_dir=expanduser("~/m3/cache"), model_dir=expanduser("~/m3/models/"), pretrained=True,
                 use_full_model=True, use_cuda=True, parallel=False, seed=0):
        super(M3Twitter, self).__init__(model_dir=model_dir, pretrained=pretrained, use_full_model=use_full_model,
                                        use_cuda=use_cuda, parallel=parallel, seed=seed)
        self.cache_dir = cache_dir
        if not os.path.isdir(self.cache_dir):
            logger.info(f'Dir {self.cache_dir} does not exist. Creating now.')
            os.makedirs(self.cache_dir)
            logger.info(f'Dir {self.cache_dir} created.')

    def transform_jsonl(self, input_file, output_file, img_path_key=None, lang_key=None, resize_img=True,
                        keep_full_size_img=False):
        with open(input_file, "r") as fhIn:
            with open(output_file, "w") as fhOut:
                for line in fhIn:
                    m3vals = self.transform_jsonl_object(line, img_path_key=img_path_key, lang_key=lang_key,
                                                         resize_img=resize_img, keep_full_size_img=keep_full_size_img)
                    fhOut.write("{}\n".format(json.dumps(m3vals)))

    def transform_jsonl_object(self, input, img_path_key=None, lang_key=None, resize_img=True,
                               keep_full_size_img=False):
        """
        input is either a Twitter tweet object (https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object)
            or a Twitter user object (https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/user-object)
        """
        if isinstance(input, str):
            input = json.loads(input)

        if "user" in input:
            user = input["user"]
        else:
            user = input

        if img_path_key != None and img_path_key in user:
            img_path = user[img_path_key]
            if resize_img:
                dotpos = img_path.rfind(".")
                img_file_resize = "{}/{}_224x224.{}".format(self.cache_dir, user["id_str"], img_path[dotpos + 1:])
                download_resize_img(img_path, img_file_resize)
            else:
                img_file_resize = img_path
        elif img_path_key != None and img_path_key in input:
            img_path = input[img_path_key]
            if resize_img:
                dotpos = img_path.rfind(".")
                img_file_resize = "{}/{}_224x224.{}".format(self.cache_dir, user["id_str"], img_path[dotpos + 1:])
                download_resize_img(img_path, img_file_resize)
            else:
                img_file_resize = img_path
        elif user["default_profile_image"]:
            # Default profile image
            img_file_resize = TW_DEFAULT_PROFILE_IMG
        else:
            img_path = user["profile_image_url_https"]
            img_path = img_path.replace("_normal", "_400x400")
            dotpos = img_path.rfind(".")
            img_file_full = "{}/{}.{}".format(self.cache_dir, user["id_str"], img_path[dotpos + 1:])
            img_file_resize = "{}/{}_224x224.{}".format(self.cache_dir, user["id_str"], img_path[dotpos + 1:])
            if not os.path.isfile(img_file_resize):
                if keep_full_size_img:
                    download_resize_img(img_path, img_file_resize, img_file_full)
                else:
                    download_resize_img(img_path, img_file_resize)
        bio = user["description"]
        if bio == None:
            bio = ""

        if lang_key != None and lang_key in user:
            lang = user[lang_key]
        elif lang_key != None and lang_key in input:
            lang = input[lang_key]
        elif bio == "":
            lang = UNKNOWN_LANG
        else:
            lang = get_lang(bio)

        output = {
            "description": bio,
            "id": user["id_str"],
            "img_path": img_file_resize,
            "lang": lang,
            "name": user["name"],
            "screen_name": user["screen_name"]
        }
        return output

    def infer_screen_name(self, screen_name, skip_cache=False):
        """
        Collect data for a Twitter screen name from the Twitter website and predict attributes with m3
        :param scren_name: A Twitter screen_name. Do not include the "@"
        :param skip_cache: If output for this screen name already exists in self.cache_dir, the results will be reused (i.e., the function will not contact the Twitter website and will not run m3).
        :return: a dictionary object with two keys. "input" contains the data from the Twitter website. "output" contains the m3 output in the `output_format` format described for m3.
        """
        screen_name = screen_name.lower()
        if screen_name[0] == "@":
            screen_name = screen_name[1:]
        if not skip_cache:
            # If a json file exists, we'll use that. Otherwise go get the data.
            try:
                with open("{}/{}.json".format(self.cache_dir, screen_name), "r") as fh:
                    logger.info("Results from cache for {}.".format(screen_name))
                    return json.load(fh)
            except:
                logger.info("Results not in cache. Fetching data from Twitter for {}.".format(screen_name))
        else:
            logger.info("skip_cache is True. Fetching data from Twitter for {}.".format(screen_name))

        try:
            data = urllib.request.urlopen("https://twitter.com/intent/user?screen_name={}".format(screen_name))
            data = data.read().decode("UTF-8")
        except urllib.error.HTTPError as err:
            logger.warning(
                "skip_cache is TrueError fetching data from Twitter. HTTP error code was {}. 404 usually indicates the screen_name is invalid.".format(
                    err.code))
            raise err

        output = self.process_twitter(data, screen_name=screen_name)
        with open("{}/{}.json".format(self.cache_dir, screen_name), "w") as fh:
            json.dump(output, fh)
        return output

    def infer_id(self, id, skip_cache=False):
        """
        Collect data for a numeric Twitter user id from the Twitter website and predict attributes with m3
        :param id: A Twitter numeric user id
        :param skip_cache: If output for this screen name already exists in self.cache_dir, the results will be reused (i.e., the function will not contact the Twitter website and will not run m3).
        :return: a dictionary object with two keys. "input" contains the data from the Twitter website. "output" contains the m3 output in the `output_format` format described for m3.
        """
        if not skip_cache:
            # If a json file exists, we'll use that. Otherwise go get the data.
            try:
                with open("{}/{}.json".format(self.cache_dir, id), "r") as fh:
                    logger.info("Results from cache for id {}.".format(id))
                    return json.load(fh)
            except:
                logger.info("Results not in cache. Fetching data from Twitter for id {}.".format(id))
        else:
            logger.info("skip_cache is True. Fetching data from Twitter for id {}.".format(id))

        try:
            data = urllib.request.urlopen("https://twitter.com/intent/user?user_id={}".format(id))
            data = data.read().decode("UTF-8")
        except urllib.error.HTTPError as err:
            logger.warning(
                "Error fetching data from Twitter. HTTP error code was {}. 404 usually indicates the id is invalid.".format(
                    err.code))
            raise err

        output = self.process_twitter(data, id=id)
        with open("{}/{}.json".format(self.cache_dir, id), "w") as fh:
            json.dump(output, fh)
        return output

    def process_twitter(self, data, screen_name=None, id=None):
        img = M3Twitter._RE_IMG.findall(data)
        bio = M3Twitter._RE_BIO.findall(data)
        name_screen_name = M3Twitter._NAME_SCREEN_NAME.findall(data)

        if len(name_screen_name) == 0:
            logger.warning("Could not retreive the name or screen_name.")
            name = ""
            if screen_name == None:
                screen_name = ""
        else:
            screen_name_parsed = name_screen_name[0][1].lower()
            name = name_screen_name[0][0]
            if screen_name == None:
                screen_name = screen_name_parsed
            elif screen_name_parsed != screen_name:
                logger.info(
                    "screen_name from Twitter does not match supplied screen_name. Using user-supplied screen_name. Twiter value is {}. User-supplied value is {}".format(
                        screen_name_parsed, screen_name))

        if len(img) == 0:
            logger.warning("Unable to extract image from Twitter. Using default image.")
            img_file_resize = TW_DEFAULT_PROFILE_IMG
        else:
            img = "https://pbs.twimg.com/profile_images/{}".format(img[0][0])
            img = img.replace("_200x200", "_400x400")
            dotpos = img.rfind(".")
            img_file_full = "{}/{}.{}".format(self.cache_dir, screen_name, img[dotpos + 1:])
            img_file_resize = "{}/{}_224x224.{}".format(self.cache_dir, screen_name, img[dotpos + 1:])
            download_resize_img(img, img_file_resize, img_file_full)
        if len(bio) == 0:
            bio = ""
            logger.warning("No bio available from Twitter")
        else:
            bio = html.unescape(M3Twitter._TAG_RE.sub('', bio[0]))

        data = [{
            "description": bio,
            "id": id if id != None else screen_name,
            "img_path": img_file_resize,
            "lang": get_lang(bio),
            "name": name,
            "screen_name": screen_name
        }]

        pred = self.infer(data, batch_size=1, num_workers=1)

        output = {
            "input": data[0],
            "output": pred[id if id != None else screen_name]
        }
        return output
