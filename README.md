# M3-Inference
This is a PyTorch implementation of the M3 (Multimodal, Multilingual, and Multi-attribute) system described in the WebConf (WWW) 2019 paper [Demographic Inference and Representative Population Estimates from Multilingual Social Media Data](https://doi.org/10.1145/3308558.3313684).

## Quick Links

- [About](#about)
- [Install](#install)
- [FAQs](#faqs)
- [Citation](#citation)
- [Contact](#more-questions)
- [License](#license)

## About
M3 is a deep learning system for demographic inference that was trained on massive Twitter dataset. It features three major attributes:

* Multimodal
    - M3 takes both vision and text inputs. Particularly, the input may contain a profile image, a name (e.g., in the form of a natural language first and last name) , a screen name, and a short self-descriptive text, e.g., a Twitter biography.
 
* Multilingual
    - M3 operates in 32 major languages spoken in Europe. They are `['en', 'cs', 'fr', 'nl', 'ar', 'ro', 'bs', 'da', 'it', 'pt', 'no', 'es', 'hr', 'tr', 'de', 'fi', 'el', 'he',
         'ru', 'bg', 'hu', 'sk', 'et', 'pl', 'lv', 'sl', 'lt', 'ga', 'eu', 'mt', 'cy', 'rm', 'is', 'un']` in [ISO 639-1 two-letter codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) (`un` stands for languages that are not in the list). 
    
* Multi-attribute
    - Thanks to multi-task learning, the model can predict three demographic attributes (gender, age, and human-vs-organization status) at the same time.

## Install
### TL;DR
`pip install m3inference`

* If there is an error with the installation of `torch`, you may install it with `conda` (see [here](https://pytorch.org/)). Alternatively, you could create a conda environment - see instructions below.

### Manually Install


#### With pip
You must have `Python>=3.6` and `pip` ready to use. Then you can:
1. Install dependency packages: `pip install -r requirements.txt`
2. Install the package `python setup.py install`

#### As a conda environment
1. Simply run `conda-env create -f env_conda.yml`, you should then have a "m3env" environment available which you can enter with `conda activate m3env`. Run everything else from within there.
2. Install the package `python setup.py install`


### How to use
#### With M3
M3 takes an input of a `jsonl` file that contains `a list of json(dict) objects` (or a python object containing the data itself) and outputs the predictions for the three attributes.

Demo with `test` dir:

1. Clone this package (`git clone https://github.com/zijwang/m3inference.git`) and follow [Manually Install](#manually-install) to install the package.

2. `Preprocess` the image to get them resized to the correct shape. To do this, at the same (root) dir, run
    ```
    python scripts/preprocess.py --source_dir test/pic/ --output_dir test/pic_resized/ --jsonl_path test/data.jsonl --jsonl_outpath test/data_resized.jsonl --verbose
    ```

   You may also run `python scripts/preprocess.py --help` to see detailed usages. Further, see [FAQs](#faqs) for more information on images.

3. In Python, run:

```
from m3inference import M3Inference
import pprint
m3 = M3Inference() # see docstring for details
pred = m3.infer('./test/data_resized.jsonl') # also see docstring for details
pprint.pprint(pred)
```

You should see results like the following:


```
OrderedDict([('720389270335135745',
              {'age': {'19-29': 0.1546,
                       '30-39': 0.114,
                       '<=18': 0.0481,
                       '>=40': 0.6833},
               'gender': {'female': 0.0066, 'male': 0.9934},
               'org': {'is-org': 0.7508, 'non-org': 0.2492}}),
             ('21447363',
              {'age': {'19-29': 0.0157,
                       '30-39': 0.9837,
                       '<=18': 0.0004,
                       '>=40': 0.0002},
               'gender': {'female': 0.9866, 'male': 0.0134},
               'org': {'is-org': 0.0002, 'non-org': 0.9998}}),
    ...
  ...
```


Each entry of the input file (`./test/data.jsonl`) should have the following keys: `id`, `name`, `screen_name`, `description`, `lang`, `img_path`. 
* The first four keys could be extracted directly from the Twitter JSON entry. 
* For `lang`, even if the official Twitter JSON entry contains this field, we recommend to try to use our [cld2](https://github.com/CLD2Owners/cld2) wrapper method (`from m3inference import get_lang`) to get the language from either user's biography/description or the user's tweets. You could also hard-code the language if you know the ground truth from other sources.
* Images should be downloaded from Twitter as 400x400 pixel images and resized to 224x224 pixels using the preprocess code above. 


The output file is a dict in which the `id`s are the keys and the predictions are the nested values. The values represents the probability of that category (`[0, 1]`).


For other model settings (e.g., output format, GPU setting, batch_size, etc.), please use the file `test/data.jsonl` as a sample input file and see the docstrings fo `M3Inference` initialization and infer method for detailed utilization.


#### With M3 Twitter Wrapper

##### Existing JSON Twitter data
If you have a Twitter JSON object representing a user but do *not* have images ready, you can use our `M3Twitter` class to:
* Download and resize the images 
* Add a detected language using CLD2 over the biography text
* Transform the JSON into the input structure required for M3.

```
from m3inference import M3Twitter
import pprint

m3twitter=M3Twitter(cache_dir="twitter_cache") #Change the cache_dir parameter to control where profile images are downloaded
m3twitter.transform_jsonl(input_file="test/twitter_cache/example_tweets.jsonl",output_file="test/twitter_cache/m3_input.jsonl")

pprint.pprint(m3twitter.infer("test/twitter_cache/m3_input.jsonl")) #Same method as M3Inference.infer(...)
```

If you already have images locally, please include the ``image_path_key`` parameter and set it to the key in your JSON object containing the path to the image locally. Similarly, if you have detected languages, you can use the ``lang_key`` parameter. An example is given in `test/test_transform_jsonl.py`

##### Nothing but a screen_name or numeric id
You can also run the Twitter wrapper directly for a Twitter screen_name or numeric id. From the command line, this looks like:


## FAQs
### What if I just have a Twitter screen name or id?

You can use the M3Twitter class to get all the needed profile information (and image) from the Twitter website. Please note this function should only be used for a small number of screen_names or numeric ids. If you have a large list, please use the Twitter API to get the required information (apart from the profile photo, which can be downloaded separately using the `.transform_jsonl(...)` method described above).

```
import pprint
from m3inference import M3Twitter
m3twitter=M3Twitter()
pprint.pprint(m3twitter.infer_id("2631881902"))
```

The `.infer_screen_name(...)`  method does the same for a Twitter screen name. All results are stored/cached in "~/m3/cache/". This directory can be changed in the M3Twitter constructor and you can skip/update the cache for a single request by setting `skip_cache=True` on the `.infer_id(...)` or `.infer_screen_name(...)` method.

You can also run these examples directly from the terminal to try things out:
```
python scripts/m3twitter.py --screen-name=barackobama
```

### How should I get the images?

If you have nothing that a screen name or numeric id, you can use the `M3Twitter.infer_screen_name(...)` or `M3Twitter.infer_id(...)` methods. Please note, however, that these methods directly access the Twitter website, not the API and therefore are suitable only to small lists. With a large list of screen_names/ids, please use the Twitter API to get user information.

Once you have Twitter JSON, you can use the `M3Twitter.transform_jsonl(...)` to download images, run language detection, and transform the data to the M3 input format.

### What if I cannot have image data?
In the package, we do provide the standalone `text-based model`. You could set `use_full_model=False` when initializing `M3Inference` object (i.e., `m3=M3Inference(use_full_model=False)`). You then do not need to provide `img_path` field in the input json file.

*Warning*: as M3 model is optimized with the best performance when both image and text inputs are available. You may experience lower performance when using the `text-based model`. We recommend using image data whenever possible to get the most accurate predictions.



## Citation
Please cite our WWW 2019 paper if you use this package in your project.

```
@inproceedings{wang2019demographic,
  title={Demographic Inference and Representative Population Estimates from Multilingual Social Media Data},
  author={Wang, Zijian and Hale, Scott A. and Adelani, David and Grabowicz, Przemyslaw A. and Hartmann, Timo and Fl{\"o"}ck, Fabian and Jurgens, David},
  booktitle={Proceedings of the 2019 World Wide Web Conference},
  year={2019},
  organization={ACM}
}
```

## More Questions

We use issues on this GitHub for all questions or suggestions.  For specific inqueries, please contact us as `hello@euagendas.org`.  Please note that we are unable to release or provide training data for this model due to existing terms of service.

## License

This source code is licensed under the GNU Affero General Public License, which allows for non-commercial re-use of this software.  For commercial inqueries, please contact us directly. Please see the LICENSE file in the root directory of this source tree for details.
