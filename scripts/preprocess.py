#!/usr/bin/env python3
# @Zijian Wang

import argparse
import logging
from m3inference.preprocess import resize_imgs, update_json


logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default=None, required=True,
                        help='The source dir that contains images that need to be resized')
    parser.add_argument('--output_dir', type=str, default=None, required=True,
                        help='The output dir of resized images')

    parser.add_argument('--jsonl_path', type=str, default=None, required=False,
                        help='(Optional) The path to the jsonl file (each line is a json object) to by updated (i.e. the file pointing to the un-resized photos.')

    parser.add_argument('--jsonl_outpath', type=str, default=None, required=False,
                        help='(Optional) The path to write the updated jsonl file (must be used with `--json_filepath')

    parser.add_argument('--force', action='store_true', required=False,
                        help='(Optional) Force resieing every image, even if it exists in the output_dir.')

    parser.add_argument('--verbose', action='store_true', required=False, help='(Optional) Verbose mode')

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    resize_imgs(args.source_dir, args.output_dir, force=args.force)
    if args.jsonl_path:
        assert args.jsonl_outpath is not None
        update_json(args.jsonl_path, args.jsonl_outpath, args.source_dir, args.output_dir)
