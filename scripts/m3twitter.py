#!/usr/bin/env python3
# @Scott Hale

from m3inference import M3Twitter
import pprint
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Retreive profile information for a Twitter screen_name or numeric user id and run m3 inference. You must supply exactly ONE of --id or --screen-name')
    parser.add_argument('--id', help='The numeric id of a Twitter user')
    parser.add_argument('--screen-name',
                        help='The screen_name of a Twitter user (i.e., everything following the @, but do not include @ itself)')
    # parser.add_argument('--skip-cache', type=bool, nargs='?',const=True, default=False,help='By default all requests are cached to the local filesystem and not refetched. Include this flag to disable/overwrite any results already in the cache.')
    parser.add_argument('--skip-cache', dest='skip_cache', action='store_true',
                        help='By default all requests are cached to the local filesystem and not refetched. Include this flag to disable/overwrite any results already in the cache.')
    parser.set_defaults(skip_cache=False)
    args = parser.parse_args()
    if (args.id == None) == (args.screen_name == None):
        # User specified both id and screen-name or neither one.
        sys.stderr.write("Exactly ONE of id or screen-name is required\n")
        parser.print_help(sys.stderr)
        quit(1)

    m3Twitter = M3Twitter()
    if args.id != None:
        pprint.pprint(m3Twitter.infer_id(args.id, skip_cache=args.skip_cache))
    else:
        pprint.pprint(m3Twitter.infer_screen_name(args.screen_name, skip_cache=args.skip_cache))
