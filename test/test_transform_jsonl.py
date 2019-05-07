
from m3inference import M3Twitter
import pprint
import json

#./twitter_cache/example_tweets.jsonl is a standard collection of tweets
# in Twitter JSON format. One JSON object per line.

print("---An example tweet---")
with open("./twitter_cache/example_tweets.jsonl","r") as fh:
	tweet=json.loads(fh.readline())
	pprint.pprint(tweet)


print("---Now let's transform this data to the input needed for m3--")
m3twitter=M3Twitter(cache_dir="./twitter_cache")
#We could transform one tweet with:
#m3twitter.transform_jsonl_object(tweet)

#But, let's transform the whole file.
m3twitter.transform_jsonl(input_file="./twitter_cache/example_tweets.jsonl",output_file="./twitter_cache/m3_input.jsonl")

print("---An entry in the m3 input file---")
with open("./twitter_cache/m3_input.jsonl","r") as fh:
	m3example=json.loads(fh.readline())
	pprint.pprint(m3example)

print("---Running m3 on this input example--")
pprint.pprint(m3twitter.infer([m3example]))


print("---Running m3 on the full file--")
pprint.pprint(m3twitter.infer("./twitter_cache/m3_input.jsonl"))


