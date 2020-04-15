# coding: utf-8
"""
Tweet a image with text via Twitter API.
Prepare 'config.py' which describes access token.

Args:
 1. Image path
 2. Tweet text

Example:
$ python tweet_image.py fig/2020_04_15_00_00_problem.png "Today's puzzle."
"""

import sys
import json
import config
from requests_oauthlib import OAuth1Session

args = sys.argv
ck = config.consumer_key
cs = config.consumer_secret
at = config.access_token
ats = config.access_token_secret

twitter = OAuth1Session(ck, cs, at, ats)  #OAuth

url_text = "https://api.twitter.com/1.1/statuses/update.json"
url_media = "https://upload.twitter.com/1.1/media/upload.json"

# Post a image
files = {"media" : open( args[1], 'rb')}
req_media = twitter.post(url_media, files = files)

# Check response
if req_media.status_code != 200:
    print ("Failed to post image: %s", req_media.text)
    exit()

# Get Media ID
media_id = json.loads(req_media.text)['media_id']
print ("Media ID: %d" % media_id)

# Tweet a text with Media ID
params = {'status': args[2], "media_ids": [media_id]}
req_media = twitter.post(url_text, params = params)

# Check response again
if req_media.status_code != 200:
    print ("Failure: %s", req_text.text)
    exit()

print ("Posted")
