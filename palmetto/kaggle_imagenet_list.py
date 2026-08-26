#!/usr/bin/env python3
import json
import os
import urllib.request


TOKEN_PATH = os.path.expanduser("~/.kaggle/access_token")
URL = (
    "https://www.kaggle.com/api/v1/competitions/data/list/"
    "imagenet-object-localization-challenge?pageSize=200"
)
OUTPUT = "/scratch/shahils/imagenet/competition-files.json"


with open(TOKEN_PATH, "r") as handle:
    token = handle.read().strip()

if not token:
    raise SystemExit("Kaggle access token is empty")

request = urllib.request.Request(
    URL,
    headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/json",
        "User-Agent": "gt-mha-imagenet-preflight/1.0",
    },
)

with urllib.request.urlopen(request, timeout=60) as response:
    payload = response.read()

data = json.loads(payload.decode("utf-8"))
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)

files = data.get("files") or data.get("datasetFiles") or []
print("Kaggle access verified; {} files listed.".format(len(files)))
for item in files:
    name = item.get("name") or item.get("nameNullable") or "<unnamed>"
    size = item.get("totalBytes") or item.get("totalBytesNullable") or "?"
    print("{}\t{}".format(name, size))
