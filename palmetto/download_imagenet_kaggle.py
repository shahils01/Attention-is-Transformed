#!/usr/bin/env python3
"""Resume the authorized Kaggle ImageNet competition bulk download."""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request


COMPETITION = "imagenet-object-localization-challenge"
API_URL = "https://www.kaggle.com/api/v1/competitions/data/download-all/{}".format(
    COMPETITION
)
TOKEN_PATH = os.path.expanduser("~/.kaggle/access_token")
ARCHIVE_NAME = COMPETITION + ".zip"
USER_AGENT = "gt-mha-imagenet-download/1.0"
CHUNK_SIZE = 8 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def authorized_redirect():
    with open(TOKEN_PATH, "r") as handle:
        token = handle.read().strip()
    if not token:
        raise RuntimeError("Kaggle access token is empty")

    request = urllib.request.Request(
        API_URL,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/octet-stream",
            "User-Agent": USER_AGENT,
        },
    )
    opener = urllib.request.build_opener(NoRedirect)
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise
        location = error.headers.get("Location")
        if not location:
            raise RuntimeError("Kaggle redirect did not include a download URL")
        return location
    else:
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise RuntimeError("Kaggle did not return the expected download redirect")
        return location


def open_payload(url, offset):
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = "bytes={}-".format(offset)
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=120
    )


def expected_total(response, offset):
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/(\d+)$", content_range)
    if match:
        return int(match.group(1))
    length = response.headers.get("Content-Length")
    if length is None:
        return None
    length = int(length)
    return offset + length if getattr(response, "status", 200) == 206 else length


def human_size(value):
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return "{:.2f} {}".format(amount, unit)
        amount /= 1024


def probe():
    url = authorized_redirect()
    request = urllib.request.Request(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT},
    )
    response = urllib.request.urlopen(request, timeout=120)
    total = expected_total(response, 0)
    response.close()
    print("Authorized bulk archive: {} ({})".format(ARCHIVE_NAME, human_size(total)))
    return total


def download(destination):
    os.makedirs(destination, exist_ok=True)
    partial = os.path.join(destination, ARCHIVE_NAME + ".partial")
    final = os.path.join(destination, ARCHIVE_NAME)
    metadata = os.path.join(destination, ARCHIVE_NAME + ".metadata.json")

    if os.path.isfile(final):
        print("Archive already complete: {} ({})".format(final, human_size(os.path.getsize(final))))
        return

    offset = os.path.getsize(partial) if os.path.exists(partial) else 0
    url = authorized_redirect()
    response = open_payload(url, offset)
    status = getattr(response, "status", response.getcode())
    if offset and status != 206:
        response.close()
        raise RuntimeError("Server did not honor resume request; partial archive was preserved")

    total = expected_total(response, offset)
    with open(metadata, "w") as handle:
        json.dump(
            {
                "competition": COMPETITION,
                "archive": ARCHIVE_NAME,
                "expected_bytes": total,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    mode = "ab" if offset else "wb"
    downloaded = offset
    started = time.time()
    last_report = started
    print("Downloading from {} of {}".format(human_size(offset), human_size(total)), flush=True)
    with open(partial, mode) as output:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_report >= 60:
                elapsed = max(now - started, 1)
                speed = (downloaded - offset) / elapsed
                print(
                    "Downloaded {} of {} at {}/s".format(
                        human_size(downloaded), human_size(total), human_size(speed)
                    ),
                    flush=True,
                )
                last_report = now
        output.flush()
        os.fsync(output.fileno())
    response.close()

    if total is not None and downloaded != total:
        raise RuntimeError(
            "Incomplete archive: downloaded {} of {} bytes; rerun to resume".format(
                downloaded, total
            )
        )
    os.replace(partial, final)
    print("Download complete: {} ({})".format(final, human_size(downloaded)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--destination", default="/scratch/shahils/imagenet/archive")
    args = parser.parse_args()
    if args.probe:
        probe()
    else:
        download(args.destination)


if __name__ == "__main__":
    main()
