#!/usr/bin/env python3

# Script to merge a sharded dataset stored in GCS.

import argparse
import os
from typing import List, Tuple
import urllib

from google.cloud import storage


def list_folder(input_path: str) -> Tuple[List[str], List[str]]:
    """Return a list of files and directories in the given folder.
    Works for local files and GCS buckets.

    Returns:
        A tuple of the form (files, dirs).
    """
    if input_path.startswith("gs://"):
        client = storage.Client()
        result = urllib.parse.urlparse(input_path)
        bucket = result.netloc
        path = result.path
        if path.startswith("/"):
            path = path[1:]
        if not path.endswith("/"):
            path = path + "/"
        blobs = client.list_blobs(bucket, prefix=path, delimiter="/")

        # This is necessary for some reason to get the list of prefixes.
        files = list(blobs)
        dirs = [f"gs://{bucket}/{prefix}" for prefix in blobs.prefixes]
    else:
        all_files = [os.path.join(input_path, x) for x in os.listdir(input_path)]
        files = [x for x in all_files if os.path.isfile(x)]
        dirs = [x for x in all_files if os.path.isdir(x)]
    return (files, dirs)


def collect_dataset(input_path: str, output_path: str):
    """Collect the dataset stored at the given input path and write the merged
    dataset to the output path."""

    _, input_shards = list_folder(input_path)
    # XXX MDW STOPPED HERE.



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Path to GCS bucket to use as input. "
        "Example: gs://bricklens-datasets/renders/2021-09-11T20:54:10",
        required=True,
    )
    parser.add_argument(
        "--output", help="Path to GCS bucket or local file path to use as output."
    )
    args = parser.parse_args()

    output = args.output or args.input + "/merged"
    collect_dataset(args.input, output)


if __name__ == "__main__":
    main()
