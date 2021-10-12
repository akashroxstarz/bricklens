#!/usr/bin/env python3

# Script to merge a sharded dataset stored in GCS.

import argparse
import os
import shutil
import urllib
import tempfile
from typing import List, Tuple

from google.cloud import storage
from rich.console import Console
from rich.progress import track
from rich.progress import Progress
from rich.spinner import Spinner


console = Console()


def parse_gs_url(gs_url: str) -> Tuple[str, str]:
    """Parse a gs:// URL into bucket and path components."""
    result = urllib.parse.urlparse(gs_url)
    bucket = result.netloc
    path = result.path
    if path.startswith("/"):
        path = path[1:]
    return (bucket, path)


def list_folder(input_path: str) -> Tuple[List[str], List[str]]:
    """Return a list of files and directories in the given folder.
    Works for local files and GCS buckets.

    Returns:
        A tuple of the form (files, dirs).
    """
    if input_path.startswith("gs://"):
        client = storage.Client()
        bucket, path = parse_gs_url(input_path)
        if not path.endswith("/"):
            path = path + "/"
        blobs = client.list_blobs(bucket, prefix=path, delimiter="/")

        # This is necessary for some reason to get the list of prefixes.
        # files = list(blobs)
        files = [os.path.split(blob.name)[1] for blob in list(blobs)]
        dirs = [f"gs://{bucket}/{prefix}" for prefix in blobs.prefixes]
    else:
        all_files = [os.path.join(input_path, x) for x in os.listdir(input_path)]
        files = [x for x in all_files if os.path.isfile(x)]
        dirs = [x for x in all_files if os.path.isdir(x)]
    return (files, dirs)


def read_file(input_path: str) -> str:
    if input_path.startswith("gs://"):
        client = storage.Client()
        bucket, path = parse_gs_url(input_path)
        bucket = client.get_bucket(bucket)
        blob = storage.Blob(path, bucket)
        return blob.download_as_text()
    else:
        with open(input_path, "r") as infile:
            return infile.read()


def copy_file(input_path: str, output_path: str) -> str:
    if not input_path.startswith("gs://") and not output_path.startswith("gs://"):
        # Local -> Local.
        shutil.copyfile(input_path, output_path)
        return

    client = storage.Client()

    if input_path.startswith("gs://") and not output_path.startswith("gs://"):
        # GCS -> Local.
        bucket, path = parse_gs_url(input_path)
        bucket = client.get_bucket(bucket)
        blob = storage.Blob(path, bucket)
        blob.download_to_filename(output_path)

    elif not input_path.startswith("gs://") and output_path.startswith("gs://"):
        # Local -> GCS.
        bucket, path = parse_gs_url(output_path)
        bucket = client.get_bucket(bucket)
        blob = bucket.blob(path)
        blob.upload_from_filename(input_path)

    else:
        # GCS -> GCS.
        inbucket, inpath = parse_gs_url(input_path)
        outbucket, outpath = parse_gs_url(output_path)
        source_bucket = client.bucket(inbucket)
        source_blob = source_bucket.blob(inpath)
        destination_bucket = client.bucket(outbucket)
        source_bucket.copy_blob(source_blob, destination_bucket, outpath)


def image_path_to_label_path(image_path: str) -> str:
    sa, sb = (
        os.sep + "images" + os.sep,
        os.sep + "labels" + os.sep,
    )  # /images/, /labels/ substrings
    return image_path.replace(sa, sb, 1).replace("." + x.split(".")[-1], ".txt")


def collect_dataset(input_path: str, output_path: str):
    """Collect the dataset stored at the given input path and write the merged
    dataset to the output path."""

    _, input_shards = list_folder(input_path)

    with console.status("Reading classes..."):
        input_classfile = os.path.join(input_shards[0], "dataset/classes.txt")
        classes = read_file(input_classfile)
    with console.status("Writing classes..."):
        output_classfile = os.path.join(output_path, "classes.txt")
        copy_file(input_classfile, output_classfile)

    train_out = ""
    val_out = ""

    for index, shard in enumerate(input_shards):
        shard = os.path.join(shard, "dataset")
        console.print(f"Processing shard {index+1}/{len(input_shards)+1}...")

        with console.status("classes.txt..."):
            shard_classes = read_file(os.path.join(shard, "classes.txt"))
            assert (
                shard_classes == classes
            ), f"Classes in shard {shard} do not match that of {input_shards[0]}."

        with console.status("train.txt..."):
            train_txt = read_file(os.path.join(shard, "train.txt"))
        with console.status("val.txt..."):
            val_txt = read_file(os.path.join(shard, "val.txt"))
        with console.status("images..."):
            image_files, _ = list_folder(os.path.join(shard, "images"))
        with console.status("debug_images..."):
            debug_image_files, _ = list_folder(os.path.join(shard, "debug_images"))
        with console.status("labels..."):
            label_files, _ = list_folder(os.path.join(shard, "labels"))

        # Process train.txt.
        for image_path in track(train_txt.splitlines(), description="train files..."):
            label_path = image_path_to_label_path(image_path)

            # Copy the image file.
            image_dir, image_filename = os.path.split(image_path)
            assert (
                image_filename in image_files
            ), f"Cannot find image {image_filename} in {shard}"

            src_image = os.path.join(shard, image_path)
            dest_image = os.path.join(image_dir, f"shard_{index:05d}_{image_filename}")
            copy_file(src_image, os.path.join(output_path, dest_image))

            # Copy the label file.
            label_dir, label_filename = os.path.split(label_path)
            assert (
                label_filename in label_files
            ), f"Cannot find label file {label_filename} in {shard}"
            src_label = os.path.join(shard, label_path)
            dest_label = os.path.join(label_dir, f"shard_{index:05d}_{label_filename}")
            copy_file(src_label, os.path.join(output_path, dest_label))

            train_out += f"{dest_image}\n"

        # Process val.txt.
        for image_path in track(val_txt.splitlines(), description="val files..."):
            label_path = image_path_to_label_path(image_path)

            # Copy the image file.
            image_dir, image_filename = os.path.split(image_path)
            assert (
                image_filename in image_files
            ), f"Cannot find image {image_filename} in {shard}"

            src_image = os.path.join(shard, image_path)
            dest_image = os.path.join(image_dir, f"shard_{index:05d}_{image_filename}")
            copy_file(src_image, os.path.join(output_path, dest_image))

            # Copy the label file.
            label_dir, label_filename = os.path.split(label_path)
            assert (
                label_filename in label_files
            ), f"Cannot find label file {label_filename} in {shard}"
            src_label = os.path.join(shard, label_path)
            dest_label = os.path.join(label_dir, f"shard_{index:05d}_{label_filename}")
            copy_file(src_label, os.path.join(output_path, dest_label))

            val_out += f"{dest_image}\n"

        # Process debug images.
        for debug_image_file in track(debug_image_files, description="debug_images..."):
            src_image = os.path.join(shard, "debug_images", debug_image_file)
            dest_image = os.path.join(
                output_path, "debug_images", f"shard_{index:05d}_{debug_image_file}"
            )
            copy_file(src_image, dest_image)

        with console.status("Writing train.txt..."):
            with tempfile.NamedTemporaryFile("w", delete=False) as outfile:
                outfile.write(train_out)
                copy_file(outfile.name, os.path.join(output_path, "train.txt"))

        with console.status("Writing val.txt..."):
            with tempfile.NamedTemporaryFile("w", delete=False) as outfile:
                outfile.write(val_out)
                copy_file(outfile.name, os.path.join(output_path, "val.txt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Path to GCS or local directory to use as input. "
        "Example: gs://bricklens-datasets/renders/2021-09-11T20:54:10",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Path to GCS or local directory to use as output.",
    )
    args = parser.parse_args()

    output = args.output or os.path.join(args.input, "merged")
    collect_dataset(args.input, output)


if __name__ == "__main__":
    main()
