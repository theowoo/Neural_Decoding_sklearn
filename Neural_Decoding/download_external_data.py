#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import re
import warnings

import requests
import rich_click as click
from rich import print
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

CHUNK_SIZE = 8192


def download_data(
    data_url: str,
    output_dir: str,
    filename: str = None,
    overwrite=None,
):
    """
    Download external data from a given URL to a specified output directory.

    Parameters
    ----------
    data_url : str, optional
        The URL of the data to download. If not provided, uses a default test
        data URL.
    output_dir : str, optional
        The directory where the data should be saved.
    filename : str, optional
        The name of the file to save the downloaded data as. If not provided,
        extracts the filename from the content disposition header.
    overwrite : bool, optional
        Whether to overwrite an existing file with the same name. If None,
        prompts the user.

    Returns
    -------
    str
        The path to the saved file.
    """

    if output_dir is None:
        output_dir = os.getcwd()

    if not os.path.isdir(output_dir):
        mkdir = click.confirm(f"Make a new directory? {output_dir}")
        if mkdir:
            os.mkdir(output_dir)
        else:
            print(f"Output directory does not exist: {output_dir}")

    tmp_filename = os.path.join(output_dir, ".data.avi")

    with open(tmp_filename, "wb") as f:
        with requests.get(data_url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers["content-length"])

            if filename is None:
                d = r.headers["content-disposition"]
                filename = re.findall('filename="(.+)"', d)[0]
            output_file = os.path.join(output_dir, filename)

            # Check if user wants to overwrite existing file
            if overwrite is None and os.path.isfile(filename):
                overwrite = click.confirm(f"File already exists, overwrite? {filename}")

            if overwrite:
                tqdm_params = {
                    "desc": "Downloading data",
                    "total": total,
                    "miniters": 1,
                    "unit": "B",
                    "unit_scale": True,
                    "unit_divisor": 1024,
                }

                with tqdm(**tqdm_params) as pb:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        pb.update(len(chunk))
                        f.write(chunk)

    if overwrite:
        os.rename(tmp_filename, output_file)
        click.secho(f"Download completed: {filename}", fg="green")

    return output_file


@click.command()
@click.argument("data-urls", required=False, nargs=-1)
@click.option("--output-dir", "-o", help="Output directory")
def download_data_cmd(data_urls: str, output_dir: str):
    """Download a data from a given URL to a specified output directory."""
    for url in data_urls:
        download_data(url, output_dir)


if __name__ == "__main__":
    download_data_cmd()
