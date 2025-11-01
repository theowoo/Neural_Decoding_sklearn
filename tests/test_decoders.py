#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import pytest

CHUNK_SIZE = 8192


def test_import():

    from Neural_Decoding.decoders import WienerFilterDecoder

    wf = WienerFilterDecoder()  # noqa: F841

    assert True


@pytest.fixture(scope="session")
def get_data(tmp_path_factory):

    import pickle

    from Neural_Decoding.download_external_data import download_data

    url = "https://datashare.mpcdf.mpg.de/s/FszSKgyXSGbnWeQ/download"
    external_data_dir = tmp_path_factory.mktemp("external")
    data_path = download_data(
        data_url=url,
        output_dir=external_data_dir,
        filename="example_data.pickle",
        overwrite=True,
    )

    with open(data_path, "rb") as f:
        neural_data, vels_binned = pickle.load(f, encoding="latin1")

    return neural_data, vels_binned


def test_get_spikes_with_history(get_data):

    from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

    neural_data, vels_binned = get_data

    bins_before = (
        6  # How many bins of neural data prior to the output are used for decoding
    )
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = (
        6  # How many bins of neural data after the output are used for decoding
    )

    X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)

    assert X.shape == (61339, 13, 52)
