#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
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


@pytest.fixture(scope="session")
def get_history(get_data):

    from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

    neural_data, vels_binned = get_data

    bins_before = 6
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 6

    X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)

    y = vels_binned

    return X, y, bins_before, bins_current, bins_after


def test_get_spikes_with_history(get_history):

    X, _, _, _, _ = get_history

    assert X.shape == (61339, 13, 52)


@pytest.fixture(scope="session")
def get_lagmat(get_data):

    from Neural_Decoding.preprocessing_funcs import LagMat

    neural_data, vels_binned = get_data

    bins_before = 6
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 6

    X = LagMat(bins_before, bins_current, bins_after).fit_transform(neural_data)

    y = vels_binned

    return X, y, bins_before, bins_current, bins_after


def test_lagmat(get_lagmat):

    from Neural_Decoding.preprocessing_funcs import LagMat

    A = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
    )

    A_lag = LagMat(1, 1, 2).fit_transform(A)

    assert np.all(
        A_lag[..., 0]
        == np.array(
            [
                [0, 0, 1, 2],
                [0, 1, 2, 3],
                [1, 2, 3, 0],
                [2, 3, 0, 0],
            ]
        )
    )

    A_lag = LagMat(3, 0, 0).fit_transform(A)

    assert np.all(
        A_lag[..., 0]
        == np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 2],
            ]
        )
    )

    X, _, _, _, _ = get_lagmat

    assert X.shape == (61339, 13, 52)


def test_lagmat_against_get_spikes_with_history(get_history, get_lagmat):

    X, _, bins_before, _, bins_after = get_history
    X_lagmat, _, _, _, _ = get_lagmat

    # the only difference in behaviour is lagmat fill with 0
    assert np.all(X_lagmat[bins_before:-bins_after] == X[bins_before:-bins_after])


@pytest.fixture(scope="session")
def set_up_train_test(get_history):

    X, y, bins_before, bins_current, bins_after = get_history

    X_flat = X.reshape(X.shape[0], (X.shape[1] * X.shape[2]))

    training_range = [0, 0.7]
    testing_range = [0.7, 0.85]
    valid_range = [0.85, 1]

    num_examples = X.shape[0]

    # Note that each range has a buffer of"bins_before" bins at the beginning,
    # and "bins_after" bins at the end
    # This makes it so that the different sets don't include overlapping neural data
    training_set = np.arange(
        int(np.round(training_range[0] * num_examples)) + bins_before,
        int(np.round(training_range[1] * num_examples)) - bins_after,
    )
    testing_set = np.arange(
        int(np.round(testing_range[0] * num_examples)) + bins_before,
        int(np.round(testing_range[1] * num_examples)) - bins_after,
    )
    valid_set = np.arange(
        int(np.round(valid_range[0] * num_examples)) + bins_before,
        int(np.round(valid_range[1] * num_examples)) - bins_after,
    )

    # Get training data
    X_train = X[training_set, :, :]
    X_flat_train = X_flat[training_set, :]
    y_train = y[training_set, :]

    # Get testing data
    X_test = X[testing_set, :, :]
    X_flat_test = X_flat[testing_set, :]
    y_test = y[testing_set, :]

    # Get validation data
    X_valid = X[valid_set, :, :]
    X_flat_valid = X_flat[valid_set, :]
    y_valid = y[valid_set, :]

    # Z-score "X" inputs.
    X_train_mean = np.nanmean(X_train, axis=0)
    X_train_std = np.nanstd(X_train, axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    X_valid = (X_valid - X_train_mean) / X_train_std

    # Z-score "X_flat" inputs.
    X_flat_train_mean = np.nanmean(X_flat_train, axis=0)
    X_flat_train_std = np.nanstd(X_flat_train, axis=0)
    X_flat_train = (X_flat_train - X_flat_train_mean) / X_flat_train_std
    X_flat_test = (X_flat_test - X_flat_train_mean) / X_flat_train_std
    X_flat_valid = (X_flat_valid - X_flat_train_mean) / X_flat_train_std

    # Zero-center outputs
    y_train_mean = np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean
    y_test = y_test - y_train_mean
    y_valid = y_valid - y_train_mean

    return X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid


def test_wiener_filter(set_up_train_test):

    from Neural_Decoding.decoders import WienerFilterDecoder
    from Neural_Decoding.metrics import get_R2

    X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid = set_up_train_test

    # Declare model
    model_wf = WienerFilterDecoder()

    # Fit model
    model_wf.fit(X_flat_train, y_train)

    # Get predictions
    y_valid_predicted_wf = model_wf.predict(X_flat_valid)

    # Get metric of fit
    R2s_wf = get_R2(y_valid, y_valid_predicted_wf)

    assert R2s_wf == pytest.approx([0.72457168, 0.71731407])


@pytest.fixture(scope="session")
def split_train_test(get_data):
    from sklearn.model_selection import train_test_split

    X, y = get_data

    X_train_test, X_val = train_test_split(X, train_size=0.85, shuffle=False)
    y_train_test, y_val = train_test_split(y, train_size=0.85, shuffle=False)

    X_train, X_test = train_test_split(
        X_train_test, train_size=0.7 / 0.85, shuffle=False
    )
    y_train, y_test = train_test_split(
        y_train_test, train_size=0.7 / 0.85, shuffle=False
    )

    return X_train, y_train, X_val, y_val


def test_wiener_filter_with_zscore_using_pipeline(split_train_test):

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from Neural_Decoding.preprocessing_funcs import LagMat

    bins_before = 6
    bins_current = 1
    bins_after = 6

    X_train, y_train, X_val, y_val = split_train_test

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bins_before, bins_current, bins_after, flat=True)),
            ("linear", LinearRegression()),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_wf = r2_score(y_val, y_val_pred, multioutput="raw_values")

    assert R2s_wf == pytest.approx([0.72457168, 0.71731407], rel=0.005)
