#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os

import numpy as np
import pytest

CHUNK_SIZE = 8192

os.environ["KERAS_BACKEND"] = "torch"

# Enforce CUDA deterministic behaviour which causes known non-determinism
# issues in RNN and LSTM
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


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

    # torch requires float32
    neural_data = neural_data.astype("float32")
    vels_binned = vels_binned.astype("float32")

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


def test_wiener_filter_sklearn(split_train_test):

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from Neural_Decoding.preprocessing_funcs import LagMat

    X_train, y_train, X_val, y_val = split_train_test

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bin_before=6, bin_current=1, bin_after=6, flat=True)),
            ("linear", LinearRegression()),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_wf = r2_score(y_val, y_val_pred, multioutput="raw_values")

    assert R2s_wf == pytest.approx([0.72457168, 0.71731407], rel=0.005)


def test_wiener_cascade(set_up_train_test):

    from Neural_Decoding.decoders import WienerCascadeDecoder
    from Neural_Decoding.metrics import get_R2

    X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid = set_up_train_test

    # Declare model
    model_wc = WienerCascadeDecoder(degree=3)

    # Fit model
    model_wc.fit(X_flat_train, y_train)

    # Get predictions
    y_valid_predicted_wc = model_wc.predict(X_flat_valid)

    # Get metric of fit
    R2s_wc = get_R2(y_valid, y_valid_predicted_wc)

    assert R2s_wc == pytest.approx([0.73127717, 0.73370796])


def test_wiener_cascade_sklearn(split_train_test):

    from sklearn.metrics import r2_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from Neural_Decoding.nonlinear import WienerCascade
    from Neural_Decoding.preprocessing_funcs import LagMat

    X_train, y_train, X_val, y_val = split_train_test

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bin_before=6, bin_current=1, bin_after=6, flat=True)),
            ("wc", MultiOutputRegressor(WienerCascade(degree=3))),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_wc = r2_score(y_val, y_val_pred, multioutput="raw_values")

    assert R2s_wc == pytest.approx([0.73127717, 0.73370796], rel=0.005)


def test_xgboost(set_up_train_test):

    from Neural_Decoding.decoders import XGBoostDecoder
    from Neural_Decoding.metrics import get_R2

    X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid = set_up_train_test

    # Declare model
    model_xgb = XGBoostDecoder(max_depth=3, num_round=200, eta=0.3, gpu=-1)

    # Fit model
    model_xgb.fit(X_flat_train, y_train)

    # Get predictions
    y_valid_predicted_xgb = model_xgb.predict(X_flat_valid)

    # Get metric of fit
    R2s_xgb = get_R2(y_valid, y_valid_predicted_xgb)

    assert R2s_xgb == pytest.approx([0.75403802, 0.76625732], rel=1e-4)


def test_xgboost_sklearn(split_train_test):

    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    from Neural_Decoding.preprocessing_funcs import LagMat

    X_train, y_train, X_val, y_val = split_train_test

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bin_before=6, bin_current=1, bin_after=6, flat=True)),
            ("xgb", XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.3)),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_xgb = r2_score(y_val, y_val_pred, multioutput="raw_values")

    assert R2s_xgb == pytest.approx([0.75403802, 0.76625732], rel=0.005)


def test_svr(set_up_train_test):

    from Neural_Decoding.decoders import SVRDecoder
    from Neural_Decoding.metrics import get_R2

    X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid = set_up_train_test

    # The SVR works much better when the y values are normalized,
    # so we first z-score the y values
    # They have previously been zero-centered, so we will just divide by the stdev
    # (of the training set)
    y_train_std = np.nanstd(y_train, axis=0)
    y_zscore_train = y_train / y_train_std
    y_zscore_valid = y_valid / y_train_std

    # Declare model
    model_svr = SVRDecoder(C=5, max_iter=1000)

    # Fit model
    model_svr.fit(X_flat_train, y_zscore_train)

    # Get predictions
    y_zscore_valid_predicted_svr = model_svr.predict(X_flat_valid)

    # Get metric of fit
    R2s_svr = get_R2(y_zscore_valid, y_zscore_valid_predicted_svr)

    # Not using notebook value to save time on iterations
    assert R2s_svr == pytest.approx([0.74705083, 0.76820803])


def test_svr_sklearn(split_train_test):

    import warnings

    from sklearn.compose import TransformedTargetRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import r2_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    from Neural_Decoding.preprocessing_funcs import LagMat

    # specifically for svr
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    X_train, y_train, X_val, y_val = split_train_test

    # standardising y in addition to X
    pipe = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lagmat", LagMat(bin_before=6, bin_current=1, bin_after=6, flat=True)),
                ("svr", MultiOutputRegressor(SVR(max_iter=1000, C=5))),
            ]
        ),
        transformer=StandardScaler(),
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_svr = r2_score(y_val, y_val_pred, multioutput="raw_values")

    assert R2s_svr == pytest.approx([0.74705083, 0.76820803], rel=0.01)


def test_dnn(set_up_train_test):

    from keras.utils import set_random_seed

    from Neural_Decoding.decoders import DenseNNDecoder
    from Neural_Decoding.metrics import get_R2

    set_random_seed(99)

    X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid = set_up_train_test

    # Declare model
    model_dnn = DenseNNDecoder(units=400, dropout=0.25, num_epochs=10)

    # Fit model
    model_dnn.fit(X_flat_train, y_train)

    # Get predictions
    y_valid_predicted_dnn = model_dnn.predict(X_flat_valid)

    # Get metric of fit
    R2s_dnn = get_R2(y_valid, y_valid_predicted_dnn)

    assert R2s_dnn == pytest.approx([0.82578506, 0.84818598], rel=0.05)


def test_dnn_sklearn(split_train_test):

    import torch
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from skorch import NeuralNetRegressor
    from torch import optim

    from Neural_Decoding.nn import FNN
    from Neural_Decoding.preprocessing_funcs import LagMat

    torch.manual_seed(99)

    X_train, y_train, X_val, y_val = split_train_test

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bin_before=6, bin_current=1, bin_after=6, flat=True)),
            (
                "fnn",
                NeuralNetRegressor(
                    module=FNN,
                    lr=0.001,
                    iterator_train__shuffle=True,
                    optimizer=optim.Adam,
                    batch_size=32,
                    module__n_targets=y_train.shape[1],
                    module__num_units=400,
                    module__frac_dropout=0.25,
                    module__n_layers=2,
                    max_epochs=10,
                    verbose=0,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_dnn = r2_score(y_val, y_val_pred, multioutput="raw_values")

    assert R2s_dnn == pytest.approx([0.82578506, 0.84818598], rel=0.05)


def test_rnn(set_up_train_test):

    from keras.utils import set_random_seed

    from Neural_Decoding.decoders import SimpleRNNDecoder
    from Neural_Decoding.metrics import get_R2

    set_random_seed(99)

    X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid = set_up_train_test

    # Declare model
    model_rnn = SimpleRNNDecoder(units=400, dropout=0, num_epochs=5)

    # Fit model
    model_rnn.fit(X_train, y_train)

    # Get predictions
    y_valid_predicted_rnn = model_rnn.predict(X_valid)

    # Get metric of fit
    R2s_rnn = get_R2(y_valid, y_valid_predicted_rnn)

    # not using notebook values because cannot match it without changing
    # the original code
    assert R2s_rnn == pytest.approx([0.7816605, 0.7862867], rel=0.05)


def test_rnn_sklearn(split_train_test):

    import torch
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from skorch import NeuralNetRegressor
    from torch import optim

    from Neural_Decoding.nn import RNN
    from Neural_Decoding.preprocessing_funcs import LagMat

    torch.manual_seed(99)

    X_train, y_train, X_val, y_val = split_train_test

    # not flat X
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bin_before=6, bin_current=1, bin_after=6, flat=False)),
            (
                "rnn",
                NeuralNetRegressor(
                    module=RNN,
                    lr=0.001,
                    iterator_train__shuffle=True,
                    optimizer=optim.RMSprop,
                    batch_size=32,
                    module__n_targets=y_train.shape[1],
                    module__num_units=400,
                    module__frac_dropout=0,
                    max_epochs=10,
                    verbose=0,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_rnn = r2_score(y_val, y_val_pred, multioutput="raw_values")

    # use different values from sklearn because random seeds not transferrable
    # between keras and torch
    assert R2s_rnn == pytest.approx([0.7513478, 0.7245656], rel=0.005)


@pytest.mark.long
def test_gru(set_up_train_test):

    from keras.utils import set_random_seed

    from Neural_Decoding.decoders import GRUDecoder
    from Neural_Decoding.metrics import get_R2

    set_random_seed(99)

    X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid = set_up_train_test

    # Declare model
    model_gru = GRUDecoder(units=400, dropout=0, num_epochs=5)

    # Fit model
    model_gru.fit(X_train, y_train)

    # Get predictions
    y_valid_predicted_gru = model_gru.predict(X_valid)

    # Get metric of fit
    R2s_gru = get_R2(y_valid, y_valid_predicted_gru)

    assert R2s_gru == pytest.approx([0.83770423, 0.83575681], rel=0.05)


@pytest.mark.long
def test_gru_sklearn(split_train_test):

    import torch
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from skorch import NeuralNetRegressor
    from torch import optim

    from Neural_Decoding.nn import GRU
    from Neural_Decoding.preprocessing_funcs import LagMat

    torch.manual_seed(99)

    X_train, y_train, X_val, y_val = split_train_test

    # not flat X
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bin_before=6, bin_current=1, bin_after=6, flat=False)),
            (
                "gru",
                NeuralNetRegressor(
                    module=GRU,
                    lr=0.001,
                    iterator_train__shuffle=True,
                    optimizer=optim.RMSprop,
                    batch_size=32,
                    module__n_targets=y_train.shape[1],
                    module__num_units=400,
                    module__frac_dropout=0,
                    max_epochs=5,
                    verbose=0,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_gru = r2_score(y_val, y_val_pred, multioutput="raw_values")

    # use different values from sklearn because random seeds not transferrable
    # between keras and torch
    assert R2s_gru == pytest.approx([0.7294793, 0.7182585], rel=0.05)


@pytest.mark.long
@pytest.mark.skip(
    reason="Keras/PyTorch certain version memory leak causes test to crash."
)
def test_lstm(set_up_train_test):

    from keras.utils import set_random_seed

    from Neural_Decoding.decoders import LSTMDecoder
    from Neural_Decoding.metrics import get_R2

    set_random_seed(99)

    X_train, X_flat_train, y_train, X_valid, X_flat_valid, y_valid = set_up_train_test

    # Declare model
    model_lstm = LSTMDecoder(units=400, dropout=0, num_epochs=5, verbose=1)

    # Fit model
    model_lstm.fit(X_train, y_train)

    # Get predictions
    y_valid_predicted_lstm = model_lstm.predict(X_valid)

    # Get metric of fit
    R2s_lstm = get_R2(y_valid, y_valid_predicted_lstm)

    assert R2s_lstm == pytest.approx([0.84809856, 0.84108359], rel=0.05)


@pytest.mark.long
def test_lstm_sklearn(split_train_test):

    import torch
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from skorch import NeuralNetRegressor
    from torch import optim

    from Neural_Decoding.nn import LSTM
    from Neural_Decoding.preprocessing_funcs import LagMat

    torch.manual_seed(99)

    X_train, y_train, X_val, y_val = split_train_test

    # not flat X
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lagmat", LagMat(bin_before=6, bin_current=1, bin_after=6, flat=False)),
            (
                "lstm",
                NeuralNetRegressor(
                    module=LSTM,
                    lr=0.001,
                    iterator_train__shuffle=False,
                    optimizer=optim.RMSprop,
                    batch_size=32,
                    module__n_targets=y_train.shape[1],
                    module__num_units=400,
                    module__frac_dropout=0,
                    max_epochs=5,
                    verbose=0,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    R2s_lstm = r2_score(y_val, y_val_pred, multioutput="raw_values")

    # use different values from sklearn because random seeds not transferrable
    # between keras and torch
    assert R2s_lstm == pytest.approx([0.7530093, 0.73186845], rel=0.005)
