"""One builder for the autoencoders of the course (Keras 3, JAX backend): depth, activation, normalisation, residual
blocks, sparsity, input noise, and a temporal body (1-D convolution, LSTM, GRU) for windows.

    model = build_autoencoder((71,), widths=(64, 32), code=8, residual=True)       # tabular
    model = build_autoencoder((96, 1), kind="conv", widths=(32, 16), code=8)       # windows of a series
    fit_autoencoder(model, X_normal)                                              # normal data only
    err = reconstruction_error(model, X)                                          # one score per row

What the arguments mean and why (class 03_02):
  widths      sizes of the hidden layers of the encoder, outermost first; the decoder mirrors them (a funnel).
  code        size of the bottleneck. Anomaly detection needs it narrower than the input, or the AE learns the identity
              (conv only: None skips the 1x1 code layer and the last strided stage is the code).
  activation  "silu" (smooth, used by ASAP 2.0) or "relu"; the output layer is linear.
  norm        "layer" | "batch" | None, before each hidden block (stabilises deeper stacks).
  residual    skip connections (He et al. 2016): each block computes x + F(x), so depth cannot make the signal worse.
  l1          L1 penalty on the *code activations*: a sparse AE (few active units per row).
  noise       Gaussian noise on the input during training only, target stays clean: a denoising AE.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")  # before importing keras

import keras
import numpy as np
from keras import layers

KINDS = ("dense", "conv", "lstm", "gru")


def _norm(x, norm):
    if norm == "layer":
        return layers.LayerNormalization()(x)
    if norm == "batch":
        return layers.BatchNormalization()(x)
    return x


def _dense_block(x, width, activation, norm, residual, dropout):
    """Dense(width) -> activation; with `residual`, one more Dense on top and a skip: out = h + F(h)."""
    h = layers.Dense(width)(_norm(x, norm))
    h = layers.Activation(activation)(h)
    if residual:
        r = layers.Dense(width, activation=activation)(_norm(h, norm))
        r = layers.Dense(width)(r)
        h = layers.Add()([h, r])
    return layers.Dropout(dropout)(h) if dropout else h


def _conv_block(x, filters, k, activation, norm, residual, stride=1, transpose=False):
    conv = layers.Conv1DTranspose if transpose else layers.Conv1D
    h = layers.Activation(activation)(conv(filters, k, strides=stride, padding="same")(_norm(x, norm)))
    if residual:
        r = layers.Activation(activation)(layers.Conv1D(filters, k, padding="same")(_norm(h, norm)))
        h = layers.Add()([h, layers.Conv1D(filters, k, padding="same")(r)])
    return h


def build_autoencoder(
    input_shape: tuple[int, ...],
    kind: str = "dense",
    widths: tuple[int, ...] = (32,),
    code: int | None = 8,
    activation: str = "silu",
    norm: str | None = None,
    residual: bool = False,
    dropout: float = 0.0,
    l1: float = 0.0,
    noise: float = 0.0,
    kernel: int = 7,
    name: str | None = None,
) -> keras.Model:
    """Autoencoder as a Keras functional model. `input_shape` is (d,) for kind "dense" and (w, d) (w steps, d channels)
    for "conv", "lstm" and "gru". For "conv" the window w must be divisible by 2**len(widths)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    reg = keras.regularizers.L1(l1) if l1 > 0 else None
    inp = layers.Input(input_shape)
    x = layers.GaussianNoise(noise)(inp) if noise else inp  # active in training only
    if kind == "dense":
        for w in widths:
            x = _dense_block(x, w, activation, norm, residual, dropout)
        x = layers.Dense(code, activation="relu" if l1 else None, activity_regularizer=reg, name="code")(x)
        for w in reversed(widths):
            x = _dense_block(x, w, activation, norm, residual, dropout)
        out = layers.Dense(input_shape[-1])(x)
    elif kind == "conv":
        w_in, d = input_shape
        if w_in % 2 ** len(widths):
            raise ValueError(f"window {w_in} must be divisible by {2 ** len(widths)}")
        for f in widths:
            x = _conv_block(x, f, kernel, activation, norm, residual, stride=2)
        if code:  # code=None: the last strided stage is the code (a bottleneck in time, not in channels)
            x = layers.Conv1D(code, 1, activation="relu" if l1 else None, activity_regularizer=reg, name="code")(x)
        for f in reversed(widths):
            x = _conv_block(x, f, kernel, activation, norm, residual, stride=2, transpose=True)
        out = layers.Conv1D(d, kernel, padding="same")(x)
    else:  # lstm | gru: the whole window is squeezed into one vector of size `code`
        w_in, d = input_shape
        rnn = layers.LSTM if kind == "lstm" else layers.GRU
        for i, u in enumerate(widths):
            x = rnn(u, return_sequences=i < len(widths) - 1)(_norm(x, norm) if i else x)
        x = layers.Dense(code, activation="relu" if l1 else None, activity_regularizer=reg, name="code")(x)
        x = layers.RepeatVector(w_in)(x)
        for u in reversed(widths):
            h = rnn(u, return_sequences=True)(_norm(x, norm))
            x = layers.Add()([x, h]) if residual and x.shape[-1] == u else h
        out = layers.TimeDistributed(layers.Dense(d))(x)
    return keras.Model(inp, out, name=name or f"{kind}_ae")


def fit_autoencoder(
    model: keras.Model,
    X: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    val: float = 0.1,
    patience: int = 8,
    seed: int = 0,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Train on normal data with Adam and the mean-squared error; `val` of the rows are held out and early stopping
    restores the best epoch (patience in epochs). Returns the Keras history."""
    keras.utils.set_random_seed(seed)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    cb = [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)] if val and patience else []
    return model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=val, callbacks=cb, verbose=verbose)


def residuals(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """The reconstruction residual x_hat - x, same shape as X: the sign and the feature of every miss."""
    return model.predict(X, batch_size=4096, verbose=0) - X


def reconstruction_error(model: keras.Model, X: np.ndarray, metric: str = "mse", per: str = "row") -> np.ndarray:
    """Anomaly score from the residuals. metric "mse" | "mae"; per "row" (one value per sample, averaged over every
    other axis), "feature" (keep the last axis) or "step" (windows: keep the time axis, shape (N, w))."""
    r = residuals(model, X)
    e = r**2 if metric == "mse" else np.abs(r)
    if per == "feature":
        return e
    if per == "step":
        return e.mean(axis=-1)
    return e.reshape(len(e), -1).mean(axis=1)
