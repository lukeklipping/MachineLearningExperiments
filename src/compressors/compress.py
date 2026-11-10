"""
Lossy compressors for batches of time-series windows.

Every compressor takes X with shape (n_windows, window_len) and a target
`ratio` (keep roughly window_len / ratio values), and returns:
    recon         the reconstructed windows, same shape as X
    bits_per_win  storage cost of ONE compressed window, in bits

The reported compression ratio is  window_len * 64 / bits_per_win, so it
includes the cost of storing coefficient positions for the sparse methods
(wavelet, dct). That is why their real ratio is lower than the target.

Each window is compressed independently, so the reconstruction at one time
step never contains information from the future of that window.
"""
import math

import numpy as np
import pywt
from scipy.fft import dct, idct

FLOAT_BITS = 64


def _index_bits(n):
    return max(1, math.ceil(math.log2(n)))


def _keep_top_k(coefs, k):
    """Zero everything except the k largest-magnitude values in each row."""
    if k >= coefs.shape[1]:
        return coefs
    idx = np.argpartition(np.abs(coefs), -k, axis=1)[:, -k:]
    rows = np.arange(coefs.shape[0])[:, None]
    out = np.zeros_like(coefs)
    out[rows, idx] = coefs[rows, idx]
    return out


def dct_compress(X, ratio):
    """Discrete cosine transform: keep the k strongest frequency components."""
    n = X.shape[1]
    k = max(1, round(n / ratio))
    coefs = dct(X, type=2, norm="ortho", axis=1)
    recon = idct(_keep_top_k(coefs, k), type=2, norm="ortho", axis=1)
    return recon, k * (FLOAT_BITS + _index_bits(n))


def wavelet_compress(X, ratio, wavelet="db4", level=4):
    """Wavelet transform: keep the k largest wavelet coefficients."""
    n = X.shape[1]
    level = min(level, pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedecn(X, wavelet, mode="periodization", level=level, axes=(1,))
    arr, slices = pywt.coeffs_to_array(coeffs, axes=(1,))
    m = arr.shape[1]
    k = max(1, round(n / ratio))
    arr = _keep_top_k(arr, k)
    coeffs = pywt.array_to_coeffs(arr, slices, output_format="wavedecn")
    recon = pywt.waverecn(coeffs, wavelet, mode="periodization", axes=(1,))[:, :n]
    return recon, k * (FLOAT_BITS + _index_bits(m))


def paa_compress(X, ratio):
    """Piecewise Aggregate Approximation: replace each segment by its mean."""
    n = X.shape[1]
    m = max(1, min(n, round(n / ratio)))
    edges = np.linspace(0, n, m + 1).round().astype(int)
    lens = np.diff(edges)
    means = np.add.reduceat(X, edges[:-1], axis=1) / lens
    return np.repeat(means, lens, axis=1), m * FLOAT_BITS


def quantize_compress(X, ratio):
    """Uniform quantization per window (baseline). bits = 64 / ratio."""
    n = X.shape[1]
    bits = max(1, round(FLOAT_BITS / ratio))
    lo = X.min(axis=1, keepdims=True)
    span = X.max(axis=1, keepdims=True) - lo
    span[span == 0] = 1.0
    levels = 2 ** bits - 1
    codes = np.round((X - lo) / span * levels)
    recon = codes / levels * span + lo
    return recon, n * bits + 2 * FLOAT_BITS      # + stored min and max


COMPRESSORS = {
    "wavelet": wavelet_compress,
    "dct": dct_compress,
    "paa": paa_compress,
    "quantize": quantize_compress,
}


def compress_windows(X, method, ratio):
    """Returns (reconstructed windows, effective compression ratio)."""
    recon, bits = COMPRESSORS[method](X, ratio)
    return recon, X.shape[1] * FLOAT_BITS / bits
