import types
from typing import Union

from math import ceil, floor
import torch as t
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import CARETUP

# numpy.fft, scipy.fft, and torch.fft each have the same basic set of methods
from scipy import fft as sf
from torch import fft as tf
dimension = 1


def fft(p: types.ModuleType, 
        f: Union[t.tensor, np.ndarray],
        inverse: bool = False):
    """
    Given an fft package p, do the fft of f using p.fft

    Why fftshift is needed: See https://stackoverflow.com/questions/5398304/fourier-transform-of-a-gaussian-is-not-a-gaussian-but-thats-wrong-python/5398901#5398901

    Why norm='forward' is needed: By default, the forward transform is not multiplied by Δ. See https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft and compare with https://shiraz-k.com/posts/harmonic-analysis/, where ωNΔ=k.
    """
    transform = p.ifft if inverse else p.fft
    return p.fftshift(transform(p.ifftshift(f), norm='forward'))

def plot_coeffs(f_hat, n_samples, color):
    # When n_samples is even, \hat f(+n/2) = \hat f(-n/2)
    indices = np.arange(-floor(n_samples/2), ceil(n_samples/2), 1)

    plt.scatter(indices, np.real(f_hat), label=r"$\hat f(\lambda)$",
                s=15.0, color=color, marker=CARETUP)
    plt.vlines(indices, 0, np.real(f_hat), color=color, 
               alpha=0.7, linewidth=0.8)
    return
               
if dimension == 1:
    # one-dimensional fft using scipy
    mu = 2.0
    length = 15.
    var = 2.5
    
    def gauss(x):
        scaling = 1/np.sqrt(2*np.pi*var)
        return scaling * np.exp(-(x - mu)**2/(2*var))

    # Transform does not actually depend on length: Yes, we multiply by 
    # 2π/length when computing coefficients, but then the discretization 
    # involves the integration measure dx, which scales by length/2π
    n_samples = 19
    n_subsamples = 10
    x = np.linspace(mu - length*0.5, mu + length*0.5, n_samples*n_subsamples)
    f = np.array([gauss(x) for x in x])

    f_hat = fft(sf, f[::n_subsamples])
    g = fft(sf, f_hat, inverse=True)

    cmap = plt.get_cmap("tab10").colors
    plt.plot(x, f, label=r"$f(x)$", color=cmap[3])
    plt.scatter(x[::n_subsamples], np.real(g), color='k', 
                s=15.0, label=r"Reconstructed $f(x)$")
    plot_coeffs(f_hat, n_samples, 'k')
    plt.legend()
    plt.xlim(-x[-1], x[-1])
    plt.ylim(bottom=0.0)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.show()

elif dimension == 2:
    mu = t.tensor([0.1, 0.4])
    sigma = 0.1
    dimensions = [10., 5.]
    raise NotImplementedError()