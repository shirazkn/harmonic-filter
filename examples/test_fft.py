import numpy as np
from matplotlib import pyplot as plt

mu = 2.0
length = 10.
var = 1.
def gauss(x):
    scaling = 1/np.sqrt(2*np.pi*var)
    return scaling * np.exp(-(x - mu)**2/(2*var))

n_samples = 31
x_vals = np.linspace(mu - length*0.5, mu + length*0.5, n_samples)
f = np.array([gauss(x) for x in x_vals])

# By default, the forward transform is not multiplied by Δ.
# See https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
# and compare with https://shiraz-k.com/posts/harmonic-analysis/ where ωNΔ=k.
# Setting norm='forward' uses the convention on my blog.

# Also see https://stackoverflow.com/questions/5398304/fourier-transform-of-a-gaussian-is-not-a-gaussian-but-thats-wrong-python/5398901#5398901.

# f(x) is length-periodic, while g(x)=f(x/length) is 1-periodic. The following should be viewed as the FFT of g(x)
f_hat = np.fft.fftshift(
            np.fft.fft(
                np.fft.ifftshift(f),
                norm='forward'
                )
            )

g = np.fft.fftshift(np.fft.ifft(
    np.fft.ifftshift(f_hat), norm='forward'))

cmap = plt.get_cmap("tab10").colors
plt.plot(x_vals, f, label=r"$f(x)$", color=cmap[0])
plt.scatter(x_vals, g, label=r"Reconstructed $f(x)$", color=cmap[1])

plt.scatter(np.arange(-n_samples//2 + 1, n_samples//2 + 1, 1), np.real(f_hat), label=r"$\hat f(\lambda)$", color=cmap[2])

plt.legend()
plt.xlim(-n_samples//2 + 1, n_samples//2)
plt.ylim(bottom=0.0)
plt.axhline(y=0, color='k', linestyle='-')
plt.show()