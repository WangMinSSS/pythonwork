import numpy as np
import pylab as pl


def tri_wave(size):
    x = np.arange(0, 1, 1.0 / size)
    y = np.where(x < 0.5, x, 1 - x)
    return x, y


def fft_comb(f, n, loops=1):
    length = len(f) * loops
    data = np.zeros(length)
    index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
    for k, p in enumerate(f[:n]):
        if k != 0:
            p *= 2
        data += np.real(p) * np.cos(k * index)
        data -= np.imag(p) * np.sin(k * index)
    return index, data

fft_size = 256
x, y = tri_wave(fft_size)
fy = np.fft.fft(y) / fft_size
pl.figure()
pl.subplot(331)
pl.plot(y, label="original triangle", linewidth=1)
j = 1
for i in [0, 1, 3, 5, 7, 9]:
    index, data = fft_comb(fy, i + 1, 1)
    pl.subplot(331+j)
    pl.plot(data, label="N=%s" % i)
    j += 1
pl.show()
