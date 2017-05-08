import numpy as np
import pylab as pl

x = np.arange(0, 2*np.pi, 2*np.pi/1000)
y = np.where(x < np.pi, np.sin(2*x), 0) + np.where(x >= np.pi, np.sin(4*x), 0)
fy = np.fft.fft(y)/len(y)
fy = np.abs(fy)