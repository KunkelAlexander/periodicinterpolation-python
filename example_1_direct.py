# import libraries
import numpy as np
from numpy.testing import assert_allclose as isclose
# set up test problem 
L, N, Ni = 2, 32, 2*32
dx, dxi  = L/N, L/Ni
f        = lambda tx, ty : np.sin(2*np.pi*tx) * np.cos(2*np.pi*ty)
# exclude last point in array because f(0) == f(L) 
t        = np.arange(0, N ) * dx 
tx, ty   = np.meshgrid(t, t) 
ti       = np.arange(0, Ni) * dxi
tix, tiy = np.meshgrid(ti, ti) 
# set up momentum array
k        = 2*np.pi*np.fft.fftfreq(N)*N/L
kx, ky   = np.meshgrid(k, k) 
# compute the DFT of the function
fk       = np.fft.fftn(f(tx, ty), norm="forward")

# METHOD 1: evaluate IDFT at point t
# use extra dimension because we need to evaluate sum for every 
# interpolation point in array tInt
fi1 = np.sum(fk[..., None, None]*np.exp(1j*(kx[..., None, None]*tix + ky[..., None, None]*tiy)), axis=(0, 1))

# check result
isclose(np.abs(fi1), np.abs(f(tix, tiy)), atol=1e-14)