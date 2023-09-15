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

# METHOD 3: zero-padding
# shift zero frequencies to center of cube
fkPad    = np.fft.fftshift(fk)
# determine size of padding of negative frequencies
NPadN    = int(np.floor(Ni/2-N/2))
# if either the input or output size is uneven
# add one additional positive frequency
NPadP    = NPadN+(Ni+N)%2
fkPad    = np.pad(fkPad, ((NPadP, NPadN), (NPadP, NPadN)))
# shift zero frequencies back to outside of cube
fkPad    = np.fft.ifftshift(fkPad)
# go back to position space
fi3      = np.fft.ifftn(fkPad, norm="forward")
# check result
isclose(np.abs(fi3), np.abs(f(tix, tiy)), atol=1e-14)