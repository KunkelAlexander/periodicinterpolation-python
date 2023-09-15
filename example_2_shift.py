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

# METHOD 2: phase rotation in frequency space
# shift in x-, y- and xy-direction
fix  = np.fft.ifftn(fk * np.exp(1j*(kx*dx/2          )), norm="forward")
fiy  = np.fft.ifftn(fk * np.exp(1j*(        + ky*dx/2)), norm="forward")
fixy = np.fft.ifftn(fk * np.exp(1j*(kx*dx/2 + ky*dx/2)), norm="forward")
# zip original array and shifted arrays
fi3      = np.zeros((Ni, Ni), dtype=complex) 
fi3[ ::2,  ::2] = f(tx, ty) 
fi3[ ::2, 1::2] = fix
fi3[1::2,  ::2] = fiy
fi3[1::2, 1::2] = fixy

# check result
isclose(np.abs(fi3), np.abs(f(tix, tiy)), atol=1e-15)