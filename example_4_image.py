# import libraries
import numpy as np
import argparse 
from PIL import Image

# parse input parameters
argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument("-i", "--input",  type=str, help="Filename of input image.", required=True)
argParser.add_argument("-o", "--output", type=str, help="Filename of output image.", required=True)
argParser.add_argument("-r", "--rescale", type=float, help="Scale factor for image.", default=1)

args = argParser.parse_args()

 #credit to stackoverflow user Bily for this unpadding function
def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]

# load image using pillow
img        = Image.open(args.input)
# convert to numpy array
f          = np.asarray(img)
Nx, Ny, Nz = f.shape
Nix, Niy   = int(Nx * args.rescale), int(Ny * args.rescale) 

# compute the DFT of the image
fk         = np.fft.fftn(f, norm="forward", axes=(0, 1))
# shift zero frequencies to center of cube
fk         = np.fft.fftshift(fk, axes=(0, 1))
 

# upscale
if args.rescale > 1:
    # pad square from the outside
    # determine size of padding of negative frequencies
    NPadXN    = int(np.floor(Nix/2-Nx/2))
    NPadYN    = int(np.floor(Niy/2-Ny/2))
    # if either the input or output size is uneven
    # add one additional positive frequency
    NPadXP    = NPadXN+(Nix+Nx)%2
    NPadYP    = NPadYN+(Niy+Ny)%2
    fk        = np.pad(fk, ((NPadXP, NPadXN), (NPadYP, NPadYN), (0, 0)))
#downscale
else:
    # unpad square from the outside
    # determine size of padding of negative frequencies
    NPadXN    = int(np.floor(Nx/2-Nix/2))
    NPadYN    = int(np.floor(Ny/2-Niy/2))
    NPadXP    = NPadXN
    NPadYP    = NPadYN
    fk        = unpad(fk, ((NPadXP, NPadXN), (NPadYP, NPadYN), (0, 0)))

# shift zero frequencies back to outside
fk    = np.fft.ifftshift(fk, axes=(0, 1))
# go back to position space
fi    = np.fft.ifftn(fk, norm="forward", axes=(0, 1))

out   = Image.fromarray(np.real(fi).clip(0, 255).astype(np.uint8))
out.save(args.output)