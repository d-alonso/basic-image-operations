import numpy as np
from skimage import img_as_float, img_as_ubyte, data
import matplotlib.pyplot as plt


def equalize_intensity(inimage, nbins=256):
    nbins = int(nbins)
    inimage = img_as_ubyte(inimage)
    histograma, bins = np.histogram(inimage.ravel(), nbins, density=True)
    acumulado = histograma.cumsum()
    transfer = np.uint8(255 * acumulado) / inimage.size

    return
    height, width = inimage.shape
    outimage = np.zeros(inimage.shape)
    for i in range(height):
        for j in range(width):
            value = int(inimage[i][j] * (nbins - 1) / 255)
            outimage[i][j] = transfer[value]

    return img_as_float(outimage / 255)  # exposure.equalize_hist(inimage, nbins)  #


def ref_equalize_intensity(inimage, nbins=256):
    histograma, bins = np.histogram(inimage.ravel(), int(nbins), density=True)
    acumulado = histograma.cumsum()
    acumulado = acumulado / acumulado[-1]  # normalizar

    imagen_ecualizada = np.interp(inimage.ravel(), bins[:-1], acumulado)
    return imagen_ecualizada.reshape(inimage.shape)


def adjust_intensity(inimage, inrange=None, outrange=None):
    if outrange is None:
        outrange = [0, 1]
    if inrange is None:
        inrange = []

    imin = inimage.min()
    imax = inimage.max()

    omin = 0
    omax = 1

    if len(inrange) == 2:
        imin = float(inrange[0])
        imax = float(inrange[1])

    if len(outrange) == 2:
        omin = float(outrange[0])
        omax = float(outrange[1])

    def gnorm(x):
        return omin + ((omax - omin) * (x - imin) / (imax - imin))

    outimage = gnorm(inimage)

    return outimage


def test():
    img = data.camera()

    outimg = adjust_intensity(img, [], [0.5, 1])

    counts, bins = np.histogram(outimg, range(257))

    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)

    plt.subplot(1, 4, 2)
    bin_counts, bin_edges, patches = plt.hist(img.ravel(), range(257))

    plt.subplot(1, 4, 3)
    plt.imshow(outimg, cmap="gray", vmin=0, vmax=255)

    plt.subplot(1, 4, 4)
    bin_counts, bin_edges, patches = plt.hist(outimg.ravel(), range(257))
    plt.show()
