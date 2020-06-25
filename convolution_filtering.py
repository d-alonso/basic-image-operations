from time import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sqrt, exp
from skimage import io, img_as_ubyte


def filter_image(inimage, kernel):
    def convolve(ker, offsety, offsetx, mat, m, n):
        acc_list = ker * mat[m - offsety:m + offsety + 1, n - offsetx:n + offsetx + 1]
        acc = np.add.reduce(acc_list, axis=None)
        return acc

    height, width = inimage.shape
    sizey, sizex = kernel.shape
    offx = int(sizex / 2)
    offy = int(sizey / 2)
    padded = np.pad(inimage, pad_width=((offy, offy), (offx, offx)), mode='symmetric')
    outimage = np.zeros(inimage.shape)

    for i in range(offy, height + offy):
        for j in range(offx, width + offx):
            outimage[i - offy][j - offx] = convolve(kernel, offy, offx, padded, i, j)
    return outimage


def gaussKernel1D(sigma):
    sigma = float(sigma)

    def gauss(x):
        return 1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2))

    N = 2 * int(3 * sigma) + 1
    halfkernel = np.zeros(int(N / 2) + 1)
    for i, j in zip(range(halfkernel.size - 1, -1, -1), range(halfkernel.size)):
        halfkernel[i] = gauss(j)
    kernel = np.pad(halfkernel, pad_width=((0, halfkernel.size - 1)), mode='reflect')
    return np.array([kernel])


def gaussianFilter(inimage, sigma):
    kern = gaussKernel1D(sigma)
    firstconv = filter_image(inimage, kern)
    outimage = filter_image(firstconv, kern.T)
    return outimage


def medianFilter(inimage, filtersize):
    filtersize = int(filtersize)
    height, width = inimage.shape
    offset = int(filtersize / 2)
    padded = np.pad(inimage, pad_width=((offset, offset), (offset, offset)), mode='symmetric')
    outimage = np.zeros(inimage.shape)

    for i in range(offset, height + offset):
        for j in range(offset, width + offset):
            outimage[i - offset][j - offset] = np.median(padded[i - offset:i + offset, j - offset:j + offset].flatten())
    return outimage


def highBoost(inimage, A, method, param):
    A = float(A)
    param = float(param)
    if method == 'gaussian':
        smoothing = gaussianFilter
    else:
        smoothing = medianFilter
    outimage = abs(A * inimage - smoothing(inimage, param))
    outimage = outimage / outimage.max()
    return outimage


def test():
    img = io.imread("C:/Users/diego/Pictures/test/descar.png", as_gray=True)
    # gaussker = gaussKernel1D(2)
    kerneltest = np.ones((9, 9)) / 81
    # kerneltest2 = np.ones((7, 7))

    start_time = time()
    output = filter_image(img, kerneltest)
    end_time = time() - start_time
    print("Tiempo de ejecucion: ", end_time)
    output = gaussianFilter(img, 4)

    # test_lena = median(img, kerneltest2)
    #
    plt.subplot(1, 2, 1)
    plt.imshow(img_as_ubyte(img), cmap="gray", vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(img_as_ubyte(output), cmap="gray", vmin=0, vmax=255)
    # plt.subplot(1, 3, 3)
    # plt.imshow(img_as_ubyte(test_lena), cmap="gray", vmin=0, vmax=255)

    plt.show()
#test()