from time import time
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte, img_as_float, color
from convolution_filtering import gaussianFilter
from edge_detection import gradient_image
from histogramas import adjust_intensity



def supresion_no_maxima(corners, indexes):
    supressed = []
    try:
        for i, j in indexes:
            if i>5 and j > 5:
                candidates = corners[i - 5:i + 6, j - 5:j + 6]
                onemax = np.where(candidates == np.amax(candidates))
                supressed.extend(list(zip(onemax[0] + i - 5, onemax[1] + j -5)))
                ies = np.arange(i - 5, i + 6)
                js = np.arange(j - 5, j + 6)
                for p in product(ies, js):
                    if p in indexes:
                        indexes.remove(p)
    except IndexError as e:
        pass
    return supressed


def corner_harris(inimage, sigmaD, sigmaI, t):
    gx, gy = gradient_image(gaussianFilter(inimage, sigmaD), 'Sobel')
    gxy = gaussianFilter(gx * gy, sigmaI)
    gxx = gaussianFilter(gx * gx, sigmaI)
    gyy = gaussianFilter(gy * gy, sigmaI)

    k = 0.04
    detS = gxx * gyy - gxy ** 2
    traceS = gxx + gyy
    harris = detS - k * traceS ** 2

    not_corners = harris < t
    harriscorners = np.copy(harris)
    harriscorners[not_corners] = 0
    corners_pairs = []
    it = np.nditer(harris, flags=['multi_index'])
    while not it.finished:
        if it[0]!=0:
            corners_pairs.append(it.multi_index)
        it.iternext()
    suprimidos = supresion_no_maxima(harriscorners, corners_pairs)
    outcorners = color.grey2rgb(img_as_ubyte(inimage))
    for i, j in suprimidos:
        outcorners[i][j] = [255, 0, 0]

    return outcorners, harris


def test():
    inimage = io.imread("C:/Users/diego/Desktop/Examen/circles150.png", as_gray=True)

    start_time = time()
    a, harrismap = corner_harris(img_as_float(inimage), 2, 1, 0.85)
    end_time = time() - start_time
    print("Tiempo de ejecucion: ", end_time)
    plt.imshow(a)
    plt.figure()
    harrismap = adjust_intensity(harrismap)
    plt.imshow(harrismap)
    plt.show()


test()
