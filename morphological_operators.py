from time import time
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte


def binimage_from_set(s, height, width):
    image = np.zeros((height, width))
    for i, j in s:
        image[i][j] = True
    return image


def scan_ones(inimage):
    # returns a set of all the ones of a binary image
    ones = set()
    height, width = inimage.shape
    for i in range(height):
        for j in range(width):
            if inimage[i][j]:
                ones.add((i, j))
    return ones


def scan_kernel(kernel, center):
    # transforms a kernel into coordinates according to the new center
    height, width = kernel.shape
    centery, centerx = center
    kernel_set = set()
    for i in range(height):
        for j in range(width):
            if kernel[i][j]:
                kernel_set.add((abs(i - (height - 1)) - centery, j - centerx))

    return kernel_set


def binarize_image(inimage, threshhold):
    # umbralización
    return inimage > threshhold


def dilate(inimage, SE, center=None):
    kernel = SE
    sizey, sizex = kernel.shape
    if center is None:
        center = (int(sizey / 2), int(sizex / 2))
    else:
        center = (center[0], center[1])
    ones_set = scan_ones(inimage)
    kernel_set = scan_kernel(kernel, center)
    height, width = inimage.shape

    centery, centerx = center
    # cálculo del padding necesario
    pady = sizey - 1 - centery
    padx = sizex - 1 - centerx
    # añade solo el padding necesario para los cálculos a la imagen
    padded = np.pad(np.zeros((height, width)), pad_width=((centery, pady), (centerx, padx)),
                    mode='reflect')

    for i, j in ones_set:
        for m, n in kernel_set:  # añadir cada punto del kernel desplazado por el punto de la imagen
            padded[i + m + centery][j + n + centerx] = True

    return padded[centery:height + centery, centerx:width + centerx]  # devuelve sin el padding


def erode(inimage, SE, center=None):
    kernel = SE
    sizey, sizex = kernel.shape
    inimage = binarize_image(inimage, 0.8)
    if center is None:
        center = (int(sizey / 2), int(sizex / 2))
    else:
        center = (center[0], center[1])

    height, width = inimage.shape

    centery, centerx = center
    # cálculo del padding necesario
    pady = sizey - 1 - centery
    padx = sizex - 1 - centerx
    # añade solo el padding necesario para los cálculos a la imagen
    # usar 1 como valor constante permite que se preserven los bordes(limites) de la imagen
    with_ones_padding = np.pad(inimage, pad_width=((centery, pady), (centerx, padx)),
                               mode='reflect')
    ones_set = scan_ones(with_ones_padding)
    kernel_set = scan_kernel(kernel, center)

    outimage = np.pad(np.zeros((height, width)), pad_width=((centery, pady), (centerx, padx)),
                      mode='constant', constant_values=1)
    for i, j in ones_set:
        subset_test = set()
        for m, n in kernel_set:  # añadir cada punto del kernel desplazado por el punto de la imagen
            subset_test.add((i + m, n + j))
        outimage[i][j] = subset_test <= ones_set

    return outimage[centery:height + centery, centerx:width + centerx]  # devuelve sin el padding


def opening(inimage, SE, center=None):
    kernel = SE
    sizey, sizex = kernel.shape
    inimage = binarize_image(inimage, 0.8)
    if center is None:
        center = (int(sizey / 2), int(sizex / 2))
    else:
        center = (center[0], center[1])

    return dilate(erode(inimage, kernel, center), kernel, center)


def closing(inimage, SE, center=None):
    kernel = SE
    sizey, sizex = kernel.shape
    inimage = binarize_image(inimage, 0.8)
    if center is None:
        center = (int(sizey / 2), int(sizex / 2))
    else:
        center = (center[0], center[1])

    return erode(dilate(inimage, kernel, center), kernel, center)


def fill(inimage, seeds, SE=None, center=None):
    if SE is None:
        SE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    kernel = SE
    inimage = binarize_image(inimage, 0.8)
    U = set()
    height, width = inimage.shape
    for i in range(height):
        for j in range(width):
            U.add((i, j))
    ones = scan_ones(inimage)
    not_ones = U - ones
    kernel = SE
    sizey, sizex = kernel.shape
    if center is None:
        center = (int(sizey / 2), int(sizex / 2))
    else:
        center = (center[0], center[1])

    for i, j in seeds:
        outset = np.zeros(inimage.shape)
        old_outset = scan_ones(np.zeros(inimage.shape))
        outset[i][j] = True
        outset = scan_ones(outset)
        while (outset - old_outset):
            old_outset = outset
            outset = scan_ones(dilate(binimage_from_set(old_outset, height, width), SE, center)) & not_ones
        ones = ones | outset
        not_ones = U - ones

    outimg = np.zeros(inimage.shape)
    for i, j in ones:
        outimg[i][j] = True

    return outimg

def test():
    # img = io.imread("C:/Users/diego/Pictures/test/binline.jpg", as_gray=True)
    # img = io.imread("C:/Users/diego/Pictures/test/binimg.gif", as_gray=True)
    img = io.imread("C:/Users/diego/Pictures/test/wolf-binimg.gif", as_gray=True)

    plt.imshow(img_as_ubyte(img), cmap="gray", vmin=0, vmax=255)
    plt.figure()
    ker = np.ones((3, 3))
    start_time = time()
    out = fill(img, [(23, 39),(44, 159)])
    # out = opening(img, ker)
    end_time = time() - start_time
    print("Tiempo de ejecucion: ", end_time)
    plt.imshow(img_as_ubyte(out), cmap="gray", vmin=0, vmax=255)
    plt.show()
