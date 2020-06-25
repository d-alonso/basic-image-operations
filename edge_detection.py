from time import time
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte
from convolution_filtering import gaussianFilter


def angle_deg(dx, dy):  # kernelx/kernely
    return np.mod(np.arctan2(dy, dx), np.pi) * 180 / np.pi


def gradient_image(inimage, operator=''):
    def convolve(kern, offsety, offsetx, mat, m, n):
        acc_list = kern * mat[m - offsety:m + offsety + 1, n - offsetx:n + offsetx + 1]
        acc = np.add.reduce(acc_list, axis=None)
        return acc

    if operator == 'Roberts':
        kernely = np.diag([-1, 1])
        kernelx = np.fliplr(kernely)
    elif operator == 'CentralDiff':
        kernely = np.zeros((3, 1))
        kernely[0, 0] = -1
        kernely[2, 0] = 1
        kernelx = np.rot90(kernely, 1)
    elif operator == 'Prewitt':
        kernelx = np.zeros((3, 3))
        kernelx[0:3, 0] = -1
        kernelx[0:3, 2] = 1
        kernely = np.rot90(kernelx, 3)
    else:
        kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    height, width = inimage.shape
    sizey, sizex = kernelx.shape
    offx = int(sizex / 2)
    offy = int(sizey / 2)
    if operator == 'Roberts':
        convx = 0
        convy = 0
    else:
        convx = offx
        convy = offy

    gx = np.zeros((height, width))
    gy = np.zeros((height, width))

    for i in range(offy, height - offy):
        for j in range(offx, width - offx):
            gx[i][j] = convolve(kernelx, convy, convx, inimage, i, j)
            gy[i][j] = convolve(kernely, convx, convy, inimage, i, j)

    return gx, gy


def supresion_no_maxima(img, angles):
    h, w = img.shape
    supressed = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            try:
                n1 = 255
                n2 = 255
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    n1 = img[i, j + 1]
                    n2 = img[i, j - 1]
                elif 22.5 <= angles[i, j] < 67.5:
                    n1 = img[i + 1, j - 1]
                    n2 = img[i - 1, j + 1]
                elif 67.5 <= angles[i, j] < 112.5:
                    n1 = img[i + 1, j]
                    n2 = img[i - 1, j]
                elif 112.5 <= angles[i, j] < 157.5:
                    n1 = img[i - 1, j - 1]
                    n2 = img[i + 1, j + 1]

                if (img[i, j] >= n1) and (img[i, j] >= n2):
                    supressed[i, j] = img[i, j]
                else:
                    supressed[i, j] = 0
            except IndexError as e:
                pass
    return supressed


def umbralizacion_histeresis(inimage, tlow, thigh):
    height, width = inimage.shape
    res = np.zeros((height, width))

    thigh = inimage.max() * thigh
    tlow = inimage.max() * tlow

    debil = 0.1
    fuerte = 1.0

    strong_i, strong_j = np.where(inimage >= thigh)
    weak_i, weak_j = np.where((inimage <= thigh) & (inimage >= tlow))

    res[strong_i, strong_j] = fuerte
    res[weak_i, weak_j] = debil

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (res[i, j] == debil):
                try:
                    if ((res[i + 1, j - 1] == fuerte) or (res[i + 1, j] == fuerte) or (res[i + 1, j + 1] == fuerte)
                            or (res[i, j - 1] == fuerte) or (res[i, j + 1] == fuerte)
                            or (res[i - 1, j - 1] == fuerte) or (res[i - 1, j] == fuerte) or (
                                    res[i - 1, j + 1] == fuerte)):
                        res[i, j] = fuerte
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    pass
    return res


def edge_canny(inimage, sigma, tlow, thigh):
    smoothed = gaussianFilter(inimage, sigma)
    gx, gy = gradient_image(smoothed, 'Sobel')
    magnitud = np.sqrt(gx ** 2 + gy ** 2)
    orientacion = angle_deg(gx, gy)
    suprimidos = supresion_no_maxima(magnitud, orientacion)
    umbralizado = umbralizacion_histeresis(suprimidos, tlow, thigh)

    return umbralizado


# img = io.imread("C:/Users/diego/Pictures/test/wolf-binimg.gif", as_gray=True)
def test():
    inimage = io.imread("C:/Users/diego/Pictures/test/descar.png", as_gray=True)

    plt.imshow(img_as_ubyte(inimage), cmap="gray", vmin=0, vmax=255)
    plt.figure()

    start_time = time()
    outs, outu = edge_canny(inimage, 1, 0.15, 0.2)
    end_time = time() - start_time
    print("Tiempo de ejecucion: ", end_time)

    plt.imshow(img_as_ubyte(outu), cmap="gray", vmin=0, vmax=255)

    plt.show()
