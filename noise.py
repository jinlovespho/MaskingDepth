import numpy as np
import PIL
from PIL import Image
from imagecorruptions import corrupt


def image_corruption(image, corruption_name, severity):
    if corruption_name == "iso_noise":
        image = iso_noise(image, severity)
    elif corruption_name == "color_quant":
        image = color_quant(image, severity)
    elif corruption_name == "dark":
        image = low_light(image, severity)
    else:
        image = corrupt(image, corruption_name=corruption_name, severity=severity)
    return image

def iso_noise(x, severity):
    c_poisson = 25
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.
    c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity-1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255.
    return Image.fromarray(np.uint8(x))

def color_quant(x, severity):
    bits = 5 - severity + 1
    x = PIL.ImageOps.posterize(Image.fromarray(x), bits)
    return x

def low_light(x, severity):
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity-1]
    x = np.array(x) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2.) * 255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=severity-1)
    return x_scaled

def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def poisson_gaussian_noise(x, severity):
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
    c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
    return Image.fromarray(np.uint8(x))