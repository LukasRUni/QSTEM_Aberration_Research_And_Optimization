# --- source: https://github.com/QSTEM/QSTEM/blob/master/stem3/debug_data/img_view.py (25.03.2018) --- #
# was changed

import numpy as np
from PIL import Image


class image_data_object:
    def __init__(self, nx, ny, res_x, res_y, thickness):
        self.pixel_shape = (nx, ny)
        self.resolution = (res_x, res_y)
        self.thickness = thickness


# --- copied --- #
def _get_dtype(hdr):
    element_size = hdr['element_size']
    if element_size == 4:  # 32-bit float
        dtype = np.float32
    elif element_size == 8:
        if hdr['is_complex'] == 1:  # 32-bit complex
            dtype = np.complex64
        else:  # double
            dtype = np.float64
    elif element_size == 16:
        dtype = np.complex128

    return dtype


# --- routine to read .tif pictures into array --- #
def read_tiff(filename):
    im = Image.open(filename + '.tif')
    imarray = np.array(im)
    return imarray


# --- reading wave function (.img file) into array , copied--- #
def readIMG(filename, debug=False):
    # total header size: 56 bytes (aus Quelle)
    hdr_dtype = np.dtype([
        ('hdr_size', '<i4'),
        ('param_size', '<i4'),
        ('comment_size', '<i4'),
        ('nx', '<i4'),
        ('ny', '<i4'),
        ('is_complex', '<i4'),
        ('element_size', '<i4'),
        ('qstem_version', '<i4'),
        ('thickness', '<f8'),
        ('x_px_size', '<f8'),
        ('y_px_size', '<f8')])
    hdr = np.fromfile(filename, dtype=hdr_dtype, count=1)
    if debug:
        print("header, param, comment sizes: ")
        print(hdr['hdr_size'], hdr['param_size'], hdr['comment_size'])
        print("Image size: ")
        print(hdr['nx'], hdr['ny'])
        if hdr['is_complex'] == 1:
            print("Data is complex.")
        print("Data element size: ")
        print(hdr['element_size'])
        print("Thickness at this image: ")
        print(hdr['thickness'])
        print("Pixel size: ")
        print(hdr['x_px_size'], hdr['y_px_size'])
    data_dtype = _get_dtype(hdr)
    f = open(filename, "rb")
    # skip the header bytes
    f.read(56)
    aux_data = f.read(8*hdr['param_size'][0])
    comments = str(f.read(hdr['comment_size'][0]))
    data = f.read(hdr['nx'][0]*hdr['ny'][0]*hdr['element_size'][0])
    data = np.frombuffer(data, count=hdr['nx'][0]*hdr['ny'][0], dtype=data_dtype)
    data.setflags(write=1)
    data = data.reshape((hdr['nx'][0], hdr['ny'][0]))

    image_data = image_data_object(int(hdr['nx'][0]), int(hdr['ny'][0]), float(hdr['x_px_size'][0]),
                                   float(hdr['y_px_size'][0]), float(hdr['thickness'][0]))
    return data, image_data


# --- saving 2D arrays in a .tif file --- #
def save_to_image(data, filename, file_format=".tif"):
    if np.iscomplexobj(data):
        realimg = Image.fromarray(data.real)
        imagimg = Image.fromarray(data.imag)
        realimg.save(filename + "_amp" + file_format)
        imagimg.save(filename + "_phase" + file_format)
    else:
        if file_format == '.png':
            data = (((data - data.min()) / (data.max() - data.min())) * 255.9).astype(np.uint8)
        img = Image.fromarray(data)
        img.save(filename + file_format)
