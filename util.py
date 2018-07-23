import numpy as np
from numba import jit
import random


# --- normalizing 1D or 2D real float arrays -> careful with complex --- #
# --- normalizing with (I - mean(I))/std(I) --- #
# @timing
@jit
def normalize(array):
    if array.dtype == int:
        raise RuntimeError('Integer devision in normalization.')
    if len(array.shape) == 2:
        std = np.std(array)
        mean = np.mean(array)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i][j] = (array[i][j]-mean)/std
    else:
        if len(array.shape) == 1:
            mean = np.mean(array)
            std = np.std(array)
            # counter = 0
            for i in range(len(array)):  # number in array:
                # array[counter] = (array[counter] - mean)/std
                # counter += 1
                array[i] = (array[i] - mean)/std
        else:
            raise RuntimeError('Normalization failed due to wrong array shape.')
    return array


# --- returns relativistic wavelength in Angström for input voltage [keV] --- #
# source: GITHUB QSTEM GUI_MATLAB WAVELENGTH.M (11.04.2018)
def get_lambda(voltage):
    emass = 510.99906
    hc = 12.3984244
    return hc/np.sqrt(voltage*(2*emass+voltage))


# --- construct reciprocal space grid according to a number of pixels and the resolution --- #
# (return_polar according to source)
def construct_reciprocal(shape, resolution, wavelength, return_polar=True):
    freqx = np.fft.fftshift(np.fft.fftfreq(shape[0], resolution[0]))
    freqy = np.fft.fftshift(np.fft.fftfreq(shape[1], resolution[1]))
    ky, kx = np.meshgrid(freqx, freqy)
    k_2 = kx * kx + ky * ky
    ret = (kx, ky, k_2)
    if return_polar:
        theta = np.sqrt(k_2*wavelength*wavelength)
        phi = np.arctan2(ky, kx)
        ret += (theta, phi)
    return ret


# --- calculates the intensity of a wave function out of its' imaginary and real part --- #
@jit(parallel=True)
def calc_intensity(real, imag):
    array = np.zeros_like(real)
    for i in range(len(real)):
        for j in range(len(imag)):
            array[i][j] = real[i][j]*real[i][j] + imag[i][j]*imag[i][j]
    return array


# --- apply a random noise on phase of wave function --- #
def phase_noise(array, radius_noise=0.):
    if not np.iscomplexobj(array):
        raise RuntimeError('Array is not complex, so phase modulation is not possible.')
    amp_old = np.absolute(array)
    phase_old = np.angle(array)
    noise = np.zeros(array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            noise[i][j] = random.uniform(-radius_noise, radius_noise)
            # noise[i][j] = random.gauss(0.0, radius_noise)
    phase_new = phase_old + noise
    real_new = amp_old * np.cos(phase_new)
    imag_new = amp_old * np.sin(phase_new)
    wave_noise = np.zeros_like(array)
    wave_noise.real = real_new
    wave_noise.imag = imag_new
    return wave_noise


###### NOT USED ######

# --- other normalization for better imaging --- #
def norm(array):
    if len(array.shape) == 2:
        new_max = 1.0
        new_min = 0.0
        maximum = array.max()
        minimum = array.min()
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i][j] = (array[i][j] - minimum)/(maximum - minimum)*(new_max-new_min) + new_min
    else:
        if len(array.shape) == 1:
            mean = np.mean(array)
            std = np.std(array)
            counter = 0
            for number in array:
                array[counter] = (array[counter] - mean)/std
                counter += 1
        else:
            raise RuntimeError('Normalization failed due to wrong array shape.')
    return array


def scherzer(voltage, cs):
    assert cs > 0
    return -1.2*np.sqrt(get_lambda(voltage) * cs)


def scherzer_point_resolution(voltage, cs):
    assert cs > 0
    return 0.6 * get_lambda(voltage) ** (3/4.) * cs ** (1 / 4.)
