import numpy as np
from scipy import spatial
from util import normalize, calc_intensity
from itertools import combinations


# --- class Minimization: all functions to import a ctf object and evaluate the error function and calculate_error --- #
# --- example for triggering calculation of a calculate_error --- #
# optimize_aberration = ['a20', 'a42', 'a40'] (Beispiel)
class Minimization:
    def __init__(self, ctf_object, picture_exp, optimize_aberration=None):
        self.optimize_aberration = optimize_aberration
        # --- importing ctf object --- #
        self.ctf = ctf_object
        self.picture_sim = np.array(calc_intensity(self.ctf.wave.real, self.ctf.wave.imag))
        self.picture_exp = picture_exp
        self.total_derivation = None
        self.error = None
        # --- factors caused by a calculate_error in reciprocal space --- #
        self.theta_factor = {
            'a22': (0.5 * np.cos(2. * (self.ctf.phi - self.ctf.aberrations['phi22'])) *
                    self.ctf.theta ** 2),
            'phi22': (self.ctf.aberrations['a22'] * np.sin(2. * (self.ctf.phi - self.ctf.aberrations['phi22'])) *
                      self.ctf.theta ** 2),
            'a20': (0.5 * self.ctf.theta ** 2),
            'a33': (1 / 3. * np.cos(3. * (self.ctf.phi - self.ctf.aberrations['phi33'])) * self.ctf.theta ** 3),
            'phi33': (self.ctf.aberrations['a33'] * np.sin(3. * (self.ctf.phi - self.ctf.aberrations['phi33'])) *
                      self.ctf.theta ** 3),
            'a31': (1 / 3. * np.cos(self.ctf.phi - self.ctf.aberrations['phi31']) * self.ctf.theta ** 3),
            'phi31': (1 / 3. * self.ctf.aberrations['a31'] * np.sin(self.ctf.phi - self.ctf.aberrations['phi31']) *
                      self.ctf.theta ** 3),
            'a44': (1 / 4. * np.cos(4. * (self.ctf.phi - self.ctf.aberrations['phi44'])) * self.ctf.theta ** 4),
            'phi44': (self.ctf.aberrations['a44'] * np.sin(4. * (self.ctf.phi - self.ctf.aberrations['phi44'])) *
                      self.ctf.theta ** 4),
            'a42': (1 / 4. * np.cos(2. * (self.ctf.phi - self.ctf.aberrations['phi42'])) * self.ctf.theta ** 4),
            'phi42': (1 / 2. * self.ctf.aberrations['a42'] * np.sin(2. * (self.ctf.phi - self.ctf.aberrations['phi42']))
                      * self.ctf.theta ** 4),
            'a40': (1 / 4. * self.ctf.theta ** 4)
        }

    # --- update ctf, simulate picture, save it in picture_sim --- #
    def _update_ctf(self):
        self.ctf.simulate_image()
        self.picture_sim = normalize(np.array(calc_intensity(self.ctf.wave.real, self.ctf.wave.imag)))

    # --- calculating the error function --- #
    # --- handing over 2D arrays auf simulated and experimental intensity --- #
    def error_function(self, array2d_sim, array2d_exp):
        if len(array2d_sim.shape) != 2 or len(array2d_exp.shape) != 2:
            raise RuntimeError('Wrong array format in calculation of error function!')
        # --- convert 2D arrays into vector --- #
        vec_sim = np.reshape(array2d_sim, array2d_sim.shape[0] * array2d_sim.shape[1])
        vec_exp = np.reshape(array2d_exp, array2d_exp.shape[0] * array2d_exp.shape[1])
        vec_diff = vec_sim - vec_exp
        # --- error = ||vec_diff|| --- #
        error = spatial.distance.euclidean(vec_sim, vec_exp)
        return error, vec_diff

    # --- complete routine to calculate the calculate_error --- #
    def calculate_error(self):
        self._update_ctf()
        error, vec_diff = self.error_function(self.picture_sim, self.picture_exp)
        self.error = error

    def derivation(self):
        self._update_ctf()
        error, vec_diff = self.error_function(self.picture_sim, self.picture_exp)
        self.error = error
        # --- bring back in if calculate_error is needed --- #
        list_derivation = []
        if self.optimize_aberration is None:
            raise RuntimeError('Derivation not possible. Please define aberrations.')
        for aberration in self.optimize_aberration:
            list_derivation.append(self._derivation_aberration(aberration))
        jacobi = self._build_jacobi(list_derivation)
        self.total_derivation = self._derivation_error_function(jacobi, vec_diff, error)

    # --- NOT USED --- #

    # TODO geteilt durch error? Achtung wenn sehr klein!!
    # --- jacobian(matrix) * vector_difference(vector) / error(scalar) --- #
    def _derivation_error_function(self, jacobi, vec_diff, error):
        return jacobi.dot(vec_diff)  #  / error

    # --- convert 2D calculate_error in 1D, connect multiple and reshape them to 2D jacobian matrix -> row vectors --- #
    def _build_jacobi(self, list_derivation):
        # --- check if shape is compatible --- #
        shape = []
        for deriv in list_derivation:
            if len(deriv.shape) != 2:
                raise RuntimeError('_build_jacobi derivations sind keine 2D arrays.')
            shape.append(deriv.shape)
        for a, b in combinations(shape, 2):
            if a != b:
                raise RuntimeError('derivations haben nicht das gleiche Format bei Erstellung der Jacobi-Matrix.')
        shape = shape[0]
        jacobi = []
        for deriv in list_derivation:
            deriv = deriv.reshape(shape[0] * shape[1])
            jacobi.append(deriv.tolist())
        jacobi = np.array(jacobi)
        return jacobi

    # TODO Untersuchung der Ableitung (Einheiten, usw.)
    # --- calculate_error gives -2*pi*i/lambda*theta_factor[aberration] in reciprocal space --- #
    # --- returns 2D array of the calculate_error for a given aberration --- #
    # --- f = z(x).conj()*z(x) , df/dx = (z(x).conj())' * z(x) + z(x).conj() * (z(x))' --- #
    def _derivation_aberration(self, aberration):
        # --- check if arrays are 2D --- #
        if len(self.ctf.wave.shape) != 2:
            raise RuntimeError('_derivation_aberration: wave has wrong format.')
        if len(self.ctf.wave_probe.shape) != 2:
            raise RuntimeError('_derivation_aberration: wave_probe has wrong format.')
        if len(self.ctf.ctf_grid.shape) != 2:
            raise RuntimeError('_derivation_aberration: ctf_grid has wrong format.')
        if len(self.theta_factor[aberration].shape) != 2:
            raise RuntimeError('_derivation_aberration: factor_derivation has wrong format.')
        # --- complex conjugate of already propagated wave --- #
        wave_cc = self.ctf.wave.conj()
        # --- complex conjugate of exiting wave --- #
        wave_probe_cc = self.ctf.wave_probe.conj()
        # --- calculate_error of wave --- #
        z_derivation = np.fft.ifft2(self.ctf.ctf_grid * self.theta_factor[aberration] *
                                    (-2.) * np.pi * 1j / self.ctf.wavelength * np.fft.fft2(self.ctf.wave_probe))
        # --- calculate_error of complex conjugated wave --- #
        z_cc_derivation = np.fft.fft2(self.ctf.ctf_grid.conj() * self.theta_factor[aberration] *
                                      2. * np.pi * 1j / self.ctf.wavelength * np.fft.ifft2(wave_probe_cc))
        # --- calculate_error of intensity --- #
        result = z_cc_derivation * self.ctf.wave + wave_cc * z_derivation
        return result

