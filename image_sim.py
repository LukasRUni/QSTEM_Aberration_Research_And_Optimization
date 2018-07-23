import numpy as np
import util


class Ctf:
    def __init__(self, voltage, wave, image_data, defocus=0., cs=0., aperture=np.float('inf'), aperture_edge=0.,
                 convergence_angle=0., focal_spread=0., aberrations={}):
        self.wave = wave
        self.wave_probe = wave
        self.resolution = image_data.resolution
        self.shape = image_data.pixel_shape
        self.ctf_grid = np.zeros(self.shape, dtype=complex)
        self.defocus = defocus
        self.cs = cs
        self.aperture = aperture * 10 ** (-3)
        self.aperture_edge = aperture_edge * 10 ** (-3)
        self.convergence_angle = convergence_angle * 10 ** (-3)
        self.focal_spread = focal_spread
        symbols = ['a22', 'phi22', 'a20', 'a33', 'phi33', 'a31', 'phi31', 'a44', 'phi44', 'a42', 'phi42', 'a40',
                   'a55', 'phi55', 'a53', 'phi53', 'a51', 'phi51', 'a66', 'phi66', 'a64', 'phi64', 'a62', 'phi62',
                   'a60']
        self.aberrations = dict(zip(symbols, [0.] * len(symbols)))
        self.aberrations['a20'] = self.defocus
        self.aberrations['a40'] = self.cs
        self.aberrations.update(aberrations)
        self.voltage = voltage
        self.wavelength = util.get_lambda(voltage)
        # --- constructs grid in the reciprocal space --- #
        self.kx, self.ky, self.k_2, self.theta, self.phi = util.construct_reciprocal(self.wave.shape, self.resolution,
                                                                                     self.wavelength)

    def _get_wave_parameter(self, image, energy, wavelength):
        if image is None:
            if self.shape is None:
                raise RuntimeError('Shape not set')
            else:
                shape = self.shape
        if image is None:
            if self.resolution is None:
                raise RuntimeError('Resolution not set')
            else:
                resolution = self.resolution
        if (wavelength is None) & (energy is None):
            if self.wavelength is None:
                raise RuntimeError('Wavelength not set')
            else:
                wavelength = self.wavelength
        return shape, resolution, wavelength

    # --- calculation of chi --- #
    def _get_chi(self, theta, phi, wavelength, aberrations):
        a = aberrations
        chi = (1 / 2. * (a["a22"] * np.cos(2. * (phi - a["phi22"])) + a["a20"]) * theta ** 2 +
               1 / 3. * (a["a33"] * np.cos(3. * (phi - a["phi33"])) + a["a31"] *
                         np.cos(1. * (phi - a["phi31"]))) * theta ** 3 +
               1 / 4. * (a["a44"] * np.cos(4. * (phi - a["phi44"])) + a["a42"] * np.cos(2. * (phi - a["phi42"])) +
                         a["a40"]) * theta ** 4 +
               1 / 5. * (a["a55"] * np.cos(5. * (phi - a["phi55"])) + a["a53"] * np.cos(3. * (phi - a["phi53"])) +
                         a["a51"] * np.cos(1. * (phi - a["phi51"]))) * (theta ** 5) +
               1 / 6. * (a["a66"] * np.cos(6. * (phi - a["phi66"])) + a["a64"] * np.cos(4. * (phi - a["phi64"])) +
                         a["a62"] * np.cos(2. * (phi - a["phi62"])) + a["a60"]) * theta ** 6)
        chi *= 2 * np.pi / wavelength
        return chi

    # --- calculating the aperture envelope --- #
    def _get_aperture_envelope(self, theta):
        if np.isfinite(self.aperture):
            aperture = np.ones_like(theta)
            aperture[theta > self.aperture + self.aperture_edge] = 0.
            ind = (theta > self.aperture) & (theta < self.aperture_edge + self.aperture)
            aperture[ind] *= 0.5 * (1 + np.cos(np.pi * (theta[ind] - self.aperture) / self.aperture_edge))
        else:
            aperture = None
        return aperture

    # --- calculating the temporal envelope --- #
    def _get_temporal_envelope(self, theta, wavelength):
        if self.focal_spread > 0.:
            temporal = np.exp(
                -np.sign(self.focal_spread) * (0.5 * np.pi / wavelength * self.focal_spread * theta ** 2) ** 2)
        else:
            temporal = None
        return temporal

    # --- calculating spatial envelope --- #
    def _get_spatial_envelope(self, theta, phi, wavelength):
        a = self.aberrations
        if self.convergence_angle > 0.:
            dchi_dq = 2 * np.pi / wavelength * ((a["a22"] * np.cos(2. * (phi - a["phi22"])) + a["a20"]) * theta +
                                                (a["a33"] * np.cos(3. * (phi - a["phi33"])) +
                                                 a["a31"] * np.cos(1. * (phi - a["phi31"]))) * theta ** 2 +
                                                (a["a44"] * np.cos(4. * (phi - a["phi44"])) +
                                                 a["a42"] * np.cos(2. * (phi - a["phi42"])) + a["a40"]) * theta ** 3 +
                                                (a["a55"] * np.cos(5. * (phi - a["phi55"])) +
                                                 a["a53"] * np.cos(3. * (phi - a["phi53"])) +
                                                 a["a51"] * np.cos(1. * (phi - a["phi51"]))) * theta ** 4 +
                                                (a["a66"] * np.cos(6. * (phi - a["phi66"])) +
                                                 a["a64"] * np.cos(4. * (phi - a["phi64"])) +
                                                 a["a62"] * np.cos(2. * (phi - a["phi62"])) + a["a60"]) * theta ** 5)
            dchi_dphi = -2 * np.pi / wavelength * (
                    1 / 2. * (2. * a["a22"] * np.sin(2. * (phi - a["phi22"]))) * theta +
                    1 / 3. * (3. * a["a33"] * np.sin(3. * (phi - a["phi33"])) +
                              1. * a["a31"] * np.sin(1. * (phi - a["phi31"]))) * theta ** 2 +
                    1 / 4. * (4. * a["a44"] * np.sin(4. * (phi - a["phi44"])) +
                              2. * a["a42"] * np.sin(2. * (phi - a["phi42"]))) * theta ** 3 +
                    1 / 5. * (5. * a["a55"] * np.sin(5. * (phi - a["phi55"])) +
                              3. * a["a53"] * np.sin(3. * (phi - a["phi53"])) +
                              1. * a["a51"] * np.sin(1. * (phi - a["phi51"]))) * theta ** 4 +
                    1 / 6. * (6. * a["a66"] * np.sin(6. * (phi - a["phi66"])) +
                              4. * a["a64"] * np.sin(4. * (phi - a["phi64"])) +
                              2. * a["a62"] * np.sin(2. * (phi - a["phi62"]))) * theta ** 5)
            spatial = np.exp(
                -np.sign(self.convergence_angle) * (self.convergence_angle / 2) ** 2 * (dchi_dq ** 2 + dchi_dphi ** 2))
        else:
            spatial = None
        return spatial

    # --- calculating the whole ctf grid in reciprocal space --- #
    def _calculate(self, image=None, wavelength=None, energy=None):
        shape, resolution, wavelength = self._get_wave_parameter(image, energy, wavelength)
        # --- modulation through chi --- #
        self.ctf_grid = np.exp(-1.j * self._get_chi(self.theta, self.phi, wavelength, self.aberrations))
        # --- modulation through aperture --- #
        aperture = self._get_aperture_envelope(self.theta)
        if aperture is not None:
            self.ctf_grid *= aperture
        # --- modulation through damping --- #
        temporal = self._get_temporal_envelope(self.theta, wavelength)
        if temporal is not None:
            self.ctf_grid *= temporal
        # --- modulation through spatial envelope --- #
        spatial = self._get_spatial_envelope(self.theta, self.phi, wavelength)
        if spatial is not None:
            self.ctf_grid *= spatial

    # TODO check if aberrations changed otherwise dont recalculate
    # --- _calculate ctf, simulate the propagated wave --- #
    def simulate_image(self):
        self._calculate()
        grid = np.fft.fft2(self.wave_probe)
        grid = np.fft.fftshift(grid)
        grid = grid * self.ctf_grid
        self.wave = np.fft.ifft2(grid)
