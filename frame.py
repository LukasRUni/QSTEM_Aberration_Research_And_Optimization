import image_sim
import util
import img_view as img
import numpy as np
import graph
import call_sub as call
import monte_carlo as mc
import derivation as deriv
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from monte_carlo import MCMethod


# - # - # MONTE-CARLO-SIMULATION # - # - #
def do_monte_carlo(window, data_image=None, image_data=None, qstem=False, update_image=False, data_output=False):
    # --- activate QSTEM for simulation of wave function exiting the specimen --- #
    if qstem:
        call.stem3('configuration.qsc', window.pathWave)
    # --- input of the complex wave function --- #
    if data_image is None:
        data_image, image_data = img.readIMG(window.pathWave)
        data_image = data_image[0:window.resolution, 0:window.resolution]
        noise = util.phase_noise(data_image, radius_noise=window.data_exp['noise'])
        image_data.pixel_shape = (window.resolution, window.resolution)
    else:
        noise = data_image[0:window.resolution, 0:window.resolution]
        image_data.pixel_shape = (window.resolution, window.resolution)
    # --- settings for image simulation --- #
    aber = {'a20': window.data_exp['defocus'], 'a40': window.data_exp['cs'], 'a22': window.data_exp['astigmatism'],
            'phi22': window.data_exp['astigmatism_angle'], 'a31': window.data_exp['coma'],
            'phi31': window.data_exp['coma_angle']}
    aperture = window.data_exp['aperture']
    aperture_edge = window.data_exp['aperture_edge']
    focal_spread = window.data_exp['focal_spread']
    convergence_angle = window.data_exp['conv_angle']
    # --- initialising the ctf class with the aberration values --- #
    ctf_object = image_sim.Ctf(window.data_exp['voltage'], noise, image_data, aperture=aperture,
                               aperture_edge=aperture_edge, focal_spread=focal_spread,
                               convergence_angle=convergence_angle, aberrations=aber)
    # --- simulation of propagation --- #
    ctf_object.simulate_image()
    # --- intensity calculation and saving of image --- #
    intensity = np.array(util.calc_intensity(ctf_object.wave.real, ctf_object.wave.imag))
    intensity = util.normalize(intensity)
    show_int = np.rot90(intensity)
    img.save_to_image(show_int, window.pathWave.replace('.img', '_picture'), file_format='.png')
    window.w.exp_image.setPixmap(QPixmap(window.pathWave.replace('.img', '_picture.png')))
    if window.save_exp:
        graph.single_plot(show_int, image_data, title='intensity', xlabel='x in A', ylabel='y in A')
    if update_image:
        return
    thickness = window.thickness['bool']
    if thickness:
        list_thickness = call.thickness_list(window.pathFolder, window.thickness['min'], window.thickness['max'])
        for image in list_thickness:
            image[0] = image[0][0:window.resolution, 0:window.resolution]
            image[2].pixel_shape = (window.resolution, window.resolution)
    # - # temperatured anneheling # - #
    # --- parameter (list) initialising --- #
    window.translate = ['a20', 'a40', 'a22', 'phi22', 'a31', 'phi31']
    window.translate = dict(zip(window.checkboxes_mc, window.translate))
    mc_parameter_list = []
    for name in window.ticked_mc:
        test = mc.MCParameter(window.translate[name], window.dictionary['min_'][name], window.dictionary['max_'][name],
                              window.dictionary['rad_'][name])
        mc_parameter_list.append(test)
    if thickness:
        thick = mc.MCParameter('thickness', 0, len(list_thickness) - 1, percent_of_interval=window.thickness['rad'],
                               thickness_list=list_thickness)
        mc_parameter_list.append(thick)
    # --- mcdata initialising --- #
    mcdata = mc.MCData(window.data_mc['iterations'], window.data_mc['temperature'], window.data_mc['epsilon'],
                       len(mc_parameter_list))
    aber = {}
    for x in mc_parameter_list:
        aber[x.name] = x.value_old
    # --- CTF initialising --- #
    if thickness:
        CTF = image_sim.Ctf(window.data_exp['voltage'], list_thickness[0][0], list_thickness[0][2],
                            aperture=aperture, aperture_edge=aperture_edge, focal_spread=focal_spread,
                            convergence_angle=convergence_angle, aberrations=aber)
    else:
        CTF = image_sim.Ctf(window.data_exp['voltage'], data_image, image_data, aperture=aperture,
                            aperture_edge=aperture_edge, focal_spread=focal_spread, convergence_angle=convergence_angle,
                            aberrations=aber)
    # --- derivation_object initialising --- #
    derivate = deriv.Minimization(CTF, intensity)
    # --- MC Method initialising --- #
    mcmethod = MCMethod(mcdata, mc_parameter_list, derivate, window)
    # --- complete MC simulation according to parameter --- #
    smallest, xdata, energy_list, temp_list, ydata_list = mcmethod.do_mc(plot_graph=window.threadless, data_output=True,
                                                                         radius_plot=False)
    if data_output:
        smallest[-1] = list_thickness[smallest[-1]][2].thickness
        ret = (smallest, xdata, energy_list, temp_list, ydata_list)
        return ret


# - # - # IMAGE-SIMULATION # - # - #
def do_imagesim(path, data, window, qstem=False, save=False):
    # --- activate QSTEM for simulation of wave function exiting the specimen --- #
    if qstem:
        call.stem3('configuration.qsc', path)
    # --- input of the complex wave function --- #
    data_image, image_data = img.readIMG(path)
    noise = util.phase_noise(data_image, radius_noise=data['noise'])
    # --- settings for image simulation --- #
    aber = {'a20': data['defocus'], 'a40': data['cs'], 'a22': data['astigmatism'], 'phi22': data['astigmatism_angle'],
            'a31': data['coma'], 'phi31': data['coma_angle']}
    aperture = data['aperture']
    aperture_edge = data['aperture_edge']
    focal_spread = data['focal_spread']
    convergence_angle = data['conv_angle']
    # --- initialising the ctf class with the aberration values --- #
    ctf_object = image_sim.Ctf(data['voltage'], noise, image_data, aperture=aperture, aperture_edge=aperture_edge,
                               focal_spread=focal_spread, convergence_angle=convergence_angle, aberrations=aber)
    # --- simulation of propagation --- #
    ctf_object.simulate_image()
    # --- intensity calculation and saving of image --- #
    intensity = np.array(util.calc_intensity(ctf_object.wave.real, ctf_object.wave.imag))
    intensity = util.normalize(intensity)
    show_int = np.rot90(intensity)
    img.save_to_image(show_int, path.replace('.img', '_picture'), file_format='.png')
    img.save_to_image(np.rot90(np.absolute(ctf_object.wave_probe)), path.replace('.img', '_amplitude'),
                      file_format='.png')
    img.save_to_image(np.rot90(np.angle(ctf_object.wave_probe)), path.replace('.img', '_phase'), file_format='.png')
    img.save_to_image(np.rot90(ctf_object.ctf_grid.real), path.replace('.img', '_ctfreal'), file_format='.png')
    img.save_to_image(np.rot90(ctf_object.ctf_grid.real), path.replace('.img', '_ctfimag'), file_format='.png')
    window.w.intensity.setPixmap(QPixmap(path.replace('.img', '_picture.png')))
    window.w.amplitude.setPixmap(QPixmap(path.replace('.img', '_amplitude.png')))
    window.w.phase.setPixmap(QPixmap(path.replace('.img', '_phase.png')))
    window.w.real.setPixmap(QPixmap(path.replace('.img', '_ctfreal.png')))
    window.w.imag.setPixmap(QPixmap(path.replace('.img', '_ctfimag.png')))
    # --- plot of the measured picture --- #
    if save:
        graph.single_plot(intensity, image_data, title='intensity', xlabel='x in A', ylabel='y in A')
        # --- creating the axis based on simulation parameter --- #
        real_room = [0.0, ctf_object.resolution[0] * ctf_object.shape[0], 0.0,
                     ctf_object.resolution[1] * ctf_object.shape[1]]
        rec_room = [ctf_object.kx.min(), ctf_object.kx.max(), ctf_object.ky.min(), ctf_object.ky.max()]
        # --- multi plot of wave (real/imag) and ctf function (real, imag) --- #
        ctf_object.wave_probe = np.rot90(ctf_object.wave_probe)
        ctf_object.ctf_grid = np.rot90(ctf_object.ctf_grid)
        graph.wave_ctf_plot(ctf_object.wave_probe, ctf_object.ctf_grid, real_room=real_room, rec_room=rec_room)


# - # - # ERROR-LANDSCAPE-SIMULATION # - # - #
def do_error_landscape(path, data, parameter_name, window):
    # --- activate QSTEM for simulation of wave function exiting the specimen --- #
    # call.stem3('configuration.qsc', path)
    # --- input of the complex wave function --- #
    data_image, image_data = img.readIMG(path)
    noise = util.phase_noise(data_image, radius_noise=data['noise'])
    comboboxes = {'Defocus': 'a20', 'Cs': 'a40', 'Astigmatism': 'a22', 'Astigm. Angle': 'phi22', 'Coma': 'a31',
                  'Coma Angle': 'phi31'}
    parameter1 = mc.MCParameter(comboboxes[parameter_name[0]], data['para_1_min'], data['para_1_max'])
    parameter2 = mc.MCParameter(comboboxes[parameter_name[1]], data['para_2_min'], data['para_2_max'])
    # --- simulation of a picture as the EXPERIMENTAL one --- #
    aber = {'a20': data['defocus'], 'a40': data['cs'], 'a22': data['astigmatism'], 'phi22': data['astigmatism_angle'],
            'a31': data['coma'], 'phi31': data['coma_angle']}
    aperture = data['aperture']
    aperture_edge = data['aperture_edge']
    focal_spread = data['focal_spread']
    convergence_angle = data['conv_angle']
    ctf_object = image_sim.Ctf(data['voltage'], noise, image_data, aperture=aperture, aperture_edge=aperture_edge,
                               focal_spread=focal_spread, convergence_angle=convergence_angle, aberrations=aber)
    ctf_object.simulate_image()
    intensity = np.array(util.calc_intensity(ctf_object.wave.real, ctf_object.wave.imag))
    intensity = util.normalize(intensity)
    # --- saving of experimental picture as .png --- #
    show_int = np.rot90(intensity)
    img.save_to_image(show_int, path.replace('.img', '_picture'), file_format='.png')
    window.w.exp_image.setPixmap(QPixmap(path.replace('.img', '_picture.png')))
    parameter1_list = np.linspace(parameter1.search_min, parameter1.search_max, data['points_1'])
    parameter2_list = np.linspace(parameter2.search_min, parameter2.search_max, data['points_2'])
    x, y = np.meshgrid(parameter1_list, parameter2_list)
    error_grid = np.zeros_like(x)
    show_graph_def_cs = True

    # --- initialising CTF object for the derivate class --- #
    ctf_object = image_sim.Ctf(data['voltage'], data_image, image_data, aperture=aperture, aperture_edge=aperture_edge,
                               focal_spread=focal_spread, convergence_angle=convergence_angle, aberrations=aber)
    # --- initialising the Minimization class --- #
    derivate = deriv.Minimization(ctf_object, intensity)
    # --- loops for varying parameters --- #
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # --- updating aberrations --- #
            derivate.ctf.aberrations[parameter1.name] = x[i][j]
            derivate.ctf.aberrations[parameter2.name] = y[i][j]
            # --- updating ctf/simulated image, executing the calculation of error function --- #
            derivate.calculate_error()
            error_grid[i][j] = derivate.error
        window.w.progressBar.setValue(i)
    # --- plots 3d grid defocus vs cs --- #
    if show_graph_def_cs:
        if window.w.check_save.isChecked():
            graph.surface_plot(x, y, error_grid, title='error', xlabel=parameter1.name,
                               ylabel=parameter2.name, save=True, path=path)
        else:
            graph.surface_plot(x, y, error_grid, title='error', xlabel=parameter1.name,
                               ylabel=parameter2.name)
        plt.show(block=True)
