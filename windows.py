from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import *
from frame import do_error_landscape, do_imagesim, do_monte_carlo
import matplotlib.pyplot as plt
import datetime
from img_view import readIMG
import random
import numpy as np
import os
import img_view as img
import util
import graph

# TODO cooling has to adapt to iterations
# TODO angezeigte Bilder werden als Dateien gespeichert, aber nicht gelöscht
# TODO normalized with array[i] = (array[i] - mean)/std correct?!?!
# TODO error calculation normieren auf Bildgröße?!?!?

# - # - # loads the main window to access the different parts of the programme # - # - #
class StartingWindow:
    def __init__(self):
        self.w = loadUi('main_window.ui')
        self.w.move(0, 0)
        self.w.dialog.rejected.connect(self.exit)

    def exit(self):
        plt.close('all')
        QCoreApplication.instance().quit()


# - # - # contains window for Simulated Annealing and all it's functions # - # - #
class SimulatedAnnealing:
    # --- initialise all variables needed and define GUI functions --- #
    def __init__(self):
        self.w = loadUi('simulated_annealing.ui')
        self.w.move(1000, 50)
        self.data = []
        self.threadless = True
        self.error = False
        self.pathWave = None
        self.pathFolder = None
        self.save_exp = False
        self.data_exp = None
        self.thickness = None
        self.data_mc = None
        self.dictionary = {'min_': [], 'max_': [], 'rad_': [], 'corr_': []}
        self.resolution = self.w.slider_resolution.value()
        self.w.button_choose_wave.clicked.connect(self.choose_wave)
        self.w.exit.clicked.connect(self.cancel)
        self.w.slider_resolution.valueChanged.connect(self.resolution_slider)
        self.w.button_choose_folder.clicked.connect(self.get_folder)
        self.w.updateImage.clicked.connect(self.update_image)
        self.w.saveImageExp.clicked.connect(self.save_image_exp)
        self.w.start.clicked.connect(self.start)
        self.w.button_temp_plot.clicked.connect(self.temperature_plot)
        self.w.button_random_aber.clicked.connect(self.random_aberrations)
        self.w.button_multi_sim.clicked.connect(self.multi_sim)
        self.w.button_select_statistics.clicked.connect(self.table_view)
        self.parameter_exp = ['aperture', 'aperture_edge', 'defocus', 'cs', 'astigmatism', 'astigmatism_angle', 'coma',
                              'coma_angle', 'focal_spread', 'conv_angle', 'noise', 'voltage']
        self.checkboxes_exp = ['aperture', 'aperture', 'defocus', 'cs', 'astigmatism', 'astigmatism', 'coma', 'coma',
                               'focal_spread', 'conv_angle', 'noise']
        self.parameter_mc = ['iterations', 'temperature', 'epsilon']
        self.checkboxes_mc = ['defocus', 'cs', 'astigmatism', 'astigmatism_angle', 'coma', 'coma_angle']
        self.ticked_mc = []
        self.prefix = ['min_', 'max_', 'rad_', 'corr_']
        self.translate = None
        edit_line = getattr(self.w, 'value_rad_thickness')
        edit_line.setValidator(QDoubleValidator())
        for name in self.parameter_exp:
            edit_line = getattr(self.w, 'value_' + name)
            edit_line.setValidator(QDoubleValidator())
        for name in self.parameter_mc:
            edit_line = getattr(self.w, 'value_' + name)
            edit_line.setValidator(QDoubleValidator())
        for name in self.checkboxes_mc:
            for prefix in self.prefix:
                edit_line = getattr(self.w, 'value_' + prefix + name)
                edit_line.setValidator(QDoubleValidator())
        self.w.tableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.w.tableWidget.resizeColumnsToContents()

    def cell(self, var=""):
        item = QTableWidgetItem()
        item.setText(var)
        return item

    # --- function for viewing the statistical analysis of a multiple simulation file --- #
    def table_view(self):
        fd = QFileDialog()
        this_dir, this_filename = os.path.split(__file__)
        fname = fd.getOpenFileName(self.w, 'Select Statistic File', this_dir, 'File Format (*.txt)')
        self.w.value_choose_statistics.setText(fname[0])
        f = open(fname[0], 'r')
        x = np.array(f.readline().split('\t')[1:], str)
        data = np.genfromtxt(f, delimiter='\t')
        for i in range(len(x)):
            x[i] = x[i].replace('\n', '')
        self.w.tableWidget.setColumnCount(len(x))
        for i in range(len(x)):
            item = self.cell(x[i])
            self.w.tableWidget.setHorizontalHeaderItem(i, item)
        for i in range(self.w.tableWidget.columnCount()):
            item = self.cell(str(data[0][i]))
            item.setFlags(Qt.ItemIsEnabled)
            item.setTextAlignment(Qt.AlignHCenter)
            self.w.tableWidget.setItem(0, i, item)
        # find data with smallest error
        best_index = data[1:, 0].argmin()
        for i in range(self.w.tableWidget.columnCount()):
            item = self.cell(str(data[best_index + 1][i]))
            item.setFlags(Qt.ItemIsEnabled)
            item.setTextAlignment(Qt.AlignHCenter)
            self.w.tableWidget.setItem(1, i, item)
        # find mean value of all data
        mean = np.mean(data[:][1:], axis=0)
        for i in range(self.w.tableWidget.columnCount()):
            item = self.cell(str(mean[i]))
            item.setFlags(Qt.ItemIsEnabled)
            item.setTextAlignment(Qt.AlignHCenter)
            self.w.tableWidget.setItem(2, i, item)
        # find standard deviation of data
        std = np.std(data[:][1:], axis=0)
        for i in range(self.w.tableWidget.columnCount()):
            item = self.cell(str(std[i]))
            item.setFlags(Qt.ItemIsEnabled)
            item.setTextAlignment(Qt.AlignHCenter)
            self.w.tableWidget.setItem(3, i, item)
        self.w.tableWidget.resizeColumnsToContents()

    # --- helper function the start a Simulated Annealing Run --- #
    def worker(self, data_image=None, image_data=None):
        smallest, xdata, energy_list, temp_list, ydata_list = do_monte_carlo(self, data_image=data_image,
                                                                             image_data=image_data, qstem=False,
                                                                             update_image=False, data_output=True)
        return smallest

    # --- runs multiple Simulated Annealing Runs for statistical purposes --- #
    def multi_sim(self):
        self.data = []
        self.update_exp_data()
        if self.error:
            return
        self.update_mc_data()
        if self.error:
            return
        number = self.w.value_multi_sim.value()
        if number == 0:
            text = self.w.display.text() + '\n Enter valid number of simulations'
            self.w.display.setText(text)
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = self.w.display.text() + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        self.w.progressBar.setMaximum(number)
        self.w.progressBar.setValue(0)
        self.threadless = False
        header = '\terror'
        for name in self.ticked_mc:
            header += '\t' + name
        if self.thickness['bool']:
            header += '\t' + 'thickness'
        path = self.pathWave.replace(os.path.basename(self.pathWave), 'statistics.txt')
        self.w.display.setText('Multi Simulation started')
        value = [0.0]
        for name in self.ticked_mc:
            value.append(self.data_exp[name])
        if self.thickness['bool']:
            value.append(self.thickness['exp_value'])
        self.data.append(value)
        # read wave function and apply eventual noise
        data_image, image_data = img.readIMG(self.pathWave)
        noise = util.phase_noise(data_image, radius_noise=self.data_exp['noise'])
        # run simulation multiple times and save best values found in file
        for i in range(number):
            result = self.worker(data_image=noise, image_data=image_data)
            self.data.append(result)
            data = np.array(self.data)
            np.savetxt(path, data, delimiter='\t', header=header)
            self.w.progressBar.setValue(i + 1)
        self.threadless = True

    # --- collects user's data for the monte carlo simulation --- #
    def update_mc_data(self):
        self.error = False
        value = []
        self.ticked_mc = []
        self.thickness = None
        self.data_mc = None
        self.translate = None
        self.dictionary = {'min_': [], 'max_': [], 'rad_': [], 'corr_': []}
        # check for thickness
        if self.w.check_mc_thickness.isChecked():
            if self.pathFolder is None:
                text = self.w.display.text() + '\n No folder'
                time = datetime.datetime.now().strftime("%H:%M:%S")
                text = text + ' (' + time + ')'
                self.w.display.setText(text)
                self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
                self.error = True
                return
            try:
                self.thickness = {'bool': True, 'min': float(self.w.value_min_thickness.text()),
                                  'max': float(self.w.value_max_thickness.text()),
                                  'rad': float(self.w.value_rad_thickness.text()),
                                  'corr': self.w.value_corr_thickness.value(),
                                  'exp_value': float(self.w.display_thickness.text())}
                if self.thickness['min'] > self.thickness['max']:
                    text = self.w.display.text() + '\nThickness Minimum > Maximum'
                    time = datetime.datetime.now().strftime("%H:%M:%S")
                    text = text + ' (' + time + ')'
                    self.w.display.setText(text)
                    self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
                    self.error = True
                    return
            except:
                text = self.w.display.text() + '\nWrong/no thickness intervall'
                time = datetime.datetime.now().strftime("%H:%M:%S")
                text = text + ' (' + time + ')'
                self.w.display.setText(text)
                self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
                self.error = True
                return
        else:
            self.thickness = {'bool': False}
        # check for parameters to fit
        for name in self.checkboxes_mc:
            checkbox = getattr(self.w, 'check_mc_' + name)
            if checkbox.isChecked():
                self.ticked_mc.append(name)
        if len(self.ticked_mc) == 0:
            text = self.w.display.text() + '\nNo parameter for simulated annealing'
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = text + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            self.error = True
            return
        for prefix in self.prefix:
            if not self.w.check_mc_correction.isChecked() and prefix == 'corr_':
                text = self.w.display.text() + '\nNo correction chosen'
                self.w.display.setText(text)
            else:
                for name in self.ticked_mc:
                    edit_line = getattr(self.w, 'value_' + prefix + name)
                    try:
                        self.dictionary[prefix].append(float(edit_line.text()))
                    except:
                        text = self.w.display.text() + '\nWrong/no parameter : ' + prefix + name
                        self.w.display.setText(text)
                        self.error = True
            self.dictionary[prefix] = dict(zip(self.ticked_mc, self.dictionary[prefix]))
        for name in self.ticked_mc:
            if self.dictionary['min_'][name] >= self.dictionary['max_'][name]:
                text = self.w.display.text() + '\nMinimum >= Maximum : ' + name
                self.w.display.setText(text)
                self.error = True
        for name in self.parameter_mc:
            edit_line = getattr(self.w, 'value_' + name)
            try:
                value.append(float(edit_line.text()))
            except:
                text = self.w.display.text() + '\nWrong/no parameter : ' + name
                self.w.display.setText(text)
                self.error = True
        self.data_mc = dict(zip(self.parameter_mc, value))
        self.data_mc['iterations'] = int(self.data_mc['iterations'])
        edit_line = getattr(self.w, 'value_steps_temp')
        try:
            self.data_mc['steps_temp'] = edit_line.value()
        except:
            text = self.w.display.text() + '\nWrong/no parameter : ' + 'steps_temp'
            self.w.display.setText(text)
            self.error = True
        self.w.progressBar.setMaximum(self.data_mc['iterations'] - self.data_mc['steps_temp'])
        if self.w.check_mc_correction.isChecked():
            edit_line = getattr(self.w, 'value_correction_step')
            try:
                self.data_mc['corr_step'] = edit_line.value()
            except:
                text = self.w.display.text() + '\nWrong/no parameter : ' + 'correction_step'
                self.w.display.setText(text)
                self.error = True
        if self.error:
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = self.w.display.text() + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return

    # --- starts single Simulated Annealing Run with plot at the end and data saved in a file --- #
    def start(self):
        # update all parameters
        self.w.progressBar.setValue(0)
        self.update_exp_data()
        if self.error:
            return
        self.update_mc_data()
        if self.error:
            return
        # run simulation
        smallest, xdata, energy_list, temp_list, ydata_list = do_monte_carlo(self, qstem=False, update_image=False,
                                                                             data_output=True)
        # saving in file
        header = '\titerations\terror\ttemperature'
        for name in self.ticked_mc:
            header += '\t' + name
        if self.thickness['bool']:
            header += '\t' + 'thickness'
        path = self.pathWave.replace(os.path.basename(self.pathWave), 'data_list.txt')
        data = [xdata, energy_list, temp_list]
        for value_list in ydata_list:
            data.append(value_list)
        data = np.array(data)
        data = np.flip(np.rot90(data, 3), 1)
        np.savetxt(path, data, delimiter='\t', header=header)

    # --- save experimental image --- #
    def save_image_exp(self):
        self.save_exp = True
        self.update_exp_data()
        do_monte_carlo(self, qstem=False, update_image=True)
        plt.show(block=True)
        self.save_exp = False

    # --- plot cooling schedule --- #
    def temperature_plot(self):
        try:
            edit_line = getattr(self.w, 'value_iterations')
            iterations = int(edit_line.text())
            edit_line = getattr(self.w, 'value_temperature')
            temperature = float(edit_line.text())
            edit_line = getattr(self.w, 'value_epsilon')
            epsilon = float(edit_line.text())
            edit_line = getattr(self.w, 'value_steps_temp')
            steps = edit_line.value()
        except:
            text = self.w.display.text() + '\nWrong/no parameter for Simulated Annealing'
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = text + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        x = []
        y = []
        temp = temperature
        for i in range(iterations):
            if (i % steps == 0) and (i != 0):
                temp *= epsilon
            x.append(i)
            y.append(temp)
        figure, ax = plt.subplots(1, 1)
        ax.set_xlabel('iterations')
        ax.set_ylabel('temperature')
        ax.set_autoscale_on(True)
        ax.grid()
        ax.set_xlim(0, iterations)
        # ax.set_ylabel(0, temperature)
        ax.plot(x, y)
        plt.show(block=True)

    # --- randomize the aberrations according to interval for simulated annealing --- #
    def random_aberrations(self):
        self.update_mc_data()
        dictionary = dict(zip(self.parameter_exp, self.checkboxes_exp))
        for name in self.ticked_mc:
            checkbox = getattr(self.w, 'check_' + dictionary[name])
            checkbox.setChecked(True)
            if dictionary[name] == 'astigmatism' or dictionary[name] == 'coma':
                if name == 'astigmatism' or name == 'coma':
                    edit_line = getattr(self.w, 'value_' + name + '_angle')
                    if edit_line.text() == '':
                        edit_line.setText(str(0.0))
                else:
                    edit_line = getattr(self.w, 'value_' + name)
                    if edit_line.text() == '':
                        edit_line.setText(str(0.0))
            value = random.uniform(self.dictionary['min_'][name], self.dictionary['max_'][name])
            edit_line = getattr(self.w, 'value_' + name)
            edit_line.setText(str(value))

    # --- check user's data for the experimental image --- #
    def update_exp_data(self):
        self.error = False
        self.data_exp = None
        value = []
        # self.w.display.setText('Please select parameters.')
        if self.pathWave is None:
            text = self.w.display.text() + '\n No wave function'
            self.w.display.setText(text)
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = self.w.display.text() + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            self.error = True
            return
        for i in range(len(self.checkboxes_exp)):
            check_box = getattr(self.w, 'check_' + self.checkboxes_exp[i])
            if check_box.isChecked():
                edit_line = getattr(self.w, 'value_' + self.parameter_exp[i])
                try:
                    value.append(float(edit_line.text()))
                except:
                    text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter_exp[i]
                    self.w.display.setText(text)
                    self.error = True
            else:
                try:
                    value.append(0.0)
                except:
                    text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter_exp[i]
                    self.w.display.setText(text)
                    self.error = True
        i = len(self.checkboxes_exp)
        while i < len(self.parameter_exp):
            name = self.parameter_exp[i]
            edit_line = getattr(self.w, 'value_' + name)
            try:
                value.append(float(edit_line.text()))
            except:
                text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter_exp[i]
                self.w.display.setText(text)
                self.error = True
            i += 1
        dic = dict(zip(self.parameter_exp, value))
        if self.error:
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = self.w.display.text() + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        self.data_exp = dic
        # do_monte_carlo(self, qstem=False, update_image=True)
        time = datetime.datetime.now().strftime("%H:%M:%S")
        text = self.w.display.text() + '\n(' + time + ')'
        self.w.display.setText(text)
        self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
        return

    # --- displays experimental image in window --- #
    def update_image(self):
        self.update_exp_data()
        do_monte_carlo(self, qstem=False, update_image=True)

    # --- file dialog for choosing a wave function for experimental image --- #
    def choose_wave(self):
        fd = QFileDialog()
        this_dir, this_filename = os.path.split(__file__)
        fname = fd.getOpenFileName(self.w, 'Select Wave Function', this_dir, 'File Format (*.img)')
        self.w.value_choose_wave.setText(fname[0])
        self.pathWave = fname[0]
        wave, image_data = readIMG(self.pathWave)
        text = str(image_data.thickness)
        self.w.display_thickness.setText(text)
        self.w.slider_resolution.setMaximum(image_data.pixel_shape[0])
        return

    # --- choose folder if thickness variation is done --- #
    def get_folder(self):
        if self.w.check_mc_thickness.isChecked():
            this_dir, this_filename = os.path.split(__file__)
            self.pathFolder = str(QFileDialog.getExistingDirectory(self.w, "Select Directory", this_dir))
            self.w.value_choose_folder.setText(self.pathFolder)
        else:
            text = self.w.display.text() + '\nActivate Thickness to choose'
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = text + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
        return

    # --- slider for user input for resolution of image --- #
    def resolution_slider(self):
        self.resolution = self.w.slider_resolution.value()
        text = str(self.resolution)
        text = text + ' x ' + text
        self.w.display_resolution.setText(text)

    # --- exiting function --- #
    def cancel(self):
        plt.close('all')
        self.w.display.setText('Please select parameters.')
        self.w.close()


# - # - # window for plotting an error landscape # - # - #
class ErrorLandscape:
    # --- initialising variables and functions of GUI --- #
    def __init__(self):
        self.w = loadUi('error_landscape.ui')
        self.w.move(1000, 50)
        self.parameter = ['aperture', 'aperture_edge', 'defocus', 'cs', 'astigmatism', 'astigmatism_angle', 'coma',
                          'coma_angle', 'focal_spread', 'conv_angle', 'noise', 'para_1_min', 'para_1_max', 'para_2_min',
                          'para_2_max', 'points_1', 'points_2', 'voltage']
        self.checkboxes = ['aperture', 'aperture', 'defocus', 'cs', 'astigmatism', 'astigmatism', 'coma', 'coma',
                           'focal_spread', 'conv_angle', 'noise']
        self.path = None
        self.data = None
        self.parameter_name = []
        for name in self.parameter:
            edit_line = getattr(self.w, 'value_' + name)
            if (name == 'points_1') or (name == 'points_2'):
                edit_line.setValidator(QIntValidator())
            else:
                edit_line.setValidator(QDoubleValidator())
        self.w.button_choose_wave.clicked.connect(self.choose_wave)
        self.w.exit.clicked.connect(self.cancel)
        self.w.start.clicked.connect(self.start_simulation)
        self.w.open_data.clicked.connect(self.plot_data)

    # --- dialog for choosing a wave function --- #
    def choose_wave(self):
        fd = QFileDialog()
        this_dir, this_filename = os.path.split(__file__)
        fname = fd.getOpenFileName(self.w, 'Select Wave Function', this_dir, 'File Format (*.img)')
        self.w.value_choose_wave.setText(fname[0])
        self.path = fname[0]
        return

    # --- starting the simulation of an error landscape --- #
    def start_simulation(self):
        error = False
        self.w.progressBar.setValue(0)
        self.data = None
        self.parameter_name = []
        value = []
        # self.w.display.setText('Please select parameters.')
        if self.path is None:
            text = self.w.display.text() + '\n No wave function'
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = text + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        if self.w.comboBox_1.currentText() == self.w.comboBox_2.currentText():
            text = self.w.display.text() + '\n Select different parameter'
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = text + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        else:
            self.parameter_name.append(self.w.comboBox_1.currentText())
            self.parameter_name.append(self.w.comboBox_2.currentText())
        for i in range(len(self.checkboxes)):
            check_box = getattr(self.w, 'check_' + self.checkboxes[i])
            if check_box.isChecked():
                edit_line = getattr(self.w, 'value_' + self.parameter[i])
                try:
                    value.append(float(edit_line.text()))
                except:
                    text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter[i]
                    self.w.display.setText(text)
                    error = True
            else:
                try:
                    value.append(0.0)
                except:
                    text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter[i]
                    self.w.display.setText(text)
                    error = True
        i = len(self.checkboxes)
        while i < len(self.parameter):
            name = self.parameter[i]
            edit_line = getattr(self.w, 'value_' + name)
            if (name == 'points_1') or (name == 'points_2'):
                try:
                    value.append(int(edit_line.text()))
                except:
                    text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter[i]
                    self.w.display.setText(text)
                    error = True
            else:
                try:
                    value.append(float(edit_line.text()))
                except:
                    text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter[i]
                    self.w.display.setText(text)
                    error = True
            i += 1
        dic = dict(zip(self.parameter, value))
        if dic['para_1_min'] >= dic['para_1_max'] or dic['para_2_min'] >= dic['para_2_max']:
            text = self.w.display.text() + '\n Minimum >= Maximum'
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = text + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        if error:
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = self.w.display.text() + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        self.data = dic
        self.w.progressBar.setMaximum(dic['points_1'] - 1)
        do_error_landscape(self.path, self.data, self.parameter_name, self)
        time = datetime.datetime.now().strftime("%H:%M:%S")
        text = self.w.display.text() + ' (' + time + ')'
        self.w.display.setText(text)
        self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
        return

    # --- read error landscape data file and plot it --- #
    def plot_data(self):
        fd = QFileDialog()
        this_dir, this_filename = os.path.split(__file__)
        fname = fd.getOpenFileName(self.w, 'Select Error Landscape Info File', this_dir, 'File Format (*.txt)')
        path = fname[0]
        f = open(path, 'r')
        header = np.array(f.readline().split('\t')[1:], str)
        for i in range(len(header)):
            header[i] = header[i].replace('\n', '')
        data = np.genfromtxt(f, delimiter='\t')
        graph.surface_plot(data[:, 0], data[:, 1], data[:, 2], title=header[0], xlabel=header[1], ylabel=header[2])
        plt.show(block=True)

    # --- exiting function --- #
    def cancel(self):
        plt.close('all')
        self.w.display.setText('Please select parameters.')
        self.w.close()


# - # - # window for image simulation # - # - #
class ImageSim:
    # --- initialising window and functions of GUI --- #
    def __init__(self):
        self.w = loadUi('image_sim.ui')
        self.w.move(0, 250)
        self.parameter = ['aperture', 'aperture_edge', 'defocus', 'cs', 'astigmatism', 'astigmatism_angle', 'coma',
                          'coma_angle', 'focal_spread', 'conv_angle', 'noise', 'voltage']
        self.checkboxes = ['aperture', 'aperture', 'defocus', 'cs', 'astigmatism', 'astigmatism', 'coma', 'coma',
                           'focal_spread', 'conv_angle', 'noise']
        self.path = None
        self.data = None
        self.parameter_name = []
        self.save = False
        for name in self.parameter:
            edit_line = getattr(self.w, 'value_' + name)
            edit_line.setValidator(QDoubleValidator())
        self.w.button_choose_wave.clicked.connect(self.choose_wave)
        self.w.exit.clicked.connect(self.cancel)
        self.w.start.clicked.connect(self.start_simulation)
        self.w.save.clicked.connect(self.save_screen)

    # --- dialog for choosing the wave function --- #
    def choose_wave(self):
        fd = QFileDialog()
        this_dir, this_filename = os.path.split(__file__)
        fname = fd.getOpenFileName(self.w, 'Select Wave Function', this_dir, 'File Format (*.img)')
        self.w.value_choose_wave.setText(fname[0])
        self.path = fname[0]
        return

    # --- simulate the image from wave function --- #
    def start_simulation(self):
        error = False
        self.data = None
        self.parameter_name = []
        value = []
        # self.w.display.setText('Please select parameters.')
        if self.path is None:
            text = self.w.display.text() + '\n No wave function'
            self.w.display.setText(text)
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = self.w.display.text() + ' (' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        for i in range(len(self.checkboxes)):
            check_box = getattr(self.w, 'check_' + self.checkboxes[i])
            if check_box.isChecked():
                edit_line = getattr(self.w, 'value_' + self.parameter[i])
                try:
                    value.append(float(edit_line.text()))
                except:
                    text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter[i]
                    self.w.display.setText(text)
                    error = True
            else:
                try:
                    value.append(0.0)
                except:
                    text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter[i]
                    self.w.display.setText(text)
                    error = True
        i = len(self.checkboxes)
        while i < len(self.parameter):
            name = self.parameter[i]
            edit_line = getattr(self.w, 'value_' + name)
            try:
                value.append(float(edit_line.text()))
            except:
                text = self.w.display.text() + '\nWrong/no parameter : ' + self.parameter[i]
                self.w.display.setText(text)
                error = True
            i += 1
        dic = dict(zip(self.parameter, value))
        if error:
            time = datetime.datetime.now().strftime("%H:%M:%S")
            text = self.w.display.text() + '(' + time + ')'
            self.w.display.setText(text)
            self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
            return
        self.data = dic
        do_imagesim(self.path, self.data, self, save=self.save)
        time = datetime.datetime.now().strftime("%H:%M:%S")
        text = self.w.display.text() + '\n(' + time + ')'
        self.w.display.setText(text)
        self.w.scrollArea.verticalScrollBar().setValue(self.w.scrollArea.verticalScrollBar().maximum())
        return

    # --- shell for saving the images in pyplot --- #
    def save_screen(self):
        self.save = True
        self.start_simulation()
        self.save = False

    # --- exiting function --- #
    def cancel(self):
        plt.close('all')
        self.w.display.setText('Please select parameters.')
        self.w.close()
