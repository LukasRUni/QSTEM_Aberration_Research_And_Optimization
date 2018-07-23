import numpy as np
import random
import graph
import matplotlib.pyplot as plt
import datetime


# --- checks if a step is accepted or not --- #
def acceptance(energy_new, energy_old, temperature):
    beta = 1/temperature
    p = min(1, np.exp(-beta*(energy_new - energy_old)))
    test = random.random()
    if p > test:
        return True
    else:
        return False


# --- general Monte-Carlo parameter --- #
class MCData:
    def __init__(self, iterations, temperature, epsilon, number_parameter):
        self.iterations = iterations
        self.n = number_parameter
        self.temperature = temperature
        self.epsilon = epsilon
        self.smallest_value_so_far = [np.float('inf')] * (self.n + 1)
        self.energy_old = 0.0
        self.energy_new = np.float('inf')


# --- class of a single parameter being optimized --- #
class MCParameter:
    # --- gives new value in an interval of +/- radius around the current value --- #
    # --- checks if it stays inside a given interval --- #
    def random_variation(self):
        new_value = self.value_old + random.uniform(-self.radius, self.radius)
        if (new_value < self.search_min) or (new_value > self.search_max):
            new_value = self.random_variation()
        return new_value

    def random_thickness(self):
        test = random.randint(-self.radius, self.radius)
        thickness_new = self.value_old + test
        if (thickness_new < self.search_min) or (thickness_new > self.search_max):
            thickness_new = self.random_variation()
        return int(thickness_new)

    # --- returns a percentage of the radius as the search radius --- #
    def radius_calc(self):
        return np.abs(self.search_max - self.search_min) * self.percent_of_interval

    def radius_thickness(self):
        return int(np.ceil(np.abs(self.search_max - self.search_min) * self.percent_of_interval))

    def __init__(self, name, search_min, search_max, percent_of_interval=1.0, thickness_list=None):
        # --- name has to be format 'a20' corresponding to the correct name of aberration in code --- #
        self.name = name
        self.percent_of_interval = percent_of_interval
        self.thickness_list = thickness_list
        if self.thickness_list is not None:
            self.search_min = 0
            self.search_max = len(thickness_list) - 1
            self.radius = self.radius_thickness()
            self.value_old = random.randint(self.search_min, self.search_max)
            self.value_new = self.random_thickness()
        else:
            self.search_min = search_min
            self.search_max = search_max
            self.radius = self.radius_calc()
            self.value_old = random.uniform(self.search_min, self.search_max)
            self.value_new = self.random_variation()

# --- contains the Monte-Carlo-Method --- #
class MCMethod:
    def __init__(self, mcdata_object, list_mcparameter_object, derivate_object, window):
        self.window = window
        self.threadless = self.window.threadless
        # --- list of classes of all parameter that will be explored --- #
        self.list_mcparameter = list_mcparameter_object
        # --- general data of the MC Method --- #
        self.mcdata = mcdata_object
        # --- derivate contains ctf class and calculate_error class for calculating the error --- #
        self.derivate_object = derivate_object
        # --- calculate the energy for the starting configuration --- #
        self.derivate_object.calculate_error()
        self.mcdata.energy_old = self.derivate_object.error

    # --- contains a whole MC step --- #
    def mc_step(self):
        accepted = False
        # --- update coordinates of the smallest value found so far --- #
        if self.mcdata.energy_old < self.mcdata.smallest_value_so_far[0]:
            self.mcdata.smallest_value_so_far[0] = self.mcdata.energy_old
            if self.threadless:
                self.window.w.display.setText('Simulated Annealing started\nSmallest error: ' +
                                              str(self.mcdata.smallest_value_so_far[0]))
            for i in range(self.mcdata.n):
                self.mcdata.smallest_value_so_far[i+1] = self.list_mcparameter[i].value_old
        # --- do a random step for every parameter and write it in the ctf class inside the calculate_error class--- #
        for i in range(self.mcdata.n):
            if self.list_mcparameter[i].thickness_list is None:
                self.list_mcparameter[i].value_new = self.list_mcparameter[i].random_variation()
                self.derivate_object.ctf.aberrations[self.list_mcparameter[i].name] = self.list_mcparameter[i].value_new
            else:
                self.list_mcparameter[i].value_new = self.list_mcparameter[i].random_thickness()
                self.derivate_object.ctf.wave_probe = \
                    self.list_mcparameter[i].thickness_list[self.list_mcparameter[i].value_new][0]
        # --- calculate energy for new parameter point --- #
        self.derivate_object.calculate_error()
        self.mcdata.energy_new = self.derivate_object.error
        # --- MC criterion if the suggested step is accepted, if yes, update parameter --- #
        if acceptance(self.mcdata.energy_new, self.mcdata.energy_old, self.mcdata.temperature):
            accepted = True
            self.mcdata.energy_old = self.mcdata.energy_new
            for i in range(self.mcdata.n):
                self.list_mcparameter[i].value_old = self.list_mcparameter[i].value_new
        return accepted

    # --- whole routine for simulated annealing --- #
    def do_mc(self, plot_graph=True, data_output=False, radius_plot=False):
        a = []
        b = []
        for name in self.window.translate:
            a.append(name)
            b.append(self.window.translate[name])
        translate = dict(zip(b, a))
        counter = 0
        radius_list = []
        radius_list.append([])
        for i in range(self.mcdata.n):
            radius_list.append([])
        radius_list[0].append(0)
        for j in range(self.mcdata.n):
            radius_list[j + 1].append(self.list_mcparameter[j].radius)
            xdata = []
            energy_list = []
            temp_list = []
            plot_list = []
            ydata_list = []
            for i in range(self.mcdata.n):
                ydata_list.append([])
        for i in range(self.mcdata.iterations):
            # --- do one mc step --- #
            check = self.mc_step()
            # --- reset on smallest value and decreasing search radius --- #
            if check:
                counter = 0
            if self.window.w.check_mc_correction.isChecked():
                if (counter > self.window.data_mc['corr_step']) and (self.mcdata.smallest_value_so_far[0] <=
                                                                     self.mcdata.energy_old):
                    self.mcdata.energy_old = self.mcdata.smallest_value_so_far[0]
                    radius_list[0].append(i)
                    for j in range(self.mcdata.n):
                        self.list_mcparameter[j].value_old = self.mcdata.smallest_value_so_far[j+1]
                        if self.list_mcparameter[j].thickness_list is None:
                            self.list_mcparameter[j].radius *= \
                                self.window.dictionary['corr_'][translate[self.list_mcparameter[j].name]]
                        else:
                            if self.list_mcparameter[j].radius > 1:  # TODO minimal thickness
                                self.list_mcparameter[j].radius = int(self.list_mcparameter[j].radius -
                                                                      self.window.thickness['corr'])
                                if self.list_mcparameter[j].radius == 1:
                                    for element in self.list_mcparameter:
                                        # TODO correction of cs in thickness search explain in thesis
                                        if element.name == 'a40':
                                            element.radius = element.radius_calc()
                        radius_list[j+1].append(self.list_mcparameter[j].radius)
                    counter = 0
            # --- cooling down (adjust cooling schedule) --- #
            if (i % self.window.data_mc['steps_temp'] == 0) and (i != 0):
                self.mcdata.temperature *= self.mcdata.epsilon
                if self.threadless:
                    self.window.w.progressBar.setValue(i)
            # if self.mcdata.smallest_value_so_far[0] < 0.2:
            #     break
            counter += 1

            if plot_graph:
                xdata.append(i)
                energy_list.append(self.mcdata.energy_old)
                temp_list.append(self.mcdata.temperature)
                for j in range(self.mcdata.n):
                    if self.list_mcparameter[j].thickness_list is None:
                        ydata_list[j].append(self.list_mcparameter[j].value_old)
                    else:
                        ydata_list[j].append(self.list_mcparameter[j].thickness_list[self.list_mcparameter[j].value_old][2].thickness)
        if self.threadless:
            self.print_smallest()
        if plot_graph:
            plot_error = graph.DynamicUpdate(0.0, self.mcdata.iterations)
            plot_temp = graph.DynamicUpdate(0.0, self.mcdata.iterations)
            plot_error.on_launch(title1='error vs. iterations')
            plot_temp.on_launch(title1='temperature vs. iterations')
            for i in range(self.mcdata.n):
                plot_list.append(graph.DynamicUpdate(0.0, self.mcdata.iterations))
                title = self.list_mcparameter[i].name + ' vs. iterations'
                plot_list[i].on_launch(title1=title)
            plot_error.on_end(xdata, energy_list)
            plot_temp.on_end(xdata, temp_list)
            for i in range(self.mcdata.n):
                plot_list[i].on_end(xdata, ydata_list[i])
            plt.show(block=True)
        if radius_plot:
            for i in range(self.mcdata.n):
                radius_figure = plt.figure()
                ax = radius_figure.gca()
                ax.grid()
                ax.set_title(self.list_mcparameter[i].name + ' radius')
                ax.scatter(radius_list[0], radius_list[i + 1])
            plt.show(block=True)
        if data_output:
            ret = (self.mcdata.smallest_value_so_far, xdata, energy_list, temp_list, ydata_list)
            return ret

    def print_smallest(self):
        text = self.window.w.display.text()
        for i in range(self.mcdata.n):
            if self.list_mcparameter[i].thickness_list is None:
                text += '\n' + self.list_mcparameter[i].name + ' : ' + str(self.mcdata.smallest_value_so_far[i + 1])
            else:
                thick = self.list_mcparameter[i].thickness_list[self.mcdata.smallest_value_so_far[i + 1]][2].thickness
                text += '\n' + self.list_mcparameter[i].name + ' : ' + str(thick)
        time = datetime.datetime.now().strftime("%H:%M:%S")
        text = text + '\n(' + time + ')'
        self.window.w.display.setText(text)
        self.window.w.scrollArea.verticalScrollBar().setValue(self.window.w.scrollArea.verticalScrollBar().maximum())
