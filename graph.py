import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os
from call_sub import unique_file


# --- creates single plot used for intensity --- #
def single_plot(array, image_data, title='', xlabel='', ylabel='', save=False, path=None, name=None):
    resolution = image_data.resolution
    shape = image_data.pixel_shape
    figure, ax = plt.subplots(1, 1)
    ax.imshow(array, cmap='gray', extent=[0.0, resolution[0]*shape[0], 0.0, resolution[1]*shape[1]])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    if save:
        if path is not None:
            if name is not None:
                figure.savefig(path + name)


# --- plots amplitude/phase/ctf_real/ctf_imag --- #
def wave_ctf_plot(wave=None, ctf=None, real_room=None, rec_room=None,
                  save=False, path=None, name=None):
    amplitude = np.absolute(wave)
    phase = np.angle(wave)
    data = [[amplitude, phase], [ctf.real, ctf.imag]]
    titles = [['wave amplitude', 'wave phase'], ['ctf real', 'ctf imag']]
    interval = [real_room, rec_room]
    axis_label = [['x in A', 'y in A'], ['kx in 1/A', 'ky in 1/A']]
    box = dict(facecolor='red', pad=3, alpha=0.2)
    figure, ax = plt.subplots(2, 2, figsize=(16, 8))
    figure.subplots_adjust(left=0.2, wspace=0.6)
    for i in range(2):
        for j in range(2):
            ax[i, j].imshow(data[i][j], cmap='gray',
                            extent=[interval[i][0], interval[i][1], interval[i][2], interval[i][3]])
            ax[i, j].set_title(titles[i][j], bbox=box)
            ax[i, j].set_xlabel(axis_label[i][0])
            ax[i, j].set_ylabel(axis_label[i][1])
    figure.tight_layout()
    plt.show(block=True)
    if save:
        if path is not None:
            if name is not None:
                figure.savefig(path + name)


# --- error landscape surface plot --- #
def surface_plot(x, y, z, title='', xlabel='', ylabel='', save=False, path='', xlim=None, ylim=None):
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_zlabel(title, fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    # --- trisurf plot --- #
    if len(x.shape) == 2:
        x = x.reshape(x.shape[0] * x.shape[1])
        y = y.reshape(y.shape[0] * y.shape[1])
        z = z.reshape(z.shape[0] * z.shape[1])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    fig1.tight_layout()
    ax.plot_trisurf(x, y, z, shade=True)
    # ax.scatter(x, y, z, c=z, s=3, cmap=cm.inferno, antialiased=False)


    if save:
        path = path.split('/')
        path = '/'.join(path[:-1]) + '/'
        if (x.shape != y.shape) or (x.shape != z.shape):
            raise RuntimeError('surface plot not possible as array shapes don\'t match')
        info = '\t' + title + '\t' + xlabel + '\t' + ylabel
        name_info = unique_file('error_plot', 'txt', path=path)
        data = np.flip(np.rot90(np.array([x, y, z]), 3), -1)
        np.savetxt(path + name_info, data, delimiter='\t', header=info)

plt.ion()


# --- class to create a dynamic graph --- #
class DynamicUpdate:
    # --- x range for the graph --- #
    def __init__(self, min_x1, max_x1, multi_plot=False, min_x2=None, max_x2=None):
        self.min_x = min_x1
        self.max_x = max_x1
        self.multi_plot = multi_plot
        if self.multi_plot:
            if (min_x2 is None) or (max_x2 is None):
                raise RuntimeError('Please enter interval for plotting.')
            else:
                self.min_x2 = min_x2
                self.max_x2 = max_x2

    # --- initialising the graph (add parameter if needed) --- #
    def on_launch(self, title1='', x_title1='', y_title1='', title2='', x_title2='', y_title_2=''):
        # --- activate for multiple plots in one window --- #
        if self.multi_plot:
            self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1)
            self.lines2, = self.ax2.plot([], [], '.')
            self.ax2.set_autoscaley_on(True)
            self.ax2.set_xlim(self.min_x2, self.max_x2)
            self.ax2.grid()
            self.ax2.set_title(title2)
            self.ax2.set(xlabel=x_title2, ylabel=y_title_2)
        # --- deactivate for multiple plots in one window --- #
        else:
            self.figure, self.ax1 = plt.subplots()
        self.lines1, = self.ax1.plot([], [], 'r')
        self.ax1.set_autoscaley_on(True)
        self.ax1.set_xlim(self.min_x, self.max_x)
        self.ax1.grid()
        self.ax1.set_title(title1)
        self.ax1.set(xlabel=x_title1, ylabel=y_title1)

    # --- update data (with the new _and_ the old points) --- #
    def on_running(self, xdata1, ydata1, xdata2=None, ydata2=None):  # edit for multiple plots
        self.lines1.set_xdata(xdata1)
        self.lines1.set_ydata(ydata1)
        self.ax1.relim()
        self.ax1.autoscale_view()
        # --- activate for multiple plots --- #
        if self.multi_plot:
            self.lines2.set_xdata(xdata2)
            self.lines2.set_ydata(ydata2)
            self.ax2.relim()
            self.ax2.autoscale_view()
        # --- We need to draw *and* flush --- #
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # TODO Speichern der Verläufe funktioniert nicht, speichert nur calculate_error
    def on_end(self, xdata1, ydata1, xdata2=None, ydata2=None, save=False, path=None, name=None):
        self.lines1.set_xdata(xdata1)
        self.lines1.set_ydata(ydata1)
        self.ax1.relim()
        self.ax1.autoscale_view()
        # --- acitvate for multiple plots --- #
        if self.multi_plot:
            self.lines2.set_xdata(xdata2)
            self.lines2.set_ydata(ydata2)
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        if save:
            if path is not None:
                if name is not None:
                    self.figure.savefig(path + name)
