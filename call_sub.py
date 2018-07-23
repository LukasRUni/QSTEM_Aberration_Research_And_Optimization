import subprocess
import os
import qsc_edit as qsc
import itertools
import img_view as img


# --- context manager for changing the current working directory --- #
# source: EMAIL (26.03.2018)
class cd:
    def __init__(self, newpath):
        self.newpath = os.path.expanduser(newpath)

    def __enter__(self):
        self.savedpath = os.getcwd()
        os.chdir(self.newpath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedpath)


# --- calls QSTEM for simulation based on .qsc file in the correct subfolder --- #
# --- path_qsc is relative path from the python programm to the .qsc file (same folder as .cfg file) --- #
# --- folders current_python -> qstem (with .qsc/.cfg) -> path_output_folder --- #
def stem3(path_qsc, path_output_folder):
    this_dir, this_filename = os.path.split(__file__)
    qsc.output_folder(path_output_folder)
    with cd(this_dir+'/qstem/'):
        subprocess.check_call('stem3 '+path_qsc)


# Source: https://stackoverflow.com/questions/33691187/how-to-save-the-file-with-different-name-
#                   and-not-overwriting-existing-one
# --- append number to filename if the name already exists --- #
def unique_file(basename, ext, path=''):
    actual_name = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(path + actual_name):
        actual_name = "%s%d.%s" % (basename, next(c), ext)
    return actual_name

# --- read relative path and thickness of different wave images and return thickness in interval --- #
# path = './qstem/data/test'
def thickness_list(path, thickness_min, thickness_max):
    # this_dir, this_filename = os.path.split(__file__)
    basename = 'wave'
    ext = 'img'
    c = itertools.count()
    actual_name = "/%s_%d.%s" % (basename, next(c), ext)
    list_thickness = []
    counter = 0
    while os.path.exists(path + actual_name):
        data, image_data = img.readIMG(path + actual_name)
        output = [data, path + actual_name, image_data]
        list_thickness.append(output)
        counter += 1
        actual_name = "/%s_%d.%s" % (basename, next(c), ext)
    list_thickness = [x for x in list_thickness if not (x[2].thickness < thickness_min or x[2].thickness > thickness_max)]
    return list_thickness