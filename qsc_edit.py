# --- ONLY NEEDED IF STEM3 IS ACTIVATED --- #

# TODO fix "slices:" and "slices between output:"
# --- find line number of a parameter in the .qsc file --- #
def _line_number_qsc_file(name):
    # --- opens existing file configurations.qsc in the subfolder qstem --- #
    file = open('./qstem/configuration.qsc')
    counter = 0
    number = 0
    for line in file:
        if line.startswith(name):  # maybe name + ':' and it has to be the exact name
            number = counter
        counter += 1
    file.close()
    return number


# --- change output folder (path_folder is subfolder) in .qsc file --- #
def output_folder(path_folder):
    line = _line_number_qsc_file('Folder')
    file = open('./qstem/configuration.qsc')
    split = file.readlines()
    file.close()
    split[line] = 'Folder: \"' + path_folder + '\"\n'
    new = open('./qstem/configuration.qsc', 'w')
    for i in range(len(split)):
        new.write(split[i])
        i += 1
    new.close()


###### NOT USED ######

# --- gets resolution per pixel in Angström out of .qsc file --- #
def get_resolution():
    resx = float(read_qsc_file('resolutionX'))
    resy = float(read_qsc_file('resolutionY'))
    return [resx, resy]


# --- after parameters were changed, .qsc file is written in the updated version --- #
def write_qsc_file():
    file = open('./qstem/configuration.qsc')
    split = file.readlines()
    file.close()
    for parameter in Para.parameterlist:
        split[parameter.linenumber] = parameter.name + ': ' + str(parameter.value)+'\n'
    new = open('./qstem/configuration.qsc', 'w')
    for i in range(len(split)):
        new.write(split[i])
        i += 1
    new.close()


# --- reads .qsc file for a specific parameter and extracts the value (only works for numbers) --- #
def read_qsc_file(name):
    file = open('./qstem/configuration.qsc', 'r')
    text = file.readlines()
    file.close()
    words = text[_line_number_qsc_file(name)].split(' ')
    for number in words:
        try:
            while number.endswith('\n') or number.endswith('\t') or number.endswith('%'):
                number = number[:-1]
            number = float(number)
            value = format(number, '.6f')
            return value
        except ValueError:
            continue


# TODO maybe unnecessary could be helpful for editing .qsc file
class Para:
    # Liste der verwendeten Parameter
    parameterlist = []

    # Initialisiert Parameter mit einem Wert und einem Werteinterval
    def __init__(self, para_name, para_value, minimum, maximum):
        self.name = para_name
        self.value = format(para_value, '.6f')
        self.interval = [minimum, maximum]
        self.parameterlist.append(self)
        self.linenumber = _line_number_qsc_file(para_name)

    # Ändert den Wert des Parameters
    def change_value(self, newvalue):
        self.value = format(newvalue, '.6f')

    def read_value(self):
        newvalue = float(read_qsc_file(self.name))
        return newvalue
