import sys
import windows
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import *


class First(QMainWindow):
    def __init__(self, parent=None):
        super(First, self).__init__(parent)
        self.starting = windows.StartingWindow()
        self.landscape = windows.ErrorLandscape()
        self.image = windows.ImageSim()
        self.annealing = windows.SimulatedAnnealing()
        self.starting.w.dialog.accepted.connect(self.confirm)

    #--- for confirmation in starting window ---#
    def confirm(self):
        if self.starting.w.errorLandscape.isChecked():
            self.landscape.w.show()
        if self.starting.w.imageSim.isChecked():
            self.image.w.show()
        if self.starting.w.simulatedAnnealing.isChecked():
            self.annealing.w.show()


def main():
    app = QApplication(sys.argv)
    main = First()
    main.starting.w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
