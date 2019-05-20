# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Basem\source\repos\KeratokonusGit\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.selectImgBtn = QtWidgets.QPushButton(self.centralwidget)
        self.selectImgBtn.setGeometry(QtCore.QRect(50, 120, 101, 41))
        self.selectImgBtn.setObjectName("selectImgBtn")
        self.OriginalImage = QtWidgets.QLabel(self.centralwidget)
        self.OriginalImage.setGeometry(QtCore.QRect(200, 10, 581, 251))
        self.OriginalImage.setFrameShape(QtWidgets.QFrame.Box)
        self.OriginalImage.setObjectName("OriginalImage")
        self.NewImage = QtWidgets.QLabel(self.centralwidget)
        self.NewImage.setGeometry(QtCore.QRect(200, 280, 581, 251))
        self.NewImage.setFrameShape(QtWidgets.QFrame.Box)
        self.NewImage.setObjectName("NewImage")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.selectImgBtn.setText(_translate("MainWindow", "Выбрать снимок"))
        self.OriginalImage.setText(_translate("MainWindow", "TextLabel"))
        self.NewImage.setText(_translate("MainWindow", "TextLabel"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


