# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Basem\source\repos\KeratokonusGit\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from MoreResult import Ui_ResultWindow
from Learning import GetThresholdImage,GetContours, PrepareContoursForArc, GetArc, GetLeastSquares, PrepareXY
from testtf import GetROI
from TestingChart import SetupPlot
from Classification import ClassifyStage, CountOccasions
from TFtoOpenCV import anotherOne
import numpy as np
import os.path

counter = np.zeros(shape=(5,5),dtype=int)
stages_filenames = {'Normal':[],'I':[],'II':[],'III':[],'IV':[]}
stages_counter = [0,0,0,0,0]
timesRunned = 0

class Ui_MainWindow(object):

    def SummaryMessage(self,stages_filenames):
        message = "На основе " + (str)(sum(map(len,stages_filenames.values()))) + " снимков был получен следующий результат:"
        for key,value in stages_filenames.items():
            if len(value) > 0:
                if key == "Normal":
                    message += "\n\n\nСледующие снимки были распознаны без отклонений от нормы:\n"
                    for item in value:
                        message += item + "\n"
                if key == "I":
                    message += "\n\n\nСледующие снимки были распознаны как I-ая стадия кератоконуса:\n"
                    for item in value:
                        message += item + "\n"
                if key == "II":
                    message += "\n\n\nСледующие снимки были распознаны как II-ая стадия кератоконуса:\n"
                    for item in value:
                        message += item + "\n"
                if key == "III":
                    message += "\n\n\nСледующие снимки были распознаны как III-ая стадия кератоконуса:\n"
                    for item in value:
                        message += item + "\n"
                if key == "IV":
                    message += "\n\n\nСледующие снимки были распознаны как IV-ая стадия кератоконуса:\n"
                    for item in value:
                        message += item + "\n"
                        
        return message

    def ShowMoreResult(self):
        self.window = QtWidgets.QMainWindow()
        self.message = self.SummaryMessage(stages_filenames)
        self.ui = Ui_ResultWindow(self.message)
        self.ui.setupUi(self.window)
        self.window.show()


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.selectImgBtn = QtWidgets.QPushButton(self.centralwidget)
        self.selectImgBtn.setGeometry(QtCore.QRect(30, 20, 121, 41))
        self.selectImgBtn.setObjectName("selectImgBtn")
        self.OriginalImage = QtWidgets.QLabel(self.centralwidget)
        self.OriginalImage.setGeometry(QtCore.QRect(200, 10, 581, 251))
        self.OriginalImage.setFrameShape(QtWidgets.QFrame.Box)
        self.OriginalImage.setObjectName("OriginalImage")
        self.NewImage = QtWidgets.QLabel(self.centralwidget)
        self.NewImage.setGeometry(QtCore.QRect(200, 280, 581, 251))
        self.NewImage.setFrameShape(QtWidgets.QFrame.Box)
        self.NewImage.setObjectName("NewImage")
        self.RadarChartBtn = QtWidgets.QPushButton(self.centralwidget)
        self.RadarChartBtn.setGeometry(QtCore.QRect(30, 470, 121, 61))
        self.RadarChartBtn.setObjectName("RadarChartBtn")
        self.MoreAboutResultsBtn = QtWidgets.QPushButton(self.centralwidget)
        self.MoreAboutResultsBtn.setGeometry(QtCore.QRect(30, 400, 121, 61))
        self.MoreAboutResultsBtn.setObjectName("MoreAboutResultsBtn")
        self.resultsText = QtWidgets.QTextEdit(self.centralwidget)
        self.resultsText.setGeometry(QtCore.QRect(10, 180, 181, 201))
        self.resultsText.setObjectName("resultsText")
        self.ResultsLabel = QtWidgets.QLabel(self.centralwidget)
        self.ResultsLabel.setGeometry(QtCore.QRect(30, 160, 151, 16))
        self.ResultsLabel.setObjectName("ResultsLabel")
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
        self.selectImgBtn.clicked.connect(self.importImage)
        self.RadarChartBtn.clicked.connect(self.ShowRadarChart)
        self.MoreAboutResultsBtn.clicked.connect(self.ShowMoreResult)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.selectImgBtn.setText(_translate("MainWindow", "Выбрать снимок"))
        self.OriginalImage.setText(_translate("MainWindow", "TextLabel"))
        self.NewImage.setText(_translate("MainWindow", "TextLabel"))
        self.RadarChartBtn.setText(_translate("MainWindow", "Диаграмма стадий \n"
"пациента"))
        self.MoreAboutResultsBtn.setText(_translate("MainWindow", "Подробнее о\n"
"результатах"))
        self.ResultsLabel.setText(_translate("MainWindow", "Результат диагностики:"))
    
    def importImage(self,MainWindow):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None,"Выберите изображение","", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        tempfile = 'currentroi.jpg'
        try:
            if filename:
                pixmap = QtGui.QPixmap(filename)
                pixmap = pixmap.scaled(self.OriginalImage.width(), self.OriginalImage.height(), QtCore.Qt.KeepAspectRatio)
                self.OriginalImage.setPixmap(pixmap)
                self.OriginalImage.setAlignment(QtCore.Qt.AlignCenter)
            filename,tempfile = GetROI(filename, tempfile)
            threshold_img, original_img = GetThresholdImage(tempfile)
            newpixmap, contours =  GetContours(threshold_img,original_img)
            #newpixmap =  QtGui.QPixmap('Ik.png')
            h,w,channel = newpixmap.shape
            bytesPerLine =  3 * w
            qImg = QtGui.QImage(newpixmap.data,w,h,bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap01= QtGui.QPixmap.fromImage(qImg)
            self.NewImage.setPixmap(pixmap01)
            self.NewImage.setAlignment(QtCore.Qt.AlignCenter)
            x,y,xy = PrepareContoursForArc(contours)
            arc_x, arc_y = GetArc(x,y,xy)
            n = 8 # polynomial degree
            _, max_x,max_y = GetLeastSquares(arc_x,arc_y,n)
            arc_x,arc_y = PrepareXY(arc_x,arc_y,max_x,max_y)
            a,_,_ = GetLeastSquares(arc_x,arc_y,n)
            a=a[1:]
    
            
            knn_count, forest_count = ClassifyStage(a,'Kerato.csv')
            if (forest_count == 'Stage'):
                knn_count,forest_count = ClassifyStage(a,'KeratoStages.csv')

            filena = os.path.basename(filename)
            for item in knn_count:
                if item == 'N':
                    counter[0][0] = counter[0][0] + 1
                    #stages_filenames['Normal'].append(os.path.basename(filename))
                if item == 'I':
                    counter[0][1] = counter[0][1] + 1
                    #stages_filenames['I'].append(os.path.basename(filename))
                if item == 'II':
                    counter[0][2] = counter[0][2] + 1
                    #stages_filenames['II'].append(os.path.basename(filename))
                if item == 'III':
                    counter[0][3] = counter[0][3] + 1
                    #stages_filenames['III'].append(os.path.basename(filename))
                if item == 'IV':
                    counter[0][4] = counter[0][4] + 1
                    #stages_filenames['IV'].append(os.path.basename(filename))

        
            for item in forest_count:
                if item == 'N':
                    counter[1][0] = counter[1][0] + 1
                    stages_filenames['Normal'].append(os.path.basename(filename))
                    stages_counter[0] += 1
                if item == 'I':
                    counter[1][1] = counter[1][1] + 1
                    stages_filenames['I'].append(os.path.basename(filename))
                    stages_counter[1] += 1
                if item == 'II':
                    counter[1][2] = counter[1][2] + 1
                    stages_filenames['II'].append(os.path.basename(filename))
                    stages_counter[2] += 1
                if item == 'III':
                    counter[1][3] = counter[1][3] + 1
                    stages_filenames['III'].append(os.path.basename(filename))
                    stages_counter[3] += 1
                if item == 'IV':
                    counter[1][4] = counter[1][4] + 1
                    stages_filenames['IV'].append(os.path.basename(filename))
                    stages_counter[4] += 1
                

            global timesRunned
            timesRunned += 1 
            currentmax = np.argmax(stages_counter)
            if timesRunned < 2:
                self.resultsText.setText('Рекомендуется загрузить более одного снимка, в ином случае результат может быть недостоверен.')
            elif timesRunned >= 2:
                if (currentmax == 0 and np.amax(stages_counter[:1])==0):
                    self.resultsText.setText('На основе снимков роговица пациента скорее всего не имеет отклонений от нормы.')
                if (currentmax == 0 and np.amax(stages_counter[:1])>0):
                    self.resultsText.setText('На основе снимков роговица пациента возможно имеет признаки кератоконуса, рекомендуется внимательно изучить снимки, нажав на кнопку "Подробнее о результатах". ')
                if (currentmax == 1):
                    self.resultsText.setText('На основе снимков роговица пациента с большой вероятностью имеет I-ую стадию кератоконус.')
                if (currentmax == 2):
                    self.resultsText.setText('На основе снимков роговица пациента с большой вероятностью имеет II-ую стадию кератоконус.')
                if (currentmax == 3):
                    self.resultsText.setText('На основе снимков роговица пациента с большой вероятностью имеет III-ую стадию кератоконус.')
                if (currentmax == 4):
                    self.resultsText.setText('На основе снимков роговица пациента с большой вероятностью имеет IV-ую стадию кератоконус.')
                else:
                    self.resultsText.setText('На основе снимков роговица пациента с большой вероятностью имеет признаки кератоконуса, рекомендуется внимательно изучить снимки, нажав на кнопку "Подробнее о результатах"')
            print(a)
        except Exception:
            print('Произошла ошибка, возможно вы выбрали не совсем корректное изображение')


    def ShowRadarChart(self,MainWindow):
        SetupPlot(counter,0)

   


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
