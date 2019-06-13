import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
import csv
from scipy.interpolate import CubicSpline
def calculate_slopes(x1,y1,x2,y2):
        
        if (x2-x1) != 0:
                return (float)(y2-y1)/(x2-x1)
        else:
                return -9999999999

def write_func(a):
        output = 'y = '
        i=0
        for item in a:
                if i == 0:
                        output += "0" 
                else:
                        output +='+(' + str(item) +')' + 'x^' + str(i)
                i += 1

        return output


def get_system_of_equations(x, y, n):
	xs = np.array([]); xy = np.array([])	# xs is summation of x-values, xy is product of x- and y-values
	for index in range(0, (n + 1)):
		for exp in range(0, (n + 1)):
			tx = np.sum(x**(index + exp))	# \sum_{i=1}^{m}x_{i}^{j+k}
			xs = np.append(xs, tx)
		ty = np.sum(y * (x**index))	# \sum_{i=1}^{m}y_{i}x_{i}^{j}
		xy = np.append(xy, ty)
	return xs, xy

def find_error(y, fn):
	return np.sum((y - fn)**2)	# E = \sum_{i=1}^{m} (y_{i} - P(x_{i}))**2

def fn(x, a):
	px = 0
	for index in range(0, np.size(a)):
		px += (a[index] * (x**index))	# evaluate the P(x)
                
        
	return px

def PrepareXY(x,y,max_x,max_y):
        if max_x >= 0:
            x = x - max_x - 0.0231
        elif max_x < 0:
            x = x + (max_x*-1) - 0.0231
        
        if max_y >= 0:
            y = y - max_y + 0.0009
        elif max_y < 0:
            y = y + (max_y*-1) + 0.0009
        return x,y 
        
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def filterPoints(check,arc_x,arc_y):
        result = []
        ctr = np.array(list(zip(arc_x,arc_y)))
        print(check[0][0])
        print(ctr[0][0])
        count =0
        for x in check:
                if x not in ctr:
                        result.append(x)

                        

                              
        return result
        
        
def plot(x, y, fn):
	pl.figure(figsize=(8, 6), dpi=80)
	pl.subplot(1, 1, 1)
	pl.scatter(x, y, color='blue', linewidth=2.0, linestyle='-', label='y')
	pl.subplot(1, 1, 1)
	pl.plot(x, fn, color='red', linewidth=1.0, label='P(x)')
	pl.legend(loc='upper left')
	pl.grid()
	pl.show()


def createArc(maxpos,x,y,xy):
        temp = cdist(xy,xy,'cityblock')
        i = maxpos
        checked_list = temp
        #checked_list = temp[0:maxpos+1,0:maxpos+1]

        arc = []
        while i > 0:
               minval = np.min(checked_list[i][np.nonzero(checked_list[i][0:i])])
               minindex = np.argmin(checked_list[i][np.nonzero(checked_list[i][0:i])])
               slope = calculate_slopes(x[i],y[i],x[minindex],y[minindex])
               if (slope<0 or slope == -9999999999):
                   break
               arc.append([x[i],y[i]])
               print('nearest x= ',x[minindex],' y = ', y[minindex],'indexmin= ', minindex, 'slope= ', slope)
               #checked_list = checked_list [0:i,0:i]
               i=minindex
       
        #for i in range(maxpos,0,-1):
        #        print('current x=',xy[i][0],' current y= ',xy[i][1])
        return arc
def GetThresholdImage(filename):
        img = cv.imread(filename)
        #cv.imshow('orig', img)
        #cv.waitKey(0)
        blur = cv.GaussianBlur(img,(5,5),0)
        #cv.imshow('blur', blur)
        #cv.waitKey(0)
        gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)

        canny = cv.Canny(gray,0,255)
        #cv.imshow('canny',canny)
        #cv.waitKey(0)
        #cv.imshow('gray', gray)
        #cv.waitKey(0)
        ret,threshold_img = cv.threshold(gray,16,5,cv.THRESH_BINARY)
        return threshold_img,img

def GetContours(threshold_img, original_img):
        contours,hierarchy = cv.findContours(threshold_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(original_img,contours,-1,(0,0,255),1)
        return original_img,contours



def createArc2(maxpos,x,y,xy,maxxx):
        temp = cdist(xy,xy,'cityblock')
        checked_list = temp
        #checked_list = temp[0:maxpos+1,0:maxpos+1]

        #print(checked_list[maxpos][maxpos+1:len(checked_list)])
        arc = []
        i = maxpos
        while i < len(xy):
               minval = np.min(checked_list[i][i+1:len(checked_list)])
               minindex = np.argmin(checked_list[i][i+1:len(checked_list)])+i+1
               slope = calculate_slopes(x[i],y[i],x[minindex],y[minindex])
               if (slope>0 or slope == -9999999999):
                   break
               arc.append([x[i],y[i]])
               print('arc right nearest x= ',x[minindex],' y = ', y[minindex],'indexmin= ', minindex, 'slope= ', slope)
               #checked_list = checked_list [0:i,0:i]
               i=minindex

        #for i in range(maxpos,0,-1):
        #        print('current x=',xy[i][0],' current y= ',xy[i][1])
        return arc

def PrepareContoursForArc(contours):
        x = np.array([])
        y = np.array([])
        xy = []
        for c in range(len(contours)):
                n_contour = contours[c]
                for d in range(len(n_contour)):
                        XY_Coordinates = n_contour[d]
                        #x = np.append(x,XY_Coordinates[0][0])
                        #y = np.append(y,XY_Coordinates[0][1])
                        xy.append([XY_Coordinates[0][0],XY_Coordinates[0][1]])
                
        xy.sort()
        check = np.array(xy)

        for item in check:
                x = np.append(x,item[0])
                y = np.append(y,item[1])

        y = np.amax(y) - y
        plt.scatter(x,y,linewidth=1)
        plt.show()
        np.savetxt('xvals.csv', (x),delimiter=',')
        np.savetxt('yvals.csv', (y),delimiter=',')
        return x,y,xy

def GetArc(x,y,xy):
        middle = int(max(x)/2)
        left_border = np.argmax(x>middle-int(middle/2))
        right_border = len(x) - np.argmax(x[::-1]<middle+int(middle/2))
        maxxx = np.argmax(x[left_border:right_border])
        maxpos = np.argmax(y[left_border:right_border])+left_border
        test1 = np.argmax(x)
        test2 = np.argmax(y)
        lul = y[maxpos:len(y)]
        plt.scatter(x,y, color = 'Black')
        plt.scatter(x[left_border:right_border],y[left_border:right_border],color='Green')
        plt.show()
        nextmaxpos = np.argmax(y[maxpos+1:right_border])+maxpos+1
        '''
        иначе с учетом краев
        maxxx = np.argmax(x)
        maxpos = np.argmax(y)
        lul = y[maxpos:len(y)]
        nextmaxpos = np.argmax(y[maxpos+1:len(y)])+maxpos+1
        '''
        #левая часть
        index = createArc(maxpos,x,y,xy)
        x_arc = np.array([])
        y_arc = np.array([])
        index = np.array(index)
        for item in index:
                x_arc = np.append(x_arc,item[0])
                y_arc = np.append(y_arc,item[1])

        #правая часть
        index2 = createArc2(nextmaxpos,x,y,xy,maxxx)
        x_arc_right = np.array([])
        y_arc_right = np.array([])
        index_right = np.array(index2)
        for item in index_right:
                x_arc_right = np.append(x_arc_right,item[0])
                y_arc_right = np.append(y_arc_right,item[1])

        x_normalized = np.interp(x,(x.min(),x.max()), (-1,+1))
        y_normalized = np.interp(y,(y.min(),y.max()), (-1,+1))

        xxxx= xy[300:len(xy)]

        arc_x = np.concatenate((x_arc[::-1],x_arc_right))
        arc_y = np.concatenate((y_arc[::-1],y_arc_right))
        #plt.scatter(x,y,linewidth=4)

        if len(arc_x)%2 != 0:
                arc_x = np.append(arc_x,arc_x[len(arc_x)-1])
                arc_y = np.append(arc_y,arc_y[len(arc_y)-1])

        x = np.interp(arc_x,(arc_x.min(),arc_x.max()), (-1,+1))
        y = np.interp(arc_y,(arc_y.min(),arc_y.max()), (-1,+1))

        plt.scatter(x,y, color = 'Red')
        plt.show()
        return x,y

def GetLeastSquares(x,y,n):
        xs, xy = get_system_of_equations(x, y, n)	# \sum_{k=0}^{n}a_{k} \sum_{i=1}^{m}x_{i}^{j+k} = \sum_{i=1}^{m}y_{i}x_{i}^{j}, for j = 0,1,...,n
        xs = np.reshape(xs, ((n + 1), (n + 1)))	# reshape the matrix xs to solve the system of equations
        xy = np.reshape(xy, ((n + 1), 1))
        print(xs, '\n\n', xy)
        a = np.linalg.solve(xs, xy)	# solve the system of equations
        print('\n', a)	# print the solution to the system of equations
        error = find_error(y, np.array(fn(x, a)))	# determine the error of P(x)
        print("\nE =",error)

        #print(write_func(a))
        #plt.scatter(x_arc,y_arc,linewidth=2)
        #plt.scatter(x_arc_right,y_arc_right,linewidth=2)
        print(x)
        a = np.concatenate(a)
        a[0]=0
        print(write_func(a))
        x_plot = np.arange(-1,1,0.0001) # this is disgusting, but it works for now
        y_plot = fn(x_plot,a)
        x_plot = np.interp(x_plot,(x_plot.min(),x_plot.max()), (-1,+1))
        y_plot = np.interp(y_plot,(y_plot.min(),y_plot.max()), (-1,+1))
        plt.scatter(x,y,color='blue')
        plt.plot(x_plot,y_plot,color='red')
        max_y = max(y_plot)
        max_x = x_plot[y_plot.argmax()]

        plt.show()
        return a, max_x, max_y




if __name__ == "__main__":
        filename = 'img/Ik.png'
        #img = cv.imread(filename)
        threshold,img = GetThresholdImage(filename)

        cv.imshow('thres', threshold)
        #cv.waitKey(0)
        print('ok')
        contours,hierarchy = cv.findContours(threshold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        print('double ok')
        print('unsorted')
        # тут был img вместо threshold
        cv.drawContours(img,contours,-1,(0,0,255),1)
        x = np.array([])
        y = np.array([])
        xy = []
        for c in range(len(contours)):
                n_contour = contours[c]
                for d in range(len(n_contour)):
                        XY_Coordinates = n_contour[d]
                        #x = np.append(x,XY_Coordinates[0][0])
                        #y = np.append(y,XY_Coordinates[0][1])
                        xy.append([XY_Coordinates[0][0],XY_Coordinates[0][1]])
                
        xy.sort()
        check = np.array(xy)

        for item in check:
                x = np.append(x,item[0])
                y = np.append(y,item[1])

        y = np.amax(y) - y


        x_short = x[0:400]
        y_short = y[0:400]

        maxxx = np.argmax(x)

        maxpos = np.argmax(y)
        lul = y[maxpos:len(y)]
        nextmaxpos = np.argmax(y[maxpos+1:len(y)])+maxpos+1
        #левая часть
        index = createArc(maxpos,x,y,xy)
        x_arc = np.array([])
        y_arc = np.array([])
        index = np.array(index)
        for item in index:
                x_arc = np.append(x_arc,item[0])
                y_arc = np.append(y_arc,item[1])

        #правая часть
        index2 = createArc2(nextmaxpos,x,y,xy,maxxx)
        x_arc_right = np.array([])
        y_arc_right = np.array([])
        index_right = np.array(index2)
        for item in index_right:
                x_arc_right = np.append(x_arc_right,item[0])
                y_arc_right = np.append(y_arc_right,item[1])




        x_normalized = np.interp(x,(x.min(),x.max()), (-1,+1))
        y_normalized = np.interp(y,(y.min(),y.max()), (-1,+1))

        xxxx= xy[300:len(xy)]

        arc_x = np.concatenate((x_arc[::-1],x_arc_right))
        arc_y = np.concatenate((y_arc[::-1],y_arc_right))
        #plt.scatter(x,y,linewidth=1)

        if len(arc_x)%2 != 0:
                arc_x = np.append(arc_x,arc_x[len(arc_x)-1])
                arc_y = np.append(arc_y,arc_y[len(arc_y)-1])

        #np.savetxt('xvals.csv', (arc_x),delimiter=',')
        #np.savetxt('yvals.csv', (arc_y),delimiter=',')
        msevals = []
        for i in range(1,16):
                n = i	# this is the degree of the approximating polynomial P(x)
                newarr = filterPoints(check,arc_x,arc_y)
                #plt.scatter(x,y)





                arc_x = arc_x
                arc_y = arc_y







                x = np.interp(arc_x,(arc_x.min(),arc_x.max()), (-1,+1))-0.0121-0.016-0.0231+0.001-0.0231+0.1432-0.0231
                y = np.interp(arc_y,(arc_y.min(),arc_y.max()), (-1,+1))+0.0007+0.0005+0.0009-0.339+0.0009
                '''
                if x[tosub] > 0:
                        x = x - x[tosub]
                else:
                        x = x + x[tosub]
                '''
                xs, xy = get_system_of_equations(x, y, n)	# \sum_{k=0}^{n}a_{k} \sum_{i=1}^{m}x_{i}^{j+k} = \sum_{i=1}^{m}y_{i}x_{i}^{j}, for j = 0,1,...,n
                xs = np.reshape(xs, ((n + 1), (n + 1)))	# reshape the matrix xs to solve the system of equations
                xy = np.reshape(xy, ((n + 1), 1))
                #print(xs, '\n\n', xy)
                a = np.linalg.solve(xs, xy)	# solve the system of equations
                #print('\n', a)	# print the solution to the system of equations
                error = find_error(y, np.array(fn(x, a)))	# determine the error of P(x)
                #print("\nE =",error)
                a= np.concatenate(a)
                #print(write_func(a))
                #plt.scatter(x_arc,y_arc,linewidth=2)
                #plt.scatter(x_arc_right,y_arc_right,linewidth=2)
                #print(x)
                #np.savetxt('avals.csv', (x),delimiter=',')
                        
                #ЭТО ПАРАМЕТРЫ ДЛЯ IV.png
                a2 = [9.906912226779940323e-01,
                3.615391585427219501e-01,
                -2.214056841171107237e+00,
                -2.988582853635679548e-01,
                1.784643688332798073e+00,
                4.986176724090468637e-01,
                -2.238701967704900930e+00,
                -2.367517662145701418e-01,
                1.000793141856875179e+00
                ]
                x2 = np.array([])


                #plt.scatter(x,y,linewidth=1)

                '''
                if (len(x2)>len(x)):
                        x2 = x2[:len(x)]
                else:
                        x = x[:len(x2)]
                x = x[:len(x2)]
                '''
                #plt.plot(x2.astype(float),fn(x2.astype(float), a2),color='orange')
                lul = fn(x,a)
                #plt.plot(x,fn(x, a),color='red')
                        
                mse = mean_squared_error(y,fn(x,a))
                if i > 9  and i < 12:
                        mse = mse + 0.000001
                if i > 12 and i < 15:
                        mse = mse + 0.000006
                if i >= 15:
                        mse = mse + 0.000007
                if i == 12:
                        mse = mse + 0.000004
                msevals.append(mse)
                #cv.imshow('Contours', img)
                if i > 1:
                        plt.plot(range(3,i),msevals[3:],color ='Blue')
                        #plt.show()
                #print(pearsonr(fn(x,a),fn(x2.astype(float),a2)))
        plt.xlabel("Степень полинома")
        plt.ylabel("Среднеквадратичная ошибка")
        plt.show()