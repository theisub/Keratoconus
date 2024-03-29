import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_curve, auc
def CountOccasions(counter, knn_counter, forest_counter):
    
    for item in knn_counter:
        if item == 'N':
             counter[0][0] = counter[0][0] + 1
        if item == 'I':
            counter[0][1] = counter[0][1] + 1
    
    for item in forest_counter:
        if item == 'N':
             counter[1][0] = counter[1][0] + 1
        if item == 'I':
            counter[1][1] = counter[1][1] + 1

    return counter

        
            
    



def ClassifyStage(a,filename):
    names = ['coef1','coef2','coef3','coef4','coef5','coef6','coef7','coef8','Class']

    #names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(filename, names=names)  
    #dataset = pd.read_csv(url,names=names)
    print(dataset.head())


    X = dataset.iloc[:, :-1].values  
    y = dataset.iloc[:, 8].values  #:,8
    #y = dataset.iloc[:, 4].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)  


    scaler = StandardScaler()  
    scaler.fit(X_train)
    a1  = [-0.07853126746190107,-1.704649206415344,-0.2176969855979349,-0.13270640553260146,0.07922632193566663,-0.1665942015560493,0.06344340069480325,0.13472648397169026]
    X_train = scaler.transform(X_train)
    X_test = np.append(X_test,a)
    X_test = X_test.reshape(1,-1)
    X_test = scaler.transform(X_test)  

    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(X_train, y_train)  

    y_pred = classifier.predict(X_test)  


    #print(confusion_matrix(y_test, y_pred))  
    #print(classification_report(y_test, y_pred))  
    #print ('Accuracy Score :',accuracy_score(y_test[0], y_pred)) 
    #print('Это '+y_test)
    print('kNN думает ' + y_pred)

    #Using the random forest classifier for the prediction 
    classifier=RandomForestClassifier() 
    classifier=classifier.fit(X_train,y_train) 
    predicted=classifier.predict(X_test) 

    print('Лес думает' + predicted)

    return y_pred, predicted




if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


    names = ['coef1','coef2','coef3','coef4','coef5','coef6','coef7','coef8','Class']


    dataset = pd.read_csv('KeratoStages.csv', names=names)  
    #dataset = pd.read_csv(url,names=names)
    print(dataset.head())


    X = dataset.iloc[:, :-1].values  
    y = dataset.iloc[:, 8].values  #:,8
    #y = dataset.iloc[:, 4].values
    rocvalues = []
    avgknn = []
    avgforest =[]
    avgsvm = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  
    scaler = StandardScaler()  
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  
    for i in range(1,100):
        classifier = KNeighborsClassifier(n_neighbors=5)  
        classifier.fit(X_train, y_train)  

        y_pred = classifier.predict(X_test)  

        avgknn = np.append(avgknn,accuracy_score(y_test, y_pred))

        #print(confusion_matrix(y_test, y_pred))  
        print(classification_report(y_test, y_pred))  
        #print ('Accuracy Score :',accuracy_score(y_test, y_pred)) 
        #print('It is '+y_test)
        #print('kNN thinks it it ' + y_pred)

        #print("TEPER FOREST")
        #Using the random forest classifier for the prediction 

        #print("Feature ranking:")

        #for f in range(X.shape[1]):
         #   print("%d. feature %d (%f)" % (f + 1, indices[f]-1, importances[indices[f]-1]))

        #plt.figure()
        #plt.title("Значимость коэффициентов при классификации")
        #plt.bar(range(X.shape[1]), importances[indices-1],
         #   color="r", yerr=std[indices-1]/2, align="center")
        #plt.xticks(range(X.shape[1]), indices)
        #plt.xlim([-1, X.shape[1]])
        #plt.show()
        classifier=RandomForestClassifier(n_estimators=100) 
        classifier=classifier.fit(X_train,y_train) 
        predicted=classifier.predict(X_test) 
        #print ('Accuracy Score :',accuracy_score(y_test, predicted)) 
        avgforest = np.append(avgforest,accuracy_score(y_test, predicted))
        #print ('Report : ') 
        print (classification_report(y_test, predicted)) 
        
        #rocvalues = np.append(rocvalues,metrics.roc_curve(y_test,predicted,pos_label=2))
        #print('forest thinks it it ' + predicted)
        #ClassifyStage('ul')

        #printing the results 
        #print ('Confusion Matrix :') 
        #print(confusion_matrix(y_test, predicted)) 
        #print ('Accuracy Score :',accuracy_score(y_test, predicted)) 
        #print ('Report : ') 
        #print (classification_report(y_test, predicted)) 

        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier

        #print("TEPER SVC")
        classifier = SVC(gamma='auto')
        classifier = classifier.fit(X_train,y_train) 
        predicted = classifier.predict(X_test) 
        
        avgsvm = np.append(avgsvm,accuracy_score(y_test, predicted))

        #print(confusion_matrix(y_test, predicted)) 
        #print ('SVC Accuracy Score :',accuracy_score(y_test, predicted)) 
        #print ('Report : ') 
        #print (classification_report(y_test, predicted)) 
    print(np.average(avgknn))
    print(np.average(avgforest))
    print(np.average(avgsvm))

    print('ehh')