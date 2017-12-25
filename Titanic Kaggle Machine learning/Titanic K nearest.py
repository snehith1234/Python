import numpy as np
import csv
from numpy.random import shuffle
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
#Functions Column
def name_remover():
    for i in range(len(values)):
        if i == 3:
            values.remove(values[i])
    return values
def name_list_maker():
    for i in range(len(values)):
        if i == 3:
            if values[i] not in last_name:
                last_name.append(values[i])
            else:
                continue
def name_lable():
    for i in range(len(last_name)):
        if last_name[i] == values[3]:
            values[3] = int(i)
    return values
def gender():
    for i in range(len(values)):
        if values[4] == 'male':
            values[4] = '0'
        elif values[4] == 'female':
            values[4] = '1'
    return values
def age_median():
    if (len(age)%2 == 0):
        point = ((len(age)/2)+ 1)
        median = age[point]
    else:
        point = (len(age)/2)
        median = age[point]
    return median
def ticket_list_maker():
    for i in range(len(values)):
        if i == 8:
            if values[i] not in tickets:
                tickets.append(values[i])
            else:
                continue
def ticket_lable():
    for i in range(len(tickets)):
        if tickets[i] == values[8]:
            values[8] = int(i)
    return values

def fare_train():
    for i in values:
        values[9] = (float(values[9]))
    return values

def cabin_remover():
    for i in range(len(values)):
        if i == 10:
            values.remove(values[i])
    return values
def embar_list_marker():
    for i in range(len(values)):
        if i == 10:
            if values[i] not in embar:
                embar.append(values[i])
        else:
            continue
def embar_lable():
    for i in range(len(embar)):
        if embar[i] == values[10]:
            values[10] = int(i)
    return values
def age_setter(median):
    for i,j in enumerate(traindata_features):
        for k in range(len(j)):
            if j[4] == '':
                j[4] = median
    return traindata_features
def fare_age_setter():
    for i,j in enumerate(traindata_features):
        for k in range(len(j)):
            if j[8] == '':
                j.remove(j[8])
                j.insert(4,median)
    return traindata_features
############################################End of train data functions###########################################
######################################Memory units data process########################
traindata_features = []
Survived = {'0': 0,'1':1}
traindata_lables = []
last_name = []
age= []
tickets=[]
embar =[]
###########################################Training file reading#################################
traindatain = open("C:/Users/snehi/Desktop/Machine learning/task/train.csv")
data_train = traindatain.readlines()[1:]
#list of traindata,lables and testdata
for lines in data_train:
    lines = lines.strip('\n')
    values = lines.split(",")
    name_remover()
    name_list_maker()
    name_lable()
    gender()
    age.append(values[5])
    ticket_list_maker()
    ticket_lable()
    fare_train() 
    cabin_remover()
    embar_list_marker()
    embar_lable()
    features = []
    for i in range(len(values)):
        if i == 1:
            traindata_lables.append(Survived[str(values[1])])
        elif i != 1:
            features.append(values[i])
    traindata_features.append(features)

median = age_median()
age_setter(median)
fare_age_setter()

#Now to get ready with test data for the input
testdatain = open("C:/Users/snehi/Desktop/Machine learning/task/test.csv")
data_test = testdatain.readlines()[1:]
############################## test functions##################################
def test_name_remover():
    for i in range(len(values)):
        if i == 2:
            values.remove(values[2])
    return values
def test_name_list_maker():
    for i in range(len(values)):
        if i == 2:
            if values[i] not in last_name:
                last_name.append(values[i])
            else:
                continue
def test_name_lable():
    for i in range(len(last_name)):
        if last_name[i] == values[2]:
            values[2] = int(i)
    return values
def test_gender():
    for i in range(len(values)):
        if values[3] == 'male':
            values[3] = 0
        elif values[3] == 'female':
            values[3] = 1
    return values

def test_age_median():
    if (len(test_age)%2 == 0):
        point = ((len(test_age)/2)+ 1)
        test_median = test_age[point]
    else:
        point = (len(test_age)/2)
        test_median = test_age[point]
    return  test_median

def test_ticket_lable_maker():
    for i in range(len(values)):
        if values[7] not in tickets:
            tickets.append(values[7])
        else:
            continue
def test_ticket_lable():
    for i in range(len(tickets)):
        if tickets[i] == values[7]:
            values[7] = int(i)
    return values

def test_fare_median():
    if (len(test_fare)%2 == 0):
        point = ((len(test_fare)/2)+ 1)
        fare_median = test_fare[point]
    else:
        point = (len(test_fare)/2)
        fare_median = test_fare[point]
    return fare_median

def fare_test_floater():
    for i,j in enumerate(testdata_features):
        for k in range(len(j)):
            j[8] = float(j[8])
    return testdata_features

def test_cabin_remover():
    for i in range(len(values)):
        if i == 9:
            values.remove(values[i])
    return values
def test_embar_lable():
    for i in range(len(embar)):
        if embar[i] == values[9]:
            values[9] = int(i)
    return values
def test_age_setter(test_median):
    for i,j in enumerate(testdata_features):
        for k in range(len(j)):
            if j[4] == '':
                j[4] = test_median
    return testdata_features
def test_missage_fare_setter():
    for i,j in enumerate(testdata_features):
        for k in range(len(j)):
            if j[8] == '':
                j.remove(j[8])
                j.insert(4,test_median)
    return testdata_features
#############################################End of test data functions############################################

############################## test memory units #######################################
testdata_features = []
test_age = []
test_fare = []
############################## Test data process #######################################
for lines in data_test:
    lines = lines.strip('\n')
    values = lines.split(",")
    test_name_remover()
    test_name_list_maker()
    test_name_lable()
    test_gender()
    test_ticket_lable_maker()
    test_ticket_lable()
    test_fare.append(values[8])
    test_cabin_remover()
    test_embar_lable()
    test_age.append(values[4])
    t_features = []
    for i in range(len(values)):
        t_features.append(values[i])
    testdata_features.append(t_features)
#fare_median =test_fare_median()
test_median = test_age_median()
test_age_setter(test_median)
test_missage_fare_setter()
fare_test_floater()
###################################test && train convertion to numpy######################

for i,j in enumerate(testdata_features):
        for k in range(len(j)):
            j[k] = float(j[k])
           
print testdata_features
for i,j in enumerate(traindata_features):
        for k in range(len(j)):
            j[k] = float(j[k])
            
print traindata_features

################################################################################
num_train_few = np.array(traindata_features)
num_test_few = np.array(testdata_features)
X = num_train_few[:,1:]
X1 = num_test_few[:,1:]
numobs = X.shape[0]
numobsX1 = X1.shape[0]
############################ test data to be predicted VALIDX_1 ########################
VALIDX_1 = X1[:int(numobsX1),:]
########################################################################################
################################### Finding K's for KNN ################################
all_ks = range(1,((numobsX1)/2))
k_best = np.zeros(len(all_ks))
num_of_exp = 1
for c in range(num_of_exp):
    print('Experiment no #%d' % (c))
    #shuffles features and labels
    inds = list(range(numobs))
    shuffle(inds)
    X = X[inds,:]
    Y = [traindata_lables[i] for i in inds]    
    TRAINX = X[:int(numobs * 0.85),:]
    TRAINY = Y[:int(numobs *0.85)]
    #standardize the features by features scaling
    VALIDX = X[:int(numobs *0.85):,:]
    VALIDY = Y[:int(numobs * 0.85):]
    scaler = preprocessing.StandardScaler().fit(TRAINX)
    TRAINX = scaler.transform(TRAINX)
    VALIDX = scaler.transform(VALIDX)
    VALIDX1= scaler.transform(VALIDX_1)
    allerrors = []
    for k in all_ks:
        knn_classifier = neighbors.KNeighborsClassifier(k)
        knn_classifier.fit(TRAINX,TRAINY)
        #print knn_classifier
        error = 0
        predict_c = knn_classifier.predict(VALIDX)
        for i in range(len(predict_c)):
            if predict_c[i] != VALIDY[i]:
                error += 1
        error_rate = error * 1.0/len(predict_c)
        allerrors.append([k,error_rate])
    allerrors = np.array(allerrors)
    #print allerrors
    bestk = np.min(allerrors[:,1])
    #print bestk
    bestk = allerrors[allerrors[:,1] == np.min(allerrors[:,1]),0]
    for k in bestk:
        k_best[all_ks.index(k)] += 1
print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
print k_best
print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
plt.bar(all_ks,k_best)
plt.show()
knn_classifier = neighbors.KNeighborsClassifier(max(k_best))
knn_classifier.fit(TRAINX,TRAINY)
#print knn_classifier
predict_c = knn_classifier.predict(VALIDX1)
print predict_c
data = []
predict_lables = []
for i in list(testdata_features):
    data.append(int(i[0]))
for i in list(predict_c):
    predict_lables.append(i)
predictions_file = open("C://Users//snehi//Desktop//data.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(data,predict_lables))
predictions_file.close()
print "Done"
