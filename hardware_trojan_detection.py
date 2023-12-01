import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
import os
import random
import time

def k_neighbors():
    """
    This function performs classification with k neighbors algorithm.
    """
    x_train, x_test, y_train, y_test = get_data()
    y_train = y_train.reshape((y_train.shape[0], ))
          
    clf = KNeighborsClassifier(n_neighbors=3)
    
    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()

    y_pred = clf.predict(x_test)

    time_ = end - start
    accuracy = 100 * accuracy_score(y_test, y_pred)

    print("### KNN ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))

    return(time_, accuracy)

def logistic_regression():
    """
    This function performs classification with logistic regression.
    """
    x_train, x_test, y_train, y_test = get_data()
    y_train = y_train.reshape((y_train.shape[0], ))

    clf = LogisticRegression(random_state=0, solver='liblinear', max_iter=300,
                             multi_class='ovr')
    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()
    
    y_pred = clf.predict(x_test)

    time_ = end - start
    accuracy = 100 * accuracy_score(y_test, y_pred)

    print("### LR ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))

    return(time_, accuracy)


def support_vector_machine():

    x_train, x_test, y_train, y_test = get_data()
    y_train = y_train.reshape((y_train.shape[0], ))
  
    classifier = SVC(kernel="rbf", C=10, gamma=1)

    start = time.time()
    classifier.fit(x_train, y_train)
    end = time.time()

    y_pred = classifier.predict(x_test)

    time_ = end - start
    accuracy = 100 * accuracy_score(y_test, y_pred)
    
    print("### SVM ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("For C : ", 10, ", Gamma: ", 1, ", kernel = rbf",
          " => Accuracy = %.2f" % (accuracy))
        

    return(time_, accuracy)

def prepare_data():
    excel_files = []
    path = os.getcwd() + '/ROFreq'
    for file in os.listdir(path):
        if file.endswith('.xlsx'):
            excel_files.append(file)


    labels = []
    labels.extend([1 for i in range(0,25)])
    labels[0] = 0
    labels[24] = 0

    df_list = []
    for file in excel_files: 
        df = pd.read_excel(f'ROFreq/{file}',header=None)
        df['labels'] = labels
        df_list.append(df)

    return df_list

def separate_train_test_data(data, samples, case):
    mask = data['labels'] == 0

    trojan_free = []
    trojan_inserted = []

    if(case == 1):
        trojan_free = data.index[mask].tolist()
        trojan_inserted = data.index[~mask].tolist()
        train_data_idx = random.sample(trojan_free,int(samples/2)) + random.sample(trojan_inserted,int(samples/2))
    elif(case == 2):
        trojan_free = data.index[mask].tolist()
        train_data_idx = trojan_free
        train_data_idx = random.sample(train_data_idx,samples)
    elif(case == 3):
        train_data_idx = random.sample(range(0,data.shape[0]),samples)

    
    x_train = data.loc[train_data_idx].reset_index(drop=True)

    test_data_idx = [i for i in range(0,data.shape[0]) if i not in train_data_idx]
    x_test = data.loc[test_data_idx].reset_index(drop=True)

    y_train = pd.DataFrame(x_train['labels']).values
    y_test = pd.DataFrame(x_test['labels']).values
    x_train = x_train.drop(['labels'],axis=1)
    x_test = x_test.drop(['labels'],axis=1)
    return (x_train, x_test, y_train, y_test)

def process_training_test_data(chips, samples, case):
    trojan_free = []
    trojan_inserted = []

    train_data_idx = random.sample(range(0,33),samples)
    test_data_idx = [i for i in range(0,33) if i not in train_data_idx]
    
    
    print(train_data_idx)
    data = [chips[i] for i in train_data_idx]
    data = pd.concat(data,ignore_index=True)

    (x_train, x_test, y_train, y_test) = separate_train_test_data(data, samples, case)


    return (x_train, x_test, y_train, y_test)

def get_data():
    chips = prepare_data()

    # 6 samples
    # Case 1: 3TF, 3TI
    samples = 20
    case = 1
 
    return process_training_test_data(chips,samples,case)
def main():
    
    support_vector_machine()
    print()
    logistic_regression()
    print()
    k_neighbors()

if __name__ == '__main__':
    main()