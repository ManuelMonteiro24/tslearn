import numpy, os, sys
from tslearn.datasets import *
from tslearn.utils import *
from tslearn.neighbors import *
from tslearn.preprocessing import *
from tslearn.piecewise import *
from sklearn.metrics import accuracy_score

#metric_name = "min_dist" # "dtw" "euclidean" "min_dist"
#sax_multivariate_output = None # True or None
# "uWave", 3, 315 fixed size not invertible!!! Problem here, ja esta a dar!? porque?
# "Libras" 2, 45 fixed size, este da bem!
# "Pendigits" 2,8 fixed size da bem
# "ECG" 2, variable 39-152
# "ECG_CUTTED", 2,39 fixed size DA BEM
# "CharacterTrajectories" 3, variable,
# "CharacterTrajectories_CUTTED" 3, 109 fixed size da bem
# "LP1-5", 6,15
# Multivariate Classifiers: euclidean K-NN, DTW K-NN, SAX multivar, SAX myapplication

def test_multivariate_methods(dataset_name,metric_name,segments,alphabet,sax_multivariate_output):

    X_train, y_train, X_test, y_test, variables = read_dataset_from_file(dataset_name)
    X_train = separate_atributes_dataset(X_train,variables)
    X_test = separate_atributes_dataset(X_test,variables)

    if metric_name == "min_dist" and sax_multivariate_output == None:
        variables_classifier = 1
        variables_original_ts = variables
    else:
        variables_classifier = variables
        variables_original_ts = None

    if metric_name == "min_dist" and sax_multivariate_output == None:
        X_train = multivariate_normalization(X_train,variables)
        X_test = multivariate_normalization(X_test,variables)
    else:
        X_train = z_normalize(X_train,variables)
        X_test = z_normalize(X_test,variables)

    if metric_name == "min_dist":
        sax_trans = SymbolicAggregateApproximation(n_segments=segments, alphabet_size=alphabet,variables_size=variables, multivariate_output= sax_multivariate_output)
        X_train = sax_trans.fit_transform(X_train,None)
        X_test = sax_trans.fit_transform(X_test,None)
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric_name, metric_params=alphabet, variables_size=variables_classifier,multivariate_output= sax_multivariate_output, variables_original_ts = variables_original_ts)
    else:
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric_name, variables_size=variables_classifier,multivariate_output= sax_multivariate_output,variables_original_ts = variables_original_ts)

    knn_clf.fit(X_train , y_train)
    predicted_labels = knn_clf.predict(X_test)
    acc = accuracy_score(y_test, predicted_labels)

    with open("tests_31_11.txt", "a") as myfile:
        a = str(acc) + " dataset_name: " + dataset_name + " metric_name: " + metric_name
        if metric_name == "min_dist":
            a += " n_segments: " + str(segments) + " alphabet_size: " + str(alphabet)
            if sax_multivariate_output == True:
                a += " sax_multivariate_ind\n"
            else:
                a += "\n"
        else:
            a += "\n"
        print(a)
        myfile.write(a)

    #print("train labels: ", y_train)
    #print("test labels ", y_test)
    #print("predicted_labels: ",predicted_labels)

mode = "sevral_tests"

if mode == "several_tests":
    #datasets = ["Libras","Pendigits","ECG_CUTTED","uWave","CharacterTrajectories_CUTTED"]
    #datasets = ["uWave","CharacterTrajectories_CUTTED"]

    #AUSLAN_CUTTED_CONST_ELIMINATED","ArabicDigits_CUTTED_CONST_ELIMINATED
    #45,4

    #"JapaneseVowels_CUTTED_CONST_ELIMINATED"
    #7

    #"AUSLAN_CUTTED_CONST_ELIMINATED","JapaneseVowels_CUTTED_CONST_ELIMINATED", "LP1_CONST_ELIMINATED","LP2_CONST_ELIMINATED","LP3_CONST_ELIMINATED","LP4_CONST_ELIMINATED","LP5_CONST_ELIMINATED",
    # "CharacterTrajectories_CUTTED","uWave","Wafer_CUTTED_CONST_ELIMINATED"
    # 45,7,15,15,15,15,15,109,104

    datasets = ["ArabicDigits_CUTTED_CONST_ELIMINATED","CharacterTrajectories_CUTTED","uWave"]
    datasets_ts_length = [4,109,315]
    methods = ["dtw"]
    multivar_sax = [None]
    alphabet_list = [5,10,15,20]
    reduction_list = [1/4, 1/2, 3/4,1]
    test_list = []

    for i in range(0,len(datasets)):
        for method_name in methods:
            if method_name == "euclidean" or method_name == "dtw":
                test_list.append([datasets[i],method_name,5,2,None])
            else:
                for multivar in multivar_sax:
                    for a in alphabet_list:
                        for reducted_length in reduction_list:
                            test_list.append([datasets[i],method_name,int(datasets_ts_length[i]*reducted_length),a,multivar])
    for list_element in test_list:
        test_multivariate_methods(list_element[0],list_element[1],list_element[2],list_element[3],list_element[4])
else:
    test_multivariate_methods("CharacterTrajectories_CUTTED","min_dist",54,30,True)
