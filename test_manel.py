import numpy, os, sys
from tslearn.datasets import *
from tslearn.utils import *
from tslearn.neighbors import *
from tslearn.preprocessing import *
from tslearn.piecewise import *
from sklearn.metrics import accuracy_score

#metric_name = "min_dist" # "dtw" "euclidean" "min_dist"
#sax_multivariate_output = None # True or None
# "uWave", 3, 315 fixed size not invertible
# "Libras" 2, 45 fixed size
# "Pendigits" 2,8 fixed size
# "ECG" 2, variable 39-152
# "ECG_CUTTED" 2,39
# "CharacterTrajectories" 3, variable
# Multivariate Classifiers: euclidean K-NN, DTW K-NN, SAX multivar, SAX myapplication

def test_multivariate_methods(dataset_name,variables,ts_length,metric_name,segments,alphabet,sax_multivariate_output):
    if metric_name == "min_dist" and sax_multivariate_output == None:
        variables_classifier = 1
    else:
        variables_classifier = variables

    X_train, y_train, X_test, y_test = read_dataset_from_file(dataset_name, variables, ts_length)
    X_train = separate_atributes_dataset(X_train,variables)
    X_test = separate_atributes_dataset(X_test,variables)

    if metric_name == "min_dist" and sax_multivariate_output == None:
        X_train = multivariate_normalization(X_train,variables)
        X_test = multivariate_normalization(X_test,variables)
    else:
        X_train = z_normalize(X_train,variables)
        X_test = z_normalize(X_test,variables)

    #for i in range(0,len(X_train[0])):
    #    print("cov: ", numpy.cov([X_train[0][i], X_train[1][i]]))
    #    print("mean1 :", ts_mean_single_var(X_train[0][i]))
    #    print("mean2: ", ts_mean_single_var(X_train[1][i]))
    #sys.exit()

    if metric_name == "min_dist":
        sax_trans = SymbolicAggregateApproximation(n_segments=segments, alphabet_size_avg=alphabet,variables_size=variables, multivariate_output= sax_multivariate_output)
        X_train = sax_trans.fit_transform(X_train,None)
        X_test = sax_trans.fit_transform(X_test,None)
    #print(X_train_sax[0])

    if metric_name == "min_dist":
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric_name, metric_params=alphabet, variables_size=variables_classifier,multivariate_output= sax_multivariate_output)
    else:
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=metric_name, variables_size=variables_classifier,multivariate_output= sax_multivariate_output)

    knn_clf.fit(X_train , y_train)
    #print("X_test:", len(X_test))
    predicted_labels = knn_clf.predict(X_test)
    #print("y_test", len(y_test))
    #print("predicted_labels", len(predicted_labels))
    acc = accuracy_score(y_test, predicted_labels)

    with open("several_tests.txt", "a") as myfile:
        a = "accuracy_score: " + str(acc) + " dataset_name: " + dataset_name + " #atributes: " + str(variables) + " ts_length: " + str(ts_length) + " metric_name: " + metric_name
        if metric_name == "min_dist":
            a += " n_segments: " + str(segments) + " alphabet_size: " + str(alphabet)
            if sax_multivariate_output == True:
                a += " sax_multivariate_output\n"
            else:
                a += "\n"
        else:
            a += "\n"
        print(a)
        myfile.write(a)

    #print("train labels: ", y_train)
    #print("test labels ", y_test)
    #print("predicted_labels: ",predicted_labels)

mode = "1_test"

if mode == "several_tests":
    datasets = ["Libras","Pendigits","ECG_CUTTED"]
    datasets_variables = [2,2,2]
    datasets_ts_length = [45,8,39]
    methods = ["euclidean","dtw","min_dist"]
    multivar_sax = [None,True]

    test_list = []

    for i in range(0,len(datasets)):
        for method_name in methods:
            if method_name == "euclidean" or method_name == "dtw":
                test_list.append([datasets[i],datasets_variables[i],datasets_ts_length[i],method_name,5,2,None])
            else:
                for multivar in multivar_sax:
                    length = datasets_ts_length[i]
                    j=2
                    while length > 2 or j < 6:
                        length = int(datasets_ts_length[i]/j)
                        for alphabet_iterator in range(2,11):
                            test_list.append([datasets[i],datasets_variables[i],datasets_ts_length[i],method_name,length,alphabet_iterator,multivar])
                        j=j+1
    for list_element in test_list:
        test_multivariate_methods(list_element[0],list_element[1],list_element[2],list_element[3],list_element[4],list_element[5],list_element[6])
else:
    test_multivariate_methods("uWave",3,315,"min_dist",150,2,None)
