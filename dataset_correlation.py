import numpy, os, sys, mpmath
from tslearn.datasets import *
from tslearn.utils import *
from tslearn.neighbors import *
from tslearn.preprocessing import *
from tslearn.piecewise import *
from sklearn.metrics import accuracy_score

def mean_var_matrix_calculation(dataset_name,covariance_matrixs_vector,variables_size):

    cov_separate_values_vector = []
    var_array = numpy.zeros(variables_size**2)
    mean_array = numpy.zeros(variables_size**2)
    for i in range(0,variables_size**2):
        cov_separate_values_vector.append([])

    iteratorr = 0
    for i in range(0,variables_size):
        for j in range(0,variables_size):
            for u in range(0,len(covariance_matrixs_vector)):
                cov_separate_values_vector[iteratorr] = covariance_matrixs_vector[u][i][j]
            iteratorr = iteratorr + 1

    for i in range(0,len(cov_separate_values_vector)):
        #mean_array[i] = numpy.mean(cov_separate_values_vector[i])
        mean_array[i] = numpy.mean(abs(cov_separate_values_vector[i]))
        var_array[i] = numpy.var(cov_separate_values_vector[i])


    final_mean = numpy.zeros(shape=(variables_size,variables_size))
    final_var = numpy.zeros(shape=(variables_size,variables_size))
    iteratorr = 0
    for i in range(0,variables_size):
        for j in range(0,variables_size):
            final_mean[i][j] = mean_array[iteratorr]
            final_var[i][j] = var_array[iteratorr]
            iteratorr = iteratorr + 1

    print("final_mean", final_mean)
    with open("covariance_matrix", "a" ) as newfile:
        newfile.write(dataset_name +"\n"+ str(final_mean) + "\n")

    return [final_mean,final_var]

def covariance_matrix_infos(dataset_name,data,variables_size):
    covariance_matrixs_vector = []

    for train_test in range(0,2):
        for j in range(0,len(data[train_test][0])):
            ts_s = []
            for u in range(0,variables_size):
                ts_s.append(data[train_test][u][j])
            covariance_matrixs_vector.append(numpy.cov(ts_s))

    mean_var_matrix_calculation(dataset_name,covariance_matrixs_vector,variables_size)

    return

def dataset_correlation(dataset_name):

    X_train, y_train, X_test, y_test, variables = read_dataset_from_file(dataset_name)
    X_train = separate_atributes_dataset(X_train,variables)
    X_test = separate_atributes_dataset(X_test,variables)
    X_train = z_normalize(X_train,variables)
    X_test = z_normalize(X_test,variables)

    covariance_matrix_infos(dataset_name,[X_train, X_test],variables)
    return




if 1 == 0:
    dataset_correlation("DATASETS/ECG_CUTTED")
else:
    datasets = ["CharacterTrajectories_CUTTED","uWave","Libras","Pendigits",
    "ECG_CUTTED","AUSLAN_CUTTED_CONST_ELIMINATED","JapaneseVowels_CUTTED_CONST_ELIMINATED",
    "LP1_CONST_ELIMINATED","LP2_CONST_ELIMINATED","LP3_CONST_ELIMINATED","LP4_CONST_ELIMINATED","LP5_CONST_ELIMINATED",
    "Wafer_CUTTED_CONST_ELIMINATED","AtrialFibrilation_FORMATED",
    "Epilepsy_FORMATED","ERing_FORMATED","EthanolConcentration_FORMATED",
    "StandWalkJump_FORMATED"]
    for dataset in datasets:
        input_str = "DATASETS/" + dataset
        dataset_correlation(input_str)
