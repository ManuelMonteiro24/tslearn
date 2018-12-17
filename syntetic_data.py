import numpy as np
import matplotlib.pyplot as plt
import sys

#data 1
#mean_0 = [0,0]
#cov_0 = [[1, ,0], [,0, 1]]

#mean_1 = [0, 0]
#cov_1 = [[1, 0.8],[0.8, 1]]

def write_dataset(dataset_name,data,class_array):
    train_test_str = ["_TRAIN","_TEST"]
    for i in range(0,2):
        ts_instance = 1
        newfile_name = dataset_name + train_test_str[i]
        with open(newfile_name, "w" ) as newfile:
            for j in range(0,len(data[i])):
                ts_index = 1
                for u in range(0,len(data[i][j])):
                    output_str = str(ts_instance) + " " + str(ts_index) + " " + class_array[i][j]
                    for variable in data[i][j][u]:
                        output_str += " " + str(variable)
                    newfile.write(output_str + "\n")

                    ts_index = ts_index + 1

                ts_instance = ts_instance + 1
    return

def group_data_to_ts(data,ts_length):
    datasets = []
    datasets.append([])
    datasets.append([])

    j=0
    while j < len(data[0]):
        ts_helper = []
        for u in range(j,j+ts_length):
            if u > len(data[0]):
                break
            variables_vector=[]
            for i in range(0,len(data)):
                variables_vector.append(data[i][u])
            ts_helper.append(variables_vector)
        if j <= ((len(data[0])/2) -ts_length):
            datasets[0].append(ts_helper) #train
        else:
            datasets[1].append(ts_helper) #test
        j = j + ts_length

    return datasets

dataset_name = "DATASETS/SYNTETICS/syntetic_data_2"
ts_length = 10

#data2
mean_0 = [0, 0,0]
cov_0 = [[1, 0,0],[0,1, 0], [0,0, 1]]

mean_1 = [0, 0,0]
cov_1 = [[1,0, 0.8],[0,1, 0], [0.8,0, 1]]

data_0 = np.random.multivariate_normal(mean_0, cov_0, 200).T
data_0_datasets = group_data_to_ts(data_0,ts_length)

train_class_array = []
test_class_array = []
for i in range(0,len(data_0_datasets[0])):
    train_class_array.append("0")
for i in range(0,len(data_0_datasets[1])):
    test_class_array.append("0")

data_1 = np.random.multivariate_normal(mean_1, cov_1, 200).T
data_1_datasets = group_data_to_ts(data_1,ts_length)
for i in range(0,len(data_1_datasets[0])):
    train_class_array.append("1")
for i in range(0,len(data_1_datasets[1])):
    test_class_array.append("1")

train_dataset = data_0_datasets[0] + data_1_datasets[0]
test_dataset = data_0_datasets[1] + data_1_datasets[1]

write_dataset(dataset_name,[train_dataset, test_dataset],[train_class_array,test_class_array])
