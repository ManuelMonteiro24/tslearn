import os, sys

variables_size = 3
dataset_name = "Epilepsy"

class_dictionary = {}
class_count = 1

dirname = os.path.abspath(os.path.dirname(sys.argv[0]))
train_test_str = ["_TRAIN","_TEST"]
for i in range(0,2):
    arff_data_flag = 0
    series_count = 1
    file_location = dirname + "/" + dataset_name + train_test_str[i] + ".arff"
    with open(file_location) as fin:
        newfile_name = dataset_name + train_test_str[i] + "_FORMATED"
        with open(newfile_name, "w") as newfile:
            for line in fin:

                if arff_data_flag == 0:
                    if line == "@data\n": #check for start of dataset values
                        arff_data_flag = 1
                    continue
                line = line.split(",")

                attribute_iterator = 0
                class_value = None
                ts_helper = []
                for j in range(0,variables_size):
                    ts_helper.append([])

                for j in range(0,len(line)):
                    if "\\n" in line[j]:
                        splitted_lines = line[j].split("\\n")
                        ts_helper[attribute_iterator].append(float(splitted_lines[0]))
                        attribute_iterator = attribute_iterator + 1
                        ts_helper[attribute_iterator].append(float(splitted_lines[1]))
                    elif j == (len(line)-1):
                        if line[j] in class_dictionary:
                            class_value = class_dictionary[line[j]]
                        else:
                            class_dictionary[line[j]] = class_count
                            class_value = class_count
                            class_count = class_count + 1
                    elif "'" in line[j]:
                        formated_value = line[j].replace("'","")
                        ts_helper[attribute_iterator].append(float(formated_value))
                    elif '"' in line[j]:
                        formated_value = line[j].replace('"',"")
                        ts_helper[attribute_iterator].append(float(formated_value))
                    else:
                        ts_helper[attribute_iterator].append(float(line[j]))

                for j in range(0,len(ts_helper[0])):
                    line_to_write = ""
                    line_to_write += str(series_count) + " " + str(j+1) + " " + str(class_value)
                    for u in range(0,variables_size):
                        line_to_write += " " + str(ts_helper[u][j])
                    line_to_write += "\n"
                    newfile.write(line_to_write)
                series_count = series_count + 1
