import os, sys, numpy

variables_size = 6
dataset_name = "Wafer"

#initializations
dirname = os.path.abspath(os.path.dirname(sys.argv[0]))
train_test_str = ["_TRAIN","_TEST"]

for variable_iterator in range(1,variables_size+1):
    for i in range(0,2):
        file_location = dirname + "/" + dataset_name + train_test_str[i]
        with open(file_location) as fin:
            newfile_name = dataset_name + "_" + str(variable_iterator) + train_test_str[i]
            with open(newfile_name, "w" ) as newfile:

                ts_index_counter = None
                ts_str = ""
                class_str = ""
                final_line_str = ""

                for line in fin:
                    line = line.split()

                    if len(line) < 4: #file not standarized
                            continue

                    if ts_index_counter == None:
                        ts_index_counter = int(line[0])

                    if int(line[0]) == ts_index_counter: #same series case
                        ts_str  = ts_str  + " " + line[2+variable_iterator]
                        class_str = line[2]

                    else: #first element of new ts case
                        #write to file saved ts
                        newfile.write(class_str + ts_str + "\n")
                        ts_str = ""
                        class_str = ""
                        final_line_str = ""
                        ts_index_counter = int(line[0])
                        #process new one
                        ts_str  = ts_str + " " + line[2+variable_iterator]
                        class_str = line[2]

                #last ts of file case
                newfile.write(class_str + ts_str + "\n")
                ts_index_counter = None
