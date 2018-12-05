import os, sys

variables_size = 22
dataset_name = "AUSLAN"
limit_to_cut =45

dirname = os.path.abspath(os.path.dirname(sys.argv[0]))
train_test_str = ["_TRAIN","_TEST"]
for i in range(0,2):
    file_location = dirname + "/" + dataset_name + train_test_str[i]
    with open(file_location) as fin:
        newfile_name = dataset_name + "_CUTTED" + train_test_str[i]
        with open(newfile_name, "w") as newfile:
            for line in fin:
                aux_line = line.split()
                if len(aux_line) != variables_size + 3:
                    continue
                elif int(aux_line[1]) > limit_to_cut:
                    continue
                else:
                    newfile.write(line)
