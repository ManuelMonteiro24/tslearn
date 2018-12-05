import os, sys, numpy

variables_size = 22
dataset_name = "AUSLAN_CUTTED"
ts_len = 45

#initializations
dirname = os.path.abspath(os.path.dirname(sys.argv[0]))
train_test_str = ["_TRAIN","_TEST"]
serie_counter = 0
line_saver_vector = []
variable_test = []
for i in range(0,variables_size):
    variable_test.append([])

for i in range(0,2):
    file_location = dirname + "/" + dataset_name + train_test_str[i]
    with open(file_location) as fin:
        newfile_name = dataset_name + "_CONST_ELIMINATED" + train_test_str[i]
        with open(newfile_name, "w") as newfile:
            for line in fin:
                aux_line = line.split()

                #file not standardized
                if len(aux_line) <= 4:
                    continue

                #end of ts reached, analyse
                if serie_counter == ts_len:
                    for j in range(0,variables_size):
                        if numpy.var(numpy.array(variable_test[j]).astype(numpy.float)) == 0:
                            line_saver_vector = []
                    if len(line_saver_vector) != 0:
                        for j in line_saver_vector:
                            newfile.write(j)
                    for w in range(0,variables_size):
                        variable_test[w]=[]
                    line_saver_vector = []
                    serie_counter =0

                line_saver_vector.append(line)
                for j in range(0,variables_size):
                    variable_test[j].append(aux_line[3+j])
                serie_counter = serie_counter +1

            for j in range(0,variables_size):
                if numpy.mean(numpy.array(variable_test[j]).astype(numpy.float)) == 0:
                    line_saver_vector = []
            if len(line_saver_vector) != 0:
                for j in line_saver_vector:
                    newfile.write(j)
            for w in range(0,variables_size):
                variable_test[w]=[]
            line_saver_vector = []
            serie_counter =0
