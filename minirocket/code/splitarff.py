
train_path = "/global/homes/s/swu264/ucr_archive/InsectSound/InsectSound_TRAIN.arff"
# train_path = "./test.arff"
# test_path = "/global/homes/s/swu264/ucr_archive/InsectSound/InsectSound_TEST.arff"
# csv_train_path = "/global/homes/s/swu264/ucr_archive/InsectSound/InsectSound_TRAIN.csv"
# csv_test_path = "/global/homes/s/swu264/ucr_archive/InsectSound/InsectSound_TEST.csv"


test = []
with open(train_path, "r") as file:
    for i in range(610):
        line = file.readline()
        test.append(line)

# Save these lines into a new file
new_file_path = "./test.arff"
with open(new_file_path, "w") as new_file:
    new_file.writelines(test)