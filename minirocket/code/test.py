from softmax import train, predict
# train_path = "/global/homes/s/swu264/ucr_archive/UCRArchive_2018/InsectWingbeatSound/InsectWingbeatSound_TRAIN.tsv"
# test_path = "/global/homes/s/swu264/ucr_archive/UCRArchive_2018/InsectWingbeatSound/InsectWingbeatSound_TEST.tsv"
train_path = "/global/homes/s/swu264/ucr_archive/InsectSound/InsectSound_TRAIN.csv"
test_path = "/global/homes/s/swu264/ucr_archive/InsectSound/InsectSound_TEST.csv"

model_etc = train(train_path, num_classes = 10, training_size = 22952)
# model_etc = train(train_path, num_classes = 11, training_size = 220)
# note: 22,952 = 25,000 - 2,048 (validation)

predictions, accuracy = predict(test_path, *model_etc)

print(accuracy)
print(accuracy)
print(accuracy)
print(accuracy)