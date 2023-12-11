import numpy as np
import pandas as pd
# from scipy.io.arff import loadarff 
import arff, numpy as np

target = {'Aedes_female':0, 'Aedes_male': 1, 'Fruit_flies': 2, 
          'House_flies': 3, 'Quinx_female': 4, 'Quinx_male':5,
         'Stigma_female': 6, 'Stigma_male': 7, 'Tarsalis_female': 8, 'Tarsalis_male':9}

train_path = "~/ucr_archive/InsectSound/InsectSound_TRAIN.arff"
test_path = "~/ucr_archive/InsectSound/InsectSound_TEST.arff"
csv_train_path = "~/ucr_archive/InsectSound/InsectSound_TRAIN.csv"
csv_test_path = "~/ucr_archive/InsectSound/InsectSound_TEST.csv"
stat = [0 for i in range(10)]


train_data = arff.load(open(train_path, 'r'))
train_data = np.array(train_data['data'])
print(train_data.shape)

np.random.shuffle(train_data)
for i in range(train_data.shape[0]):
    stat[target[train_data[i][-1]]] += 1
    train_data[i][-1] = target[train_data[i][-1]]
train_data = pd.DataFrame(train_data)
train_data.to_csv(csv_train_path, sep=',', index=False, header=False)

test_data = arff.load(open(test_path, 'r'))
test_data = np.array(test_data['data'])


for i in range(test_data.shape[0]):
    stat[target[test_data[i][-1]]] += 1
    test_data[i][-1] = target[test_data[i][-1]]    
test_data = pd.DataFrame(test_data)
test_data.to_csv(csv_test_path, sep=',', index=False, header=False)

print(stat)