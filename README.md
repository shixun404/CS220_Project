#CS220_Project

## Command:

### MiniRocket

```
cd minirocket/code
```

- arff file to csv file
```
python arff2csv.py 
```

- Train and test minirocket pytorch using InsectSound Dataset
```
python test.py 
```

### NeuroSim

- NeuroSim command

```
python inference.py --dataset InsectSound --model Rocket --mode WAGE --inference 1 --cellBit 1  --subArray 32 --parallelRead 32 
```

- NeuroSim only supports power of 2
- Modify network.csv, param.csv and compile.


### UCR Dataset

- Feasible Dataset, other dataset is not available due to small training size.

```
InsectSound finished, accuracy=0.75544                                          
FordA finished, accuracy=0.9424242424242424                                     
ElectricDevices finished, accuracy=0.6038127350538193                           
FruitFlies finished, accuracy=0.9546903065067501                                
Crop finished, accuracy=0.34809523809523807                                     
```

- Open `ucr_info.csv` to check UCR Time Seires Dataset.