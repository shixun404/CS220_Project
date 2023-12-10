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
