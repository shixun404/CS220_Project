import numpy as np
import pandas as pd
import argparse
import os
import arff, numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str,
                         default="/global/cfs/cdirs/m4271/swu264/ucr_archive")
    parser.add_argument('--dataset', '-d', type=str, nargs="*")
    args = parser.parse_args()
    
    dataset_path = args.input

    dataset_list = os.listdir(dataset_path)  
    dataset_list = args.dataset if args.dataset is not None else [file for file in dataset_list if os.path.isdir(os.path.join(dataset_path, file)) ]
    print(dataset_list)
    summary_df = pd.DataFrame(index=dataset_list, columns=['n_train', 'nc_train','n_test', 'nc_test'])
    # summary_df = pd.read_csv(os.path.join(dataset_path, 'ucr_info.csv'))
    for dataset in dataset_list:
        # if dataset != "InsectSound":
        #     continue
        summary_df.to_csv(os.path.join(dataset_path, 'ucr_info.csv'))
        train_tsv_path = os.path.join(dataset_path, dataset, f"{dataset}_TRAIN.tsv")  
        test_tsv_path = os.path.join(dataset_path, dataset, f"{dataset}_TEST.tsv")  
        train_arff_path = os.path.join(dataset_path, dataset, f"{dataset}_TRAIN.arff")  
        test_arff_path = os.path.join(dataset_path, dataset, f"{dataset}_TEST.arff")  
        
        
        if os.path.exists(train_tsv_path) and os.path.exists(test_tsv_path):
            try: 
                train_data = pd.read_csv(train_tsv_path, sep='\t', index_col=False, header=None)
                test_data = pd.read_csv(test_tsv_path, sep='\t', index_col=False, header=None)
                # Calculating the number of rows and unique elements in the first column
                
                combined_labels = pd.concat([train_data.iloc[:, 0], test_data.iloc[:, 0]])
                labels, unique = pd.factorize(combined_labels)
                train_data.iloc[:, 0] = labels[:len(train_data)]
                test_data.iloc[:, 0] = labels[len(train_data):]

                num_rows = len(train_data)
                num_classes = len(unique)
                summary_df.at[dataset, 'n_train'] = num_rows
                summary_df.at[dataset, 'nc_train'] = num_classes

                # Calculating the number of rows and unique elements in the first column
                num_rows = len(test_data)
                num_classes = len(unique)

                train_data.to_csv(train_tsv_path, sep='\t', index=False, header=False)
                test_data.to_csv(test_tsv_path, sep='\t', index=False, header=False)
                summary_df.at[dataset, 'n_test'] = num_rows
                summary_df.at[dataset, 'nc_test'] = num_classes
            except Exception as e:
                pass
            continue
                


        train_data = arff.load(open(train_arff_path, 'r'))
        train_data = np.array(train_data['data'])

        np.random.shuffle(train_data)
        print(train_data.shape)
        train_data = pd.DataFrame(train_data)
        test_data = arff.load(open(test_arff_path, 'r'))
        test_data = np.array(test_data['data'])
        print(test_data.shape)
        test_data = pd.DataFrame(test_data)

        combined_labels = pd.concat([train_data.iloc[:, -1], test_data.iloc[:, -1]])
        labels, unique = pd.factorize(combined_labels)
        train_data.iloc[:, -1] = labels[:len(train_data)]
        test_data.iloc[:, -1] = labels[len(train_data):]

        # Store the last column in a variable
        last_column = train_data.iloc[:, -1]

        # Drop the last column from the DataFrame
        train_data.drop(train_data.columns[-1], axis=1, inplace=True)

        # Insert the stored column at the first position without setting a new name
        train_data.insert(0, 'Class', last_column)

        train_data.to_csv(train_tsv_path, sep='\t', index=False, header=False)


        # Store the last column in a variable
        last_column = test_data.iloc[:, -1]

        # Drop the last column from the DataFrame
        test_data.drop(test_data.columns[-1], axis=1, inplace=True)

        # Insert the stored column at the first position without setting a new name
        test_data.insert(0, 'Class', last_column)


        test_data.to_csv(test_tsv_path, sep='\t', index=False, header=False)

        # Calculating the number of rows and unique elements in the first column
        num_rows = len(train_data)
        num_classes = train_data.iloc[:, 0].nunique()
        summary_df.at[dataset, 'n_train'] = num_rows
        summary_df.at[dataset, 'nc_train'] = num_classes

        df = pd.read_csv(test_tsv_path, sep='\t')

        # Calculating the number of rows and unique elements in the first column
        num_rows = len(test_data)
        num_classes = test_data.iloc[:, 0].nunique()
        summary_df.at[dataset, 'n_test'] = num_rows
        summary_df.at[dataset, 'nc_test'] = num_classes
    summary_df.to_csv(os.path.join(dataset_path, 'ucr_info.csv'))