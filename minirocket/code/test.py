from softmax import train, predict
import torch
import os.path
import argparse
import pickle as pkl
import pandas as pd





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str,
                         default="/global/cfs/cdirs/m4271/swu264/ucr_archive/")
    parser.add_argument('--output', '-o', type=str, 
                        default="/global/homes/s/swu264/rocket/DNN_NeuroSim_V1.4/Inference_pytorch/log/")
    parser.add_argument('--features', '-f', type=int, nargs="*", default=[8192])
    args = parser.parse_args()
    
    dataset_path = args.input
    output_path = args.output
    features_list = args.features
    dataset_list = os.listdir(dataset_path)  
    dataset_list = [file for file in dataset_list]

    
    # if os.path.exists(os.path.join(output_path, 'rocket_ucr_accuracy.csv')):
    #     accuracy_df = pd.read_csv(os.path.join(output_path, 'rocket_ucr_accuracy.csv'))
    # else:
    accuracy_df = pd.DataFrame(index=dataset_list, columns=features_list)
    ucr_df = pd.read_csv(os.path.join(dataset_path, "ucr_info.csv"))
    ucr_df.set_index('Unnamed: 0', inplace=True)
    for dataset in dataset_list:
        # if dataset != "InsectSound":
        #     continue
        for feature in features_list:
            accuracy_df.to_csv(os.path.join(output_path, 'rocket_ucr_accuracy.'), index=True)
            train_path = os.path.join(dataset_path, dataset, f"{dataset}_TRAIN.tsv")
            test_path = os.path.join(dataset_path, dataset, f"{dataset}_TEST.tsv")
            save_path = os.path.join(output_path, dataset, f"{feature}")
            
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            try:
                parameters, model, f_mean, f_std  = train(train_path, num_classes=int(ucr_df.at[dataset, 'nc_train']),
                                                        training_size=int(ucr_df.at[dataset, 'n_train']) - 2 ** 11, num_features=feature)
            except Exception as e:
                accuracy_df.at[dataset, feature] = None
                print(f"{dataset} failed")
                continue

            if parameters == 0:
                accuracy_df.at[dataset, feature] = None
                print(f"{dataset} failed")
                continue
            
            torch.save(model.state_dict(), os.path.join(save_path, f'Rocket_{dataset}.pth'))
            with open(os.path.join(save_path, f'Rocket_parameters_{dataset}.pkl'), 'wb') as f:
                pkl.dump((parameters, f_mean, f_std), f)

            predictions, accuracy = predict(test_path, parameters, model, f_mean, f_std, num_features=feature)
            print(f"{dataset} finished, accuracy={accuracy}")
            accuracy_df.at[dataset, feature] = accuracy
    accuracy_df.to_csv(os.path.join(output_path, 'rocket_ucr_accuracy.csv'), index=True)
    print(accuracy_df.at['InsectSound', feature])