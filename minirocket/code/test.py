from softmax import train, predict
import torch
import os.path
import argparse
import pickle as pkl
import pandas as pd





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str,
                         default="/global/homes/s/swu264/ucr_archive/")
    parser.add_argument('--output', '-o', type=str, 
                        default="/global/homes/s/swu264/rocket/DNN_NeuroSim_V1.4/Inference_pytorch/log/")
    parser.add_argument('--features', '-f', type=int, nargs="*", default=[8192])
    args = parser.parse_args()
    
    dataset_path = args.input
    output_path = args.output
    features_list = args.features
    dataset_list = os.listdir(dataset_path)  
    dataset_list = [file for file in dataset_list]

    accuracy_df = pd.DataFrame(index=dataset_list, columns=features_list)

    for dataset in dataset_list:
        for feature in features_list:
            train_path = os.path.join(dataset_path, dataset, f"{dataset}_TRAIN.csv")
            test_path = os.path.join(dataset_path, dataset, f"{dataset}_TEST.csv")
            save_path = os.path.join(output_path, dataset, f"{feature}")

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            parameters, model, f_mean, f_std  = train(train_path, num_classes=10,
                                                       training_size=22952, num_features=feature)


            torch.save(model.state_dict(), os.path.join(save_path, f'Rocket_{dataset}.pth'))
            with open(os.path.join(save_path, f'Rocket_parameters_{dataset}.pkl'), 'wb') as f:
                pkl.dump((parameters, f_mean, f_std), f)

            predictions, accuracy = predict(test_path, parameters, model, f_mean, f_std, num_features=feature)

            accuracy_df.at[dataset, feature] = accuracy
    accuracy_df.to_csv(os.path.join(output_path, 'rocket_ucr_accuracy.csv'), index=True)
    print(accuracy_df)