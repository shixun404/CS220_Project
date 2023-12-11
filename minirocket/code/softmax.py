# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification

# https://arxiv.org/abs/2012.08791

import copy
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim

from minirocket import fit, transform


class Rocket(nn.Module):
    def __init__(self, num_classes=10, num_features=84 * (10_000 // 84)):
        super(Rocket, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(num_features, num_classes, bias=False))
        
    def forward(self, x):
        out = self.classifier(x)
        return out

def train(path, num_classes, training_size, num_features, **kwargs):
    # torch.cuda.set_device('cuda:0')
    # print(torch.cuda.current_device())
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # assert 0
    # print(torch.cuda.is_available() )
    # assert 0
    # torch.cuda.set_device
    # -- init ------------------------------------------------------------------

    # default hyperparameters are reusable for any dataset
    args = \
    {
        "num_features"    : 10_000,
        "validation_size" : 2 ** 11,
        "chunk_size"      : 2 ** 12,
        "minibatch_size"  : 256,
        "lr"              : 1e-4,
        "max_epochs"      : 50,
        "patience_lr"     : 5,  #  50 minibatches
        "patience"        : 10, # 100 minibatches
        "cache_size"      : training_size  # set to 0 to prevent caching
    }
    args = {**args, **kwargs}

    # _num_features = 84 * (args["num_features"] // 84)
    _num_features = num_features
    num_chunks = np.int32(np.ceil(training_size / args["chunk_size"]))

    def init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight.data, 0)
            # nn.init.constant_(layer.bias.data, 0)

    # -- cache -----------------------------------------------------------------

    # cache as much as possible to avoid unecessarily repeating the transform
    # consider caching to disk if appropriate, along the lines of numpy.memmap

    cache_X = torch.zeros((args["cache_size"], _num_features))
    cache_Y = torch.zeros(args["cache_size"], dtype = torch.long)
    cache_count = 0
    fully_cached = False

    # -- model -----------------------------------------------------------------
    
    model = Rocket(num_classes=num_classes, num_features=_num_features)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # assert 0
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, min_lr = 1e-8, patience = args["patience_lr"])
    model.apply(init)

    # -- validation data -------------------------------------------------------

    # gotcha: copy() is essential to avoid competition for memory access with read_csv(...)
    validation_data = pd.read_csv(path,
                                  header = None,
                                  sep = ",",
                                  nrows = args["validation_size"],
                                  engine = "c").values.copy()
    Y_validation, X_validation = torch.LongTensor(validation_data[:, -1]), validation_data[:, :-1].astype(np.float32)
    # print(Y_validation[0], X_validation[0])
    # assert 0
    # -- run -------------------------------------------------------------------

    minibatch_count = 0
    best_validation_loss = np.inf
    stall_count = 0
    stop = False

    print("Training... (faster once caching is finished)")

    for epoch in range(args["max_epochs"]):

        if epoch > 0 and stop:
            break

        if not fully_cached:
            file = pd.read_csv(path,
                               header = None,
                               sep = ",",
                               skiprows = args["validation_size"],
                               chunksize = args["chunk_size"],
                               engine = "c")

        for chunk_index in range(num_chunks):

            a = chunk_index * args["chunk_size"]
            b = min(a + args["chunk_size"], training_size)
            _b = b - a

            if epoch > 0 and stop:
                break

            print(f"Epoch {epoch + 1}; Chunk = {chunk_index + 1}...".ljust(80, " "), end = "\r", flush = True)

            # if not fully cached, read next file chunk
            if not fully_cached:

                # gotcha: copy() is essential to avoid competition for memory access with read_csv(...)
                training_data = file.get_chunk().values[:_b].copy()
                Y_training, X_training = torch.LongTensor(training_data[:, -1]), training_data[:, :-1].astype(np.float32)
                
                if epoch == 0 and chunk_index == 0:

                    parameters = fit(X_training, args["num_features"])

                    # transform validation data
                    X_validation_transform = transform(X_validation, parameters)

            # if cached, retrieve from cache
            if b <= cache_count:

                X_training_transform = cache_X[a:b]
                Y_training = cache_Y[a:b]

            # else, transform and cache
            else:

                # transform training data
                X_training_transform = transform(X_training, parameters)

                if epoch == 0 and chunk_index == 0:

                    # per-feature mean and standard deviation
                    f_mean = X_training_transform.mean(0)
                    f_std = X_training_transform.std(0) + 1e-8

                    # normalise validation features
                    X_validation_transform = (X_validation_transform - f_mean) / f_std
                    X_validation_transform = torch.FloatTensor(X_validation_transform)[:, :8192]

                # normalise training features
                X_training_transform = (X_training_transform - f_mean) / f_std
                X_training_transform = torch.FloatTensor(X_training_transform)[:, :8192]

                # cache as much of the transform as possible
                if b <= args["cache_size"]:
                    # print(X_training_transform.shape)
                    # print(cache_X[a:b].shape, X_training_transform[:b-a].shape)
                    s = min(b - a, X_training_transform.shape[0])
                    # print(X_training_transform.shape, cache_X.shape)
                    # assert 0
                    cache_X[a:a + s] = X_training_transform[:s]
                    cache_Y[a:a + s] = Y_training[:s]
                    cache_count = a + s

                    if cache_count >= training_size:
                        fully_cached = True

            minibatches = torch.randperm(len(X_training_transform)).split(args["minibatch_size"])

            # train on transformed features
            for minibatch_index, minibatch in enumerate(minibatches):

                if epoch > 0 and stop:
                    break

                if minibatch_index > 0 and len(minibatch) < args["minibatch_size"]:
                    break

                # -- training --------------------------------------------------

                optimizer.zero_grad()
                _Y_training = model(X_training_transform[minibatch])
                training_loss = loss_function(_Y_training, Y_training[minibatch])
                training_loss.backward()
                optimizer.step()
                # print(training_loss)

                minibatch_count += 1

                if minibatch_count % 10 == 0:

                    _Y_validation = model(X_validation_transform)
                    validation_loss = loss_function(_Y_validation, Y_validation)

                    scheduler.step(validation_loss)

                    if validation_loss.item() >= best_validation_loss:
                        stall_count += 1
                        if stall_count >= args["patience"]:
                            stop = True
                            print(f"\n<Stopped at Epoch {epoch + 1}>")
                    else:
                        best_validation_loss = validation_loss.item()
                        best_model = copy.deepcopy(model)
                        if not stop:
                            stall_count = 0

    return parameters, best_model, f_mean, f_std

def predict(path,
            parameters,
            model,
            f_mean,
            f_std,
            **kwargs):
    # torch.cuda.set_device('cuda:0')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args = \
    {
        "score"      : True,
        "chunk_size" : 2 ** 12,
        "test_size"  : None
    }
    args = {**args, **kwargs}

    file = pd.read_csv(path,
                       header = None,
                       sep = ",",
                       chunksize = args["chunk_size"],
                       nrows = args["test_size"],
                       engine = "c")

    predictions = []

    correct = 0
    total = 0

    for chunk_index, chunk in enumerate(file):

        print(f"Chunk = {chunk_index + 1}...".ljust(80, " "), end = "\r")

        # gotcha: copy() is essential to avoid competition for memory access with read_csv(...)
        test_data = chunk.values.copy()
        Y_test, X_test = test_data[:, -1], test_data[:, :-1].astype(np.float32)
        # print(test_data[0])
        # assert 0
        # print(test_data[:, :-1].shape)
        # assert 0
        X_test_transform = transform(X_test, parameters)
        X_test_transform = (X_test_transform - f_mean) / f_std
        X_test_transform = torch.FloatTensor(X_test_transform)[:, :8192]

        _predictions = model(X_test_transform).argmax(1).numpy()
        predictions.append(_predictions)
        # print(_predictions, Y_test)

        total += len(test_data)
        correct += (_predictions == Y_test).sum()

    if args["score"]:
        return np.concatenate(predictions), correct / total
    else:
        return np.concatenate(predictions)
