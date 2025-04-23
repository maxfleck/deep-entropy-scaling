import numpy as np

import copy
import shutil
import os
import tempfile
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ray.train import Checkpoint
from ray import train

from deep_entropy_scaling.nn_model import model0_a, model0_ci, model0_ci_poly, model0_ci_poly2
from deep_entropy_scaling.nn_model import model0_ci_norm, model0_ci_enorm, model0_ci_norm2
from deep_entropy_scaling.nn_model import model0_ci_n2pe, model0_ci_p2n2
from deep_entropy_scaling.nn_dataset import dataset, minmax_from_json


def freeze_children(model, freeze):
    for name, child in model.named_children():
        if name in freeze:
            for param in child.parameters():
                param.requires_grad = False
    return


def unfreeze_children(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
    return


def check_freeze_children(model):
    for name, child in model.named_children():
        for param in child.parameters():
            if param.requires_grad:
                print(name, ": not freezed, grad = ", param.requires_grad)
            else:
                print(name, ": freezed, grad = ", param.requires_grad)
    return


def load_model0(config):
    # print(config)
    embedding_size = config["embedding_size"]
    entropy_feature_size = config["n_entropy_features"]
    ref_feature_size = config["n_features_ref"]

    if  config["build"] == "model0_ci":
        print("load model0_ci:", config["build"])
        model = model0_ci(embedding_size, entropy_feature_size,
                          ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        a = [int(config["n_nodes_entropy_feature"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)
    elif  config["build"] == "model0_ci_norm":
        print("load model0_ci_norm:", config["build"])
        model = model0_ci_norm(embedding_size, entropy_feature_size,
                          ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        a = [int(config["n_nodes_entropy_feature"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)    
    elif  config["build"] == "model0_ci_enorm":
        print("load model0_ci_enorm:", config["build"])
        model = model0_ci_enorm(embedding_size, entropy_feature_size,
                          ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        a = [int(config["n_nodes_entropy_feature"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)            
    elif  config["build"] == "model0_ci_norm2":
        print("load model0_ci_norm2:", config["build"])
        model = model0_ci_norm2(embedding_size, entropy_feature_size,
                          ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        a = [int(config["n_nodes_entropy_feature"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)  
    elif  config["build"] == "model0_ci_n2":
        print("load model0_ci_n2:", config["build"])
        model = model0_ci_norm2(embedding_size, entropy_feature_size,
                          ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        a = [int(config["n_nodes_entropy_feature"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)          
    elif config["build"] == "model0_ci_poly":
        print("load model0_ci_poly:", config["build"])
        model = model0_ci_poly(embedding_size, entropy_feature_size,
                               ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        # a = [int(config["n_nodes_entropy_feature"])]
        # b = int(config["n_layers_parameter"])
        # layers = a*b
        # model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)
    elif config["build"] == "model0_ci_n2pe":
        print("load model0_ci_n2pe:", config["build"])
        model = model0_ci_n2pe(embedding_size, entropy_feature_size,
                               ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        # a = [int(config["n_nodes_entropy_feature"])]
        # b = int(config["n_layers_parameter"])
        # layers = a*b
        # model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)        
    elif config["build"] == "model0_ci_poly_eta":
        print("load model0_ci_poly:", config["build"])
        model = model0_ci_poly(embedding_size, entropy_feature_size,
                               ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        # a = [int(config["n_nodes_entropy_feature"])]
        # b = int(config["n_layers_parameter"])
        # layers = a*b
        # model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)
    elif config["build"] == "model0_ci_poly2":
        print("load model0_ci_poly2:", config["build"])
        model = model0_ci_poly2(embedding_size, entropy_feature_size,
                                ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        # a = [int(config["n_nodes_feature_ref"])]
        # b = int(config["n_layers_ref"])
        # layers = a*b
        # model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        # a = [int(config["n_nodes_entropy_feature"])]
        # b = int(config["n_layers_parameter"])
        # layers = a*b
        # model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)
    elif config["build"] ==  "model0_ci_p2n2":
        print("load model0_ci_p2n2:", config["build"])
        model = model0_ci_p2n2(embedding_size, entropy_feature_size,
                                ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        # a = [int(config["n_nodes_feature_ref"])]
        # b = int(config["n_layers_ref"])
        # layers = a*b
        # model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        # a = [int(config["n_nodes_entropy_feature"])]
        # b = int(config["n_layers_parameter"])
        # layers = a*b
        # model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)        
    elif config["build"] == "model0_ci_poly2_eta":
        print("load model0_ci_poly2:", config["build"])
        model = model0_ci_poly2(embedding_size, entropy_feature_size,
                                ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        # a = [int(config["n_nodes_feature_ref"])]
        # b = int(config["n_layers_ref"])
        # layers = a*b
        # model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        # a = [int(config["n_nodes_entropy_feature"])]
        # b = int(config["n_layers_parameter"])
        # layers = a*b
        # model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)
    else:
        print("load model0_a:", config["build"])
        model = model0_a(embedding_size, entropy_feature_size,
                         ref_feature_size)
        # if "n_nodes_feature_ref" in config.keys():
        a = [int(config["n_nodes_feature_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_feature_net(layers)
        # if "n_nodes_ref" in config.keys():
        a = [int(config["n_nodes_ref"])]
        b = int(config["n_layers_ref"])
        layers = a*b
        model.build_ref_parameter_net(layers)
        # if "n_nodes_entropy_feature" in config.keys():
        a = [int(config["n_nodes_entropy_feature"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_entropy_feature_net(layers)
        # if "n_nodes_parameter" in config.keys():
        a = [int(config["n_nodes_parameter"])]
        b = int(config["n_layers_parameter"])
        layers = a*b
        model.build_parameter_net(layers)

    if "checkpoint_path" in config.keys() and config["checkpoint_path"]:
        try:
            model_state, _ = torch.load(config["checkpoint_path"],
                                        weights_only=True)
        except ValueError:
            model_state = torch.load(config["checkpoint_path"],
                                     weights_only=True)
        print("load1")
    else:
        try:
            model_state, _ = torch.load(config["state_dict_path"],
                                        weights_only=True)
        except ValueError:
            model_state = torch.load(config["state_dict_path"],
                                     weights_only=True)        
        print("load0")
    model.load_state_dict(model_state)
    return model


def train_loop(config_train, config, epochs=1, freeze=[],
               n_restart=1, patience=20, report_result=True):

    # data_train = config["data_train"]
    # data_val = config["data_val"]

    y_features = ["log_value"]
    # scalerX = joblib.load(config["model_path"]+'Xscaler.gz')
    # scalerY = joblib.load(config["model_path"]+'Yscaler.gz')
    scalerX = minmax_from_json(config["model_path"]+'Xscaler.json')
    scalerY = minmax_from_json(config["model_path"]+'Yscaler.json')
    data_train = dataset(config["data_csv_train"], y_features=y_features,
                         scalerX=scalerX, scalerY=scalerY)
    data_val = dataset(config["data_csv_val"], y_features=y_features,
                       scalerX=scalerX, scalerY=scalerY)
    # config["data_train"] = data_train
    # config["data_val"] = data_val

    dataloader = DataLoader(data_train, config_train["batch_size"],
                            shuffle=True, pin_memory=False)
    # Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_loss = 1e6
    best_l1 = 1e6
    best_val_loss = 1e6
    best_val_l1 = 1e6
    i_re_best = 0

    model = load_model0({**config_train, **config,
                        "n_entropy_features": config_train["n_features"]})
    # ini weights
    best_model_weights = copy.deepcopy(model.state_dict())

    # re_losses = []
    # re_l1s = []
    # re_val_losses = []
    # re_val_l1s = []
    n_patience = patience

    for i_re in np.arange(n_restart):

        model = load_model0({**config_train, **config,
                             "n_entropy_features": config_train["n_features"]})

        # device = "cpu"
        # if torch.cuda.is_available():
        #     device = "cuda:0"
        #     if torch.cuda.device_count() > 1:
        #         model = nn.DataParallel(model)
        # model.to(device)

        loss_fn = nn.MSELoss()
        mead_fn = nn.L1Loss()
        # optimizer = optim.SGD(model.parameters(), lr=0.001,
        # momentum=0.9, weight_decay=1e-7)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config_train["lr"], weight_decay=1e-7)

        # losses = []
        # l1s = []
        # val_losses = []
        # val_l1s = []

        for epoch in range(epochs):  # loop over the dataset multiple times
            # print(epoch)
            for X_train_scaled, y_train_scaled in dataloader:

                y_pred = model(X_train_scaled)
                loss = loss_fn(y_pred, y_train_scaled)

                # Zero the gradients before running the backward pass.
                model.zero_grad()

                # Backward pass: compute gradient of the loss with respect
                # to all the learnable parameters of the model.
                # Internally, the parameters of each Module are stored
                # in Tensors with requires_grad=True, so this call will
                # compute gradients for all learnable parameters in the model.
                loss.backward()

                # Update the weights using gradient descent.
                # Each parameter is a Tensor, so we can access
                # its gradients like we did before.
                # with torch.no_grad():
                #    for param in model.parameters():
                #        param -= learning_rate * param.grad
                optimizer.step()

                #print("y_pred.shape, y_train_scaled.shape")
                #print(y_pred.shape, y_train_scaled.shape)
            y_pred = model(data_train.X_scaled)
            loss = loss_fn(y_pred, data_train.Y_scaled)
            l1 = mead_fn(y_pred, data_train.Y_scaled)

            y_pred_val = model(data_val.X_scaled)
            val_loss = loss_fn(y_pred_val, data_val.Y_scaled)
            val_l1 = mead_fn(y_pred_val, data_val.Y_scaled)

            # losses.append(loss)
            # l1s.append(l1)
            # val_losses.append(val_loss)
            # val_l1s.append(val_l1)

            # Early stopping
            if val_loss < best_val_loss:
                best_loss = loss
                best_l1 = l1
                best_val_loss = val_loss
                best_val_l1 = val_l1
                best_model_weights = copy.deepcopy(model.state_dict())
                n_patience = patience  # Reset patience counter
                i_re_best = i_re

            elif epoch >= patience:
                n_patience -= 1
                if n_patience == 0:
                    print("early termination")
                    break
            # print("")
            # gc.collect()

        # re_losses.append(losses)
        # re_l1s.append(l1s)
        # re_val_losses.append(val_losses)
        # re_val_l1s.append(val_l1s)
        print("fin RE")

    model.load_state_dict(best_model_weights)
    # if report_result:
    checkpoint_name = "checkpoint.pt"
    train_report = {"val_loss": float(best_val_loss),
                    "val_l1": float(best_val_l1),
                    "train_loss": float(best_loss), "train_l1": float(best_l1),
                    "epoch": epoch, "i_re_best": i_re_best}
    if report_result:
        with tempfile.TemporaryDirectory() as tcd:
            path = os.path.join(tcd, checkpoint_name)
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(tcd)

            train.report(
                train_report,
                checkpoint=checkpoint,
            )
        print("Finished Training", epoch)
        return model
    else:
        print("Finished Training", epoch)
        return model, train_report


def train_looper(config_train, config):

    config["checkpoint_path"] = ""

    # train all
    freeze = []
    train_loop(config_train, config, epochs=100,
               patience=20, n_restart=1, freeze=freeze,
               report_result=True)
    return


def best_model(config, exp_error=True):

    config["checkpoint_path_native"] = config["checkpoint_path"]
    cpath = os.path.join(config["model_path"], "checkpoint.pt")
    config["checkpoint_path"] = cpath
    try:
        shutil.copyfile(config["checkpoint_path_native"], cpath)
    except shutil.SameFileError:
        print("checkpoint copy failed (probably already the same file.)")

    data_csv_train = config["data_csv_train"]
    data_csv_val = config["data_csv_val"]
    data_csv_test = config["data_csv_test"]

    y_features = ["log_value"]
    # scalerX = joblib.load(config["model_path"]+'Xscaler.gz')
    # scalerY = joblib.load(config["model_path"]+'Yscaler.gz')
    scalerX = minmax_from_json(config["model_path"]+'Xscaler.json')
    scalerY = minmax_from_json(config["model_path"]+'Yscaler.json')
    data_train = dataset(data_csv_train, y_features=y_features,
                         scalerX=scalerX, scalerY=scalerY)
    data_val = dataset(data_csv_val, y_features=y_features,
                       scalerX=scalerX, scalerY=scalerY)
    data_test = dataset(data_csv_test, y_features=y_features,
                        scalerX=scalerX, scalerY=scalerY)

    model = load_model0(config)

    a = ["training", "validation", "test"]
    b = [data_train, data_val, data_test]
    for dd, data in zip(a, b):
        for ii in range(data.n_species):
            X, Y = data.get_species(ii)
            if len(Y.shape) > 1:
                # print(Y.shape)
                log_vis = Y[:, 0]
                resd_entropy = X[:, 1].detach().numpy()

                y_pred0 = torch.squeeze(model.forward_ref(X))
                # print( y_pred0.shape, log_vis.shape )

                log_vis_star = log_vis - y_pred0
                log_vis_star = torch.squeeze(log_vis_star).detach().numpy()
                # print( log_vis_star.shape, resd_entropy.shape )

                deg = int(config["n_features"])
                z = np.polyfit(resd_entropy, log_vis_star, deg)
                yy = np.poly1d(z)(resd_entropy)
                y_pred = torch.Tensor(yy)

                if ii == 0:
                    plt.plot(resd_entropy, log_vis_star, "kx",
                             label="experimental")
                else:
                    plt.plot(resd_entropy, log_vis_star, "kx")
                info = data.get_species_keep(ii)
                name = info["iupac_name"].iloc[0]
                plt.plot(resd_entropy, y_pred.detach().numpy(), ".",
                         label=str(name))
        plt.title(dd)
        plt.legend(bbox_to_anchor=(1, 1.1))
        plt.savefig(config["model_path"]+dd+"_noise.png", bbox_inches="tight")
        plt.savefig(config["model_path"]+dd+"_noise.pdf", bbox_inches="tight")
        plt.show()
        plt.close()

    losses = {"training": 0, "validation": 0, "test": 0}
    l1s = {"training": 0, "validation": 0, "test": 0}
    loss_fn = nn.MSELoss()
    mead_fn = nn.L1Loss()

    a = ["training", "validation", "test"]
    b = [data_train, data_val, data_test]
    for dd, data in zip(a, b):
        for ii in range(data.n_species):
            X, Y = data.get_species(ii)
            if len(Y.shape) > 1:
                # Y = scalerY.inverse_transform(Y)
                resd_entropy = np.squeeze(X[:, 1].detach().numpy())

                # y_pred = scalerY.inverse_transform(model(X))
                y_pred = model(X)
                losses[dd] += loss_fn(y_pred, Y).detach().numpy()
                l1s[dd] += mead_fn(y_pred, Y).detach().numpy()
                # resd_entropy *= m

                Y = scalerY.inverse_transform(Y.detach().numpy())
                y_pred = scalerY.inverse_transform(y_pred.detach().numpy())

                yy = np.squeeze(Y)
                y_pred = np.squeeze(y_pred)
                # print( resd_entropy.shape, yy.shape, y_pred.shape )
                if ii == 0:
                    plt.plot(resd_entropy, yy, "kx", label="experimental")
                else:
                    plt.plot(resd_entropy, yy, "kx")
                if exp_error:
                    error = np.abs((np.exp(y_pred) - np.exp(yy)) / np.exp(yy))
                    error = np.mean(error)
                else:
                    error = np.mean(np.abs((y_pred - yy) / yy))
                # print(error)
                info = data.get_species_keep(ii)
                name = info["iupac_name"].iloc[0]
                plt.plot(resd_entropy, y_pred, ".",
                         label=str(name)+" "+str(round(error*100, 2))+"%"
                         )
        plt.title(dd)
        plt.legend(bbox_to_anchor=(1, 1.1))
        plt.savefig(config["model_path"]+dd+".png", bbox_inches="tight")
        plt.savefig(config["model_path"]+dd+".pdf", bbox_inches="tight")
        plt.show()
        plt.close()

    config = {**config, **{"loss": losses, "l1": l1s}}
    fname = os.path.join(config["model_path"], "best_result_config.json")
    with open(fname, "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=False, default=str)

    return model, data_train
