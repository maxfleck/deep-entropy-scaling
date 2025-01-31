import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn
import torch
import json

from torch.utils.data import Dataset


def minmax_to_dict(Scaler):
    minmax_dict = {}
    minmax_dict["min_"] = list(Scaler.min_)
    minmax_dict["scale_"] = list(Scaler.scale_)
    minmax_dict["data_min_"] = list(Scaler.data_min_)
    minmax_dict["data_max_"] = list(Scaler.data_max_)
    minmax_dict["data_range_"] = list(Scaler.data_range_)
    minmax_dict["n_features_in_"] = Scaler.n_features_in_
    minmax_dict["n_samples_seen_"] = Scaler.n_samples_seen_
    # minmax_dict["feature_names_in_"] = Scaler.feature_names_in
    return minmax_dict


def minmax_to_json(Scaler, filename):
    minmax_dict = minmax_to_dict(Scaler)
    with open(filename, "w") as outfile:
        json.dump(minmax_dict, outfile,
                  indent=4, sort_keys=False)
    return


def minmax_from_dict(minmax_dict):
    Scaler = sklearn.preprocessing.MinMaxScaler()
    a = np.atleast_2d(minmax_dict["data_min_"])
    b = np.atleast_2d(minmax_dict["data_max_"])
    c = np.concatenate((a, b))
    return Scaler.fit(c)


def minmax_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return minmax_from_dict(data)


class dataset(Dataset):
    """
    - standard dataset for final training
    - returns random batches
    """
    # Constructor with defult values
    def __init__(self, data_csv, log_transform=True,
                 x_features=["temperature", "resd_entropy", "molarweight",
                             "m", "sigma", "epsilon_k", "kappa_ab",
                             "epsilon_k_ab", "mu"],
                 y_features=["log_value"],
                 keep_features=["iupac_name", "cas", "family"],
                 ScalerX=sklearn.preprocessing.MinMaxScaler(),
                 ScalerY=sklearn.preprocessing.MinMaxScaler(),
                 scaler_fit=False, scalerX=None, scalerY=None,
                 keepXY=False,
                 ):
        self.x_features = x_features
        self.y_features = y_features
        self.keep_features = keep_features
        self.data_csv = data_csv
        data = pd.read_csv(data_csv)
        # print("data.keys",data.keys())
        data["resd_entropy"] = np.abs(data["resd_entropy"])
        # print(data.shape)
        if log_transform:
            data = data[data["pressure"] > 0]
            data["log_pressure"] = np.log(data["pressure"])
            data["log_value"] = np.log(data["value"])
        # print(data.shape)

        X = np.array(data[x_features])
        Y = np.array(data[y_features])
        if keepXY:
            self.X = X
            self.Y = Y
        self.keep = data[keep_features]
        if "cas" in self.keep:
            self.encoded_species = LabelEncoder().fit_transform(data["cas"])
            self.n_species = np.unique(self.encoded_species).shape[0]
        elif "iupac_name" in self.keep:
            self.encoded_species = LabelEncoder().fit_transform(data["iupac_name"])
            self.n_species = np.unique(self.encoded_species).shape[0]            
        if "family" in self.keep:
            self.encoded_families = LabelEncoder().fit_transform(data["family"])
            self.n_families = np.unique(self.encoded_families).shape[0]
        del data

        # norm data here
        if scaler_fit:
            self.ScalerX = ScalerX
            self.ScalerY = ScalerY
            self.scalerX = self.ScalerX.fit(X)
            self.scalerY = self.ScalerY.fit(Y)
        else:
            self.ScalerX = "external"
            self.ScalerY = "external"
            self.scalerX = scalerX
            self.scalerY = scalerY

        self.X_scaled = torch.Tensor(self.scalerX.transform(X))
        self.Y_scaled = torch.Tensor(self.scalerY.transform(Y))

        self.len = self.Y_scaled.shape[0]

        self.species_indexes = []
        for n in np.unique(self.encoded_species):
            p = np.where(self.encoded_species == n)
            self.species_indexes.append(np.squeeze(p))    

        self.families_indexes = []
        for n in np.unique(self.encoded_families):
            p = np.where(self.encoded_families == n)
            self.families_indexes.append(np.squeeze(p))            
        return

    # Getter
    def __getitem__(self, index):
        sample = self.X_scaled[index].float(), self.Y_scaled[index].float()
        return sample

    # Get Length
    def __len__(self):
        return self.len

    def get_keep(self, index):
        return self.keep.iloc[index]

    def get_data(self):
        return self.X_scaled.float(), self.Y_scaled.float()

    # Getter
    def get_species(self, index):
        # print( index )
        p = self.species_indexes[index]
        # print( p.shape )
        sample = self.X_scaled[p].float(), self.Y_scaled[p].float()
        # print( sample[0].size(), sample[1].size() )
        return sample

    def get_species_keep(self, index):
        return self.keep.iloc[self.species_indexes[index]]
