import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feos.si import * # SI numbers and constants
#from si_units import * # SI numbers and constants
from feos.pcsaft import *
from feos.eos import *
import feos

import os
#import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from filelock import FileLock
from torch.utils.data import random_split,DataLoader, TensorDataset
#from typing import Dict
#from ray import train, tune
#from ray.train import Checkpoint
#from ray.tune.schedulers import ASHAScheduler


def collision_integral( T, p):
    """
    computes analytical solution of the collision integral

    T: reduced temperature
    p: parameters

    returns analytical solution of the collision integral
    """
    A,B,C,D,E,F,G,H,R,S,W,P = p
    return A/T**B + C/np.exp(D*T) + E/np.exp(F*T) + G/np.exp(H*T) + R*T**B*np.sin(S*T**W - P)

def get_omega22(red_temperature):
    """
    computes analytical solution of the omega22 collision integral

    red_temperature: reduced temperature

    returns omega22
    """
    p22 = [ 
         1.16145,0.14874,0.52487,
         0.77320,2.16178,2.43787,
         0.0,0.0,-6.435/10**4,
         18.0323,-0.76830,7.27371
        ]
    return collision_integral(red_temperature,p22)

def get_CE_viscosity_reference_new( data ):
    """
    computes Chapman-Enskog viscosity reference for an array of temperatures
    uses pc-saft parameters

    temperature: array of temperatures
    saft_parameters: pc saft parameter object build with feos

    returns reference
    """
    epsilon = data["epsilon_k"]*KELVIN
    sigma   = data["sigma"]*ANGSTROM
    m       = data["m"]
    M       = data["molarweight"]*GRAM/MOL
    temperature = data["temperature"]*KELVIN
    density     = data["molar_density"]*MOL/METER**3
    red_temperature = temperature/epsilon
    red_density     = density*sigma**3*NAV

    omega22 = get_omega22(red_temperature)

    sigma2 = sigma**2
    M_SI = M

    sq1  = np.sqrt( M_SI * KB * temperature / NAV /np.pi /METER**2 / KILOGRAM**2 *SECOND**2 ) *METER*KILOGRAM/SECOND
    div1 = omega22 * sigma2 * m
    viscosity_reference = 5/16* sq1 / div1 #*PASCAL*SECOND
    
    data["red_temperature"] = red_temperature
    data["red_density"] = red_density
    data["ln_eta_ref_new"] = np.log( viscosity_reference /PASCAL/SECOND )
    return data

def get_entropy(sub):
    
    M = sub["molarweight"]*(GRAM/MOL)
    m = sub["m"]
    
    identifier = Identifier( cas="000-00-0", name="dummy" )
    saftrec = PcSaftRecord( m=sub["m"], sigma=sub["sigma"], epsilon_k=sub["epsilon_k"],
                          kappa_ab=sub["kappa_ab"],epsilon_k_ab=sub["epsilon_k_ab"],mu=sub["mu"] )

    pr = PureRecord(identifier=identifier,model_record=saftrec,molarweight=sub["molarweight"])

    saft_paras = PcSaftParameters.from_records( [pr], np.array([[0.0]] ) )

    eos = EquationOfState.pcsaft(saft_paras)
    try:
        if "liquid" in sub["state"]:
            state = State(eos,temperature=sub["temperature"]*KELVIN, 
                          pressure=sub["pressure"]*PASCAL,
                          density_initialization="liquid"
                         )
        else:
            state = State(eos,temperature=sub["temperature"]*KELVIN, 
                          pressure=sub["pressure"]*PASCAL,
                          density_initialization="vapor"
                         )
        sub["resd_entropy"]      = -state.specific_entropy(Contributions.ResidualNvt)/ KB /NAV *M
        sub["red_resd_entropy"]  = sub["resd_entropy"]/m
        sub["molar_density"]          = state.density / MOL * METER**3            
    except:
        sub["resd_entropy"]  = -1
        sub["red_resd_entropy"]  = -1
        sub["molar_density"] = -1

    return sub