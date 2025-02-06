# deep-entropy-scaling

IF YOURE LOOKING FOR THE GITHUB OF "Physics-informed multi-fidelity Gaussian process: modeling the effect of water and temperature on the viscosity of a deep eutectic solvent":

[### PLEASE GO HERE ###](https://github.com/maxfleck/DES_multifidelity_eyring)


A Generalized Deep Entropy Scaling Architecture Framework to Predict Viscosities as proposed in the following publication:

```
@article{fleck2025generalized,
  title={A Generalized Deep Entropy Scaling Architecture Framework to Predict Viscosities},
  author={Fleck M, Klenk T, Darouich S, Spera MBM, Hansen N.},
  journal={ChemRxiv preprint chemrxiv-2025-jrjj9},
  doi={doi:10.26434/chemrxiv-2025-jrjj9},
  year={2025}
}
```

### Notebooks:

Start:

- demo: example that shows how to use the full model

Advanced:

- main_train: training setup to train the full architecture
- main_denoise: training setup to train denoising



The model was trained with PC-SAFT parameters from the following publications. Use those Parameters if you want to obtain the full predictive capabilities of model train71. You find the Parameter file in demo_data/.
If you use them, please cite the following paper:

```
@article{winter2023understanding,
  title={Understanding the language of molecules: Predicting pure component parameters for the PC-SAFT equation of state from SMILES},
  author={Winter, Benedikt and Rehner, Philipp and Esper, Timm and Schilling, Johannes and Bardow, Andr{\'e}},
  journal={arXiv preprint arXiv:2309.12404},
  year={2023}
}
```
