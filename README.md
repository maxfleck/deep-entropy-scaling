# deep-entropy-scaling

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

## Notebooks:

### Quickstart:

install [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

then run (all packages will be downloaded and saved locally)
```
uv run jupyter-lab
```

### Start:

- demo: example that shows how to use the full model

### Advanced:

We cannot publish training data. Therefore you wont be able to run the following notebooks. Use your own data and build your dataset. Then you can load this data and train your models.

- main_train: training setup to train the full architecture
- main_denoise: training setup to train denoising



The model was trained with PC-SAFT parameters from the following publications. Use those Parameters if you want to obtain the full predictive capabilities of model train71. You find the Parameter file in demo_data/.
If you use them, please cite the following paper:

```
@article{winter2025understanding,
  title={Understanding the language of molecules: Predicting pure component parameters for the PC-SAFT equation of state from SMILES},
  author={Winter, Benedikt and Rehner, Philipp and Esper, Timm and Schilling, Johannes and Bardow, Andr{\'e}},
  journal={Digital Discovery},
  year={2025},
  publisher={Royal Society of Chemistry}
}
```
