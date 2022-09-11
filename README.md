# Medical Time Series Prediction
**MSc Computational Statistics and Machine Learning, University College London**

Modelling and prediction of medical time series from biophysical models and ICU data, using echo state networks (ESN) [1] and ensemble Kalman filters (AD-EnKF) [2].

Requires packages in [requirements.txt](requirements.txt) and Python 3.9. 

Currently requires custom branch of torchdiffeq for AD-EnKF, pending merge of https://github.com/rtqichen/torchdiffeq/pull/210:
```
pip install git+https://github.com/slishak/torchdiffeq@manually-reject-step
```

Install [torch.interp1d](https://github.com/aliutkus/torchinterp1d) in the same way (only required for fitting biomechanical models with AD-EnKF):
```
pip install git+https://github.com/aliutkus/torchinterp1d
```

## Packages

Two Python packages are included in this respository. 
[`time_series_prediction`](time_series_prediction) contains code that implements ESN and AD-EnKF with PyTorch. Example ODE problems are the Lorenz and Rössler attractors, and a forced Van der Pol oscillator.
[`biomechanical_models`](biophysical_models) contains Python implementations of Smith's inertial/non-cardiovascular models [3], and Jallon's heart-lung model [4]. Parameters from Paeme [5] are also used.

## Example scripts

The [`examples`](examples) folder contains various Python scripts and Jupyter notebooks which call code from the packages above. In VS Code, it is recommended to add the following snippet to your `.vscode/settings.json` file in order that the packages are visible on the Python path:

```json
{
    "terminal.integrated.env.windows": {"PYTHONPATH": "${workspaceFolder}"}
}
```

The example scripts are not documented in as detail as the implementation packages. Some key example scripts are:
- [`cvs_example.py`](examples/cvs_example.py): Simulate a cardiovascular system ODE model. Requires some commenting/uncommenting to choose the type of simulation to run.
- [`esn_hyperparameter_sweep.ipynb`](examples/esn_hyperparameter_sweep.ipynb) runs a large sweep of Echo State Networks on the example ODE problems with varied hyperparameters, and plots the results.
- [`esn_hyperparameter_sweep_jallon.ipynb`](examples/esn_hyperparameter_sweep_jallon.ipynb) does the same but using data from the Jallon heart-lung model
- [`enkf_activation.ipynb`](examples/enkf_activation.ipynb) is an example sweep of activation function options when training an RNN with AD-EnKF on the three example ODE problems.
- [`enkf_jallon.ipynb`](examples/enkf_jallon.ipynb) tries to fit RNNs of two different architectures to data from the Jallon heart-lung model


## References
1. Herbert Jaeger. ‘The “echo state” approach to analysing and training recurrent neural networks-with an erratum note’. In: Bonn, Germany: German National Research Center for Information Technology GMD Technical Report 148 (Jan. 2001).
2. Yuming Chen, Daniel Sanz-Alonso and Rebecca Willett. ‘Auto-differentiable Ensemble Kalman Filters’. In: SIAM Journal on Mathematics of Data Science 4.2 (2022), pp. 801–833. doi: 10.1137/21M1434477. url: https://doi.org/10.1137/21M1434477.
3. Bram W Smith et al. ‘Minimal haemodynamic system model including ventricular interaction and valve dynamics’. en. In: Med. Eng. Phys. 26.2 (Mar. 2004), pp. 131–139.
4. Julie Fontecave Jallon et al. ‘A model of mechanical interactions between heart and lungs’. In: Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 367.1908 (2009), pp. 4741–4757. doi: 10.1098/rsta.2009.0137. eprint: url: https://royalsocietypublishing.org/doi/abs/10.1098/rsta.2009.0137.
5. Sabine Paeme et al. ‘Mathematical multi-scale model of the cardiovascular system including mitral valve dynamics. Application to ischemic mitral insufficiency’. In: BioMedical Engineering OnLine 10.1 (Sept. 2011), p. 86. issn: 1475-925X. doi: 10.1186/1475-925X-10-86. url: https://doi.org/10.1186/1475-925X-10-86.
