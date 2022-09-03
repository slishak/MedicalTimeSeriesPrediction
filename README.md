# Medical Time Series Prediction
**MSc Computational Statistics and Machine Learning, University College London**

Modelling and prediction of medical time series from biophysical models and ICU data, using echo state networks (ESN) and ensemble Kalman filters (AD-EnKF).

Requires packages in [requirements.txt](requirements.txt) and Python 3.9. 

Currently requires custom branch of torchdiffeq
`pip install git+https://github.com/slishak/torchdiffeq@manually-reject-step`

Install [torch.interp1d](https://github.com/aliutkus/torchinterp1d) in the same way (only required for fitting biomechanical models with AD-EnKF):
`pip install git+https://github.com/aliutkus/torchinterp1d`

Work in progress.