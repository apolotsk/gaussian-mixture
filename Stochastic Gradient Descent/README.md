# Gaussian Mixture: Find Parameters using [Stochastic Gradient Descent][stochastic-gradient-descent-wiki]

This is an example of finding parameters for a Gaussian Mixture using a mathematical optimization mathod [Stochastic Gradient Descent][stochastic-gradient-descent-wiki].
- Finds $θ := \arg\max_θ L(θ|X)$.
- $L(θ|X)$ is the **expected log-likelihood** of $θ$ given $X$.
- Converges to local _maximum likelihood_.
- Is implemented using PyTorch and NumPy.

[stochastic-gradient-descent-wiki]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

## Usage

```
$ python 'Gaussian Mixture.py'
The real probability probability of selecting Gaussian 1 is 0.30 and Gaussian 2 is 0.70.
The real mean of Gaussian 1 is 13.00 and Gaussian 1 is 8.00.
The real standard deviation of Gaussian 1 is 1.00 and Gaussian 1 is 1.40.
The real sample count of Gaussian 1 is 307 and Gaussian 2 is 693.

Predicting the parameters given only the samples.
The predicted probability probability of selecting Gaussian 1 is 0.31 and Gaussian 2 is 0.72.
The predicted mean of Gaussian 1 is 13.09 and Gaussian 2 is 7.95.
The predicted standard deviation of Gaussian 1 is 0.97 and Gaussian 2 is 1.36.
The predicted sample count of Gaussian 1 is 304.1 and Gaussian 2 is 695.9.
```

![Histogram of the observable data and the latent data](.README.md/Histogram%20of%20the%20observable%20data%20and%20the%20latent%20data.svg)

![Histogram of the observable data](.README.md/Histogram%20of%20the%20observable%20data.svg)

![Find parameters](.README.md/Find%20parameters.gif)
