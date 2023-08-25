# Gaussian Mixture

This is examples of finding (inferring) parameters of a Gaussian Mixture
- using [Expectation-maximization](./Expectation-maximization/) (implemented with NumPy), or
- using [Stochastic Gradient Descent](./Stochastic%20Gradient%20Descent/) (implemented with PyTorch).


## The problem description
- They (as opposed to us) have $n$ (for example, 2) Gaussians.
- They select one Gaussian according to some distribution.
- They sample from the selected Gaussian.
- They repeat the process several times.
- They tell us a set of samples.
- They do not tell us which Gaussian was used for each sample, but we want to know.
- They do not tell us the distribution to select a Gaussian, but we want to know.

## The problem definitions
- $x$ is the sample value.
  - $x$ is the _observable_ data.
  </br>![Histogram of the observable data](Expectation-maximization/.README.md/Histogram%20of%20the%20observable%20data.svg)
- $z$ is the selected Gaussian.
  - $z$ is the _latent_ data.
  - $z \in \{ 1, 2, ..., n \}$.
   </br>![Histogram of the observable data and the latent data](Expectation-maximization/.README.md/Histogram%20of%20the%20observable%20data%20and%20the%20latent%20data.svg)
- They know the set of $(x,z)$, but tell us only the set of $x$.
- We want to infer (or find) the the parameters (of the Gaussians) $θ$.
  - This allows to infer the set of $(x,z)$.

$θ := \{ p(z|θ), \mu_{z|θ}, \sigma_{z|θ} \text{ for } z \in \{1, 2, ..., n \} \}$
- $p(z|θ)$ is the probability of selecting Gaussian $z$.
- $\mu_{z|θ}$ is the mean of the Gaussian $z$.
- $\sigma_{z|θ}$ is the standard deviation of the Gaussian $z$.


## [A solution using Expectation-maximization](./Expectation-maximization/)

```
$ python 'Expectation-maximization/Gaussian Mixture.py'
```

![Find parameters](Expectation-maximization/.README.md/Find%20parameters.gif)


## [A solution using Stochastic Gradient Descent](./Stochastic%20Gradient%20Descent/)

```
$ python 'Stochastic Gradient Descent/Gaussian Mixture.py'
```

![Find parameters](Stochastic%20Gradient%20Descent/.README.md/Find%20parameters.gif)
