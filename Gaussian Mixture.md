### Gaussian Mixture

Description:
- They have $n$ Gaussians.
- They select one Gaussian according to some distribution.
- They sample from the selected Gaussian.
- They repeat the process several times.
- They tell us a set of samples.
- They do not tell us which Gaussian was used for each sample.
- They do not tell us the distribution to select a Gaussian.

Definitions:
- $x$ is the sample value.
  - $x$ is the _observable_ data.
  </br>![Histogram of the observable data](.Gaussian%20Mixture.md/Histogram%20of%20the%20observable%20data.svg)
- $z$ is the selected Gaussian.
  - $z$ is the _latent_ data.
  - $z \in \{ 1, 2, ..., n \}$.
   </br>![Histogram of the observable data and the latent data](.Gaussian%20Mixture.md/Histogram%20of%20the%20observable%20data%20and%20the%20latent%20data.svg)
- That is, they know the set of $(x,z)$, but tell us only the set of $x$.
- We want to infer the biases (the parameters).
  - This allows to infer the set of $(x,z)$.
- We use the _Expectation-maximization_ algorithm given the set of $x$.

$θ := \{ p(z|θ), \mu_{z|θ}, \sigma_{z|θ} \text{ for } z \in \{1, 2, ..., n \} \}$
- $p(z|θ)$ is the probability of selecting Gaussian $z$.
- $\mu_{z|θ}$ is the mean of the Gaussian $z$.
- $\sigma_{z|θ}$ is the standard deviation of the Gaussian $z$.

$\color{green}{p(x,z|θ)} = p(z|θ) p(x|z,θ)$
- $p(x|z,θ) := \mathcal{N}(x|\mu_{z|θ},\sigma_{z|θ})$
  - $\mathcal{N}(x|\mu,\sigma) := \frac 1 {\sigma {\sqrt {2\pi }}} e^{- \frac {(x-\mu)^2} {2 \sigma^2}}$
  - $\mathcal{N}(x|\mu,\sigma)$ is a normal (Gaussian) distribution.
  - $p(x=a|z,θ)$ is the probability to sample value $a$ from the Gaussian $z$.

$θ_{t+1} := \{ p(z|θ_{t+1}), \mu_{z|θ_{t+1}}, \sigma_{z|θ_{t+1}} \text{ for } z \in \{1, 2, ..., n \} \}$
- $p(z|θ_{t+1}) := \sum_x \color{orange}{p(z|x,θ_t)} / |X|$
  - $\sum_x p(z|x,θ_t)$ is the number of coin $z$ choices.
  - $|X|$ is the number of any coin choices.
- $\mu_{z|θ_{t+1}} := \sum_x x \cdot p(x|z,θ_{t+1})$
  - $p(x|z,θ_{t+1}) := \color{orange}{p(z|x,θ_t)} / |X| / p(z|θ_{t+1})$
- $\sigma_{z|θ_{t+1}} := \sqrt{ \sum_x (x-\mu_{z|θ_{t+1}})^2 \cdot p(x|z,θ_{t+1}) }$

Proof:
- $p(z|θ_{t+1})$
  - $:= \arg\max_{p(z|θ)} L(θ|θ_t,X)$
  - $\frac \partial {\partial p(z|θ)} L(θ|θ_t,X) = 0$
    - $= \frac \partial {\partial p(z|θ)}( L(θ|θ_t,X) - 𝜆 \cdot (\sum_z p(z|θ)-1))$
      - $𝜆$ is the _Lagrange multiplier_.
    - $= \frac \partial {\partial p(z|θ)} L(θ|θ_t,X) - \frac \partial {\partial p(z|θ)} 𝜆 \cdot (\sum_z p(z|θ)-1)$
    - $= \frac \partial {\partial p(z|θ)} L(θ|θ_t,X) - 𝜆$
    - $= \sum_x \color{orange}{p(z|x,θ_t)}/|X|/p(z|θ) - 𝜆$
  - $\sum_x \color{orange}{p(z|x,θ_t)}/|X|/p(z|θ) - 𝜆 = 0$
  - $𝜆 = \sum_x \color{orange}{p(z|x,θ_t)}/|X|/p(z|θ)$
  - $𝜆 \cdot p(z|θ) = \sum_x \color{orange}{p(z|x,θ_t)}/|X|$
  - $\sum_z 𝜆 \cdot p(z|θ) = \sum_z \sum_x \color{orange}{p(z|x,θ_t)}/|X|$
  - $𝜆 \cdot \sum_z p(z|θ) = \sum_x \sum_z \color{orange}{p(z|x,θ_t)}/|X|$
  - $𝜆 \cdot 1 = \sum_x 1/|X|$
  - $𝜆 = 1$
  - $= \sum_x \color{orange}{p(z|x,θ_t)} / |X|$
- $p(x|z,θ_{t+1})$
  - $:= \arg\max_{p(x|z,θ)} L(θ|θ_t,X)$
  - $\frac \partial {\partial p(x|z,θ)} L(θ|θ_t,X) = 0$
    - $= \frac \partial {\partial p(x|z,θ)}( L(θ|θ_t,X) - 𝜆 \cdot (\sum_z p(x|z,θ)-1))$
      - $𝜆$ is the _Lagrange multiplier_.
    - $= \color{orange}{p(z|x,θ_t)}/|X|/p(x|z,θ) - 𝜆$
  - $\color{orange}{p(z|x,θ_t)}/|X|/p(x|z,θ) - 𝜆 = 0$
  - $𝜆 = \color{orange}{p(z|x,θ_t)}/|X|/p(x|z,θ)$
  - $𝜆 \cdot p(x|z,θ) = \color{orange}{p(z|x,θ_t)}/|X|$
  - $\sum_x 𝜆 \cdot p(x|z,θ) = \sum_x \color{orange}{p(z|x,θ_t)}/|X|$
  - $𝜆 \cdot \sum_x p(x|z,θ) = p(z|θ)$
  - $𝜆 \cdot 1 = p(z|θ)$
  - $𝜆 = p(z|θ)$
  - $\color{orange}{p(z|x,θ_t)}/|X| / p(x|z,θ) - p(z|θ) = 0$
  - $p(x|z,θ) = \color{orange}{p(z|x,θ_t)}/|X| / p(z|θ)$
  - $= \color{orange}{p(z|x,θ_t)} / |X| / p(z|θ_{t+1})$
- $\mu_{z|θ_{t+1}}$
  - $:= \arg\max_{\mu_{z|θ}} L(θ|θ_t,X)$
  - $\frac \partial {\partial \mu_{z|θ}} L(θ|θ_t,X) = 0$
    - $= \sum_x \frac \partial {\partial p(x|z,θ)} L(θ|θ_t,X) \cdot \frac {\partial p(x|z,θ)} {\partial \mu_{z|θ}}$
  - $\frac \partial {\partial p(x|z,θ)} L(θ|θ_t,X)$
    - $= \color{orange}{p(z|x,θ_t)}/|X|/p(x|z,θ)$
  - $\frac {\partial p(x|z,θ)} {\partial \mu_{z|θ}}$
    - $= \frac 1 {\sigma_{z|θ} {\sqrt {2\pi }}} e^{- \frac {(x-\mu_{z|θ})^2} {2 \sigma_{z|θ}^2}} \cdot (-\frac 1 {2\sigma_{z|θ}^2} \cdot 2 \cdot (x-\mu_{z|θ}) \cdot (-1))$
    - $= p(x|z,θ) \cdot (x-\mu_{z|θ})/\sigma_{z|θ}^2$
  - $\sum_x \color{orange}{p(z|x,θ_t)}/|X| \cdot (x-\mu_{z|θ})/\sigma_{z|θ}^2 = 0$
  - $\sum_x \color{orange}{p(z|x,θ_t)}/|X| \cdot (x-\mu_{z|θ}) = 0$
  - $\sum_x \color{orange}{p(z|x,θ_t)}/|X| \cdot \mu_{z|θ} = \sum_x \color{orange}{p(z|x,θ_t)}/|X| \cdot x$
  - $\mu_{z|θ} \cdot \sum_x \color{orange}{p(z|x,θ_t)}/|X| = \sum_x x \cdot \color{orange}{p(z|x,θ_t)}/|X|$
  - $\mu_{z|θ} \cdot p(z|θ) = \sum_x x \cdot \color{orange}{p(z|x,θ_t)}/|X|$
  - $\mu_{z|θ} = \sum_x x \cdot \color{orange}{p(z|x,θ_t)}/|X| / p(z|θ)$
  - $\mu_{z|θ} = \sum_x x \cdot p(x|z,θ)$
  - $= \sum_x x \cdot p(x|z,θ_{t+1})$
- $\sigma_{z|θ_{t+1}}$
  - $:= \arg\max_{\sigma_{z|θ}} L(θ|θ_t,X)$
  - $\frac \partial {\partial \sigma_{z|θ}} L(θ|θ_t,X) = 0$
    - $= \sum_x \frac \partial {\partial p(x|z,θ)} L(θ|θ_t,X) \cdot \frac {\partial p(x|z,θ)} {\partial \sigma_{z|θ}}$
  - $\frac {\partial p(x|z,θ)} {\partial \sigma_{z|θ}}$
    - $= \frac 1 {\sigma_{z|θ} {\sqrt {2\pi }}} e^{- \frac {(x-\mu_{z|θ})^2} {2 \sigma_{z|θ}^2}} \cdot \frac {-(x-\mu_{z|θ})^2} 2 \frac {-2} {\sigma_{z|θ}^3} - \frac 1 {\sigma_{z|θ} {\sqrt {2\pi }}} e^{- \frac {(x-\mu_{z|θ})^2} {2 \sigma_{z|θ}^2}} \cdot \frac 1 \sigma_{z|θ}$
    - $= p(x|z,θ) \cdot (x-\mu_{z|θ})^2\sigma_{z|θ}^{-3} - p(x|z,θ) \cdot \sigma_{z|θ}^{-1}$
    - $= p(x|z,θ) \cdot \sigma_{z|θ}^{-3} \left( (x-\mu_{z|θ})^2 - \sigma_{z|θ}^2 \right)$
  - $\sum_x \color{orange}{p(z|x,θ_t)}/|X| \cdot \sigma_{z|θ}^{-3} \left( (x-\mu_{z|θ})^2 - \sigma_{z|θ}^2 \right) = 0$
  - $\sum_x \color{orange}{p(z|x,θ_t)}/|X| \cdot \left( (x-\mu_{z|θ})^2 - \sigma_{z|θ}^2 \right) = 0$
  - $\sum_x \color{orange}{p(z|x,θ_t)}/|X| \cdot \sigma_{z|θ}^2 = \sum_x \color{orange}{p(z|x,θ_t)}/|X| \cdot (x-\mu_{z|θ})^2$
  - $\sigma_{z|θ}^2 \cdot \sum_x \color{orange}{p(z|x,θ_t)}/|X| = \sum_x (x-\mu_{z|θ})^2 \cdot \color{orange}{p(z|x,θ_t)}/|X|$
  - $\sigma_{z|θ}^2 \cdot p(z|θ) = \sum_x (x-\mu_{z|θ})^2 \cdot \color{orange}{p(z|x,θ_t)}/|X|$
  - $\sigma_{z|θ}^2 = \sum_x (x-\mu_{z|θ})^2 \cdot \color{orange}{p(z|x,θ_t)}/|X| / p(z|θ)$
  - $\sigma_{z|θ}^2 = \sum_x (x-\mu_{z|θ})^2 \cdot p(x|z,θ)$
  - $= \sqrt{ \sum_x (x-\mu_{z|θ_{t+1}})^2 \cdot p(x|z,θ_{t+1}) }$
