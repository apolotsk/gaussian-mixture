# Gaussian Mixture: Find Parameters using [Expectation-maximization][expectation-maximization-wiki]

This is an example of finding parameters for a Gaussian Mixture using a mathematical optimization mathod [Expectation-maximization][expectation-maximization-wiki].
- Finds $θ := \arg\max_θ L(θ|X)$.
- $L(θ|X)$ is the **expected log-likelihood** of $θ$ given $X$.
- [Converges to local][expectation-maximization-wiki] _maximum likelihood_.
- Is implemented using NumPy.
- Is based on a [lecture from University of Washington course "Foundations of Computing II"][slides].

[expectation-maximization-wiki]: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm

[slides]: http://courses.cs.washington.edu/courses/cse312/11wi/slides/12em.pdf "University of Washington, Course 'Foundations of Computing II', Lecture '12:	Expectation Maximization'"

## Usage

```
$ python 'Gaussian Mixture.py'
The real probability probability of selecting Gaussian 1 is 0.30 and Gaussian 2 is 0.70.
The real mean of Gaussian 1 is 13.00 and Gaussian 1 is 8.00.
The real standard deviation of Gaussian 1 is 1.00 and Gaussian 1 is 1.40.
The real sample count of Gaussian 1 is 307 and Gaussian 2 is 693.

Finding the parameters given only the samples.
The predicted probability probability of selecting Gaussian 1 is 0.30 and Gaussian 2 is 0.70.
The predicted mean of Gaussian 1 is 13.09 and Gaussian 2 is 7.96.
The predicted standard deviation of Gaussian 1 is 0.97 and Gaussian 2 is 1.36.
The predicted sample count of Gaussian 1 is 304.1 and Gaussian 2 is 695.9.
```

![Histogram of the observable data and the latent data](.README.md/Histogram%20of%20the%20observable%20data%20and%20the%20latent%20data.svg)

![Histogram of the observable data](.README.md/Histogram%20of%20the%20observable%20data.svg)

![Find parameters](.README.md/Find%20parameters.gif)

## The algorithm

1. $θ_0 :=$ random values of the parameters.
2. $t :=$ 0 to $∞$:
   1. $L(θ|θ_t,X) := \sum_x p(x) \sum_z{\color{orange}p(z|x,θ_t)}\log {\color{green}p(x,z|θ)}$
      - $L(θ|θ_t,X)$ is an approximation to $-\text{D}_\text{KL}(p(x)||p(x|θ))$.
      - Is called _expectation step_.
      - ${\color{orange}p(z|x,θ_t)} = {\color{darkgreen}p(x,z|θ_t)}/\sum_z {\color{darkgreen}p(x,z|θ_t)}$
      - $p(x) := 1/|X|$.
        - [Because $x$ are independent and identically distributed][mle-wiki].
   2. $θ_{t+1} := \arg\max_θ L(θ|θ_t,X)$
      - Is called _maximization step_.
   3. If $θ$ converged, then break the loop.

[mle-wiki]: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation#Properties

$θ_{t+1}$ is defined as following:
- $p(z|θ_{t+1}) := \sum_x {\color{orange}p(x,z|θ^\star_t)}$
  - ${\color{orange}p(x,z|θ^\star_t)} := {\color{orange}p(z|x,θ_t)}/|X|$
- $\mu_{z|θ_{t+1}} := \sum_x p(x|z,θ_{t+1}) \cdot x$
  - $p(x|z,θ_{t+1}) := {\color{orange}p(x,z|θ^\star_t)} / p(z|θ_{t+1})$
- $\sigma_{z|θ_{t+1}} := \sqrt{ \sum_x p(x|z,θ_{t+1}) \cdot (x-\mu_{z|θ_{t+1}})^2 }$

Proof:

- $p(z|θ_{t+1})$
  - $:= \arg\max_{p(z|θ)} L(θ|θ_t,X)$
  - $\frac \partial {\partial p(z|θ)} L(θ|θ_t,X) = 0$
    - $= \frac \partial {\partial p(z|θ)}( L(θ|θ_t,X) - 𝜆 \cdot (\sum_z p(z|θ)-1))$
      - $𝜆$ is the _Lagrange multiplier_.
    - $= \frac \partial {\partial p(z|θ)} L(θ|θ_t,X) - \frac \partial {\partial p(z|θ)} 𝜆 \cdot (\sum_z p(z|θ)-1)$
    - $= \frac \partial {\partial p(z|θ)} L(θ|θ_t,X) - 𝜆$
    - $= \sum_x {\color{orange}p(x,z|θ^\star_t)}/p(z|θ) - 𝜆$
  - $\sum_x {\color{orange}p(x,z|θ^\star_t)}/p(z|θ) - 𝜆 = 0$
  - $𝜆 = \sum_x {\color{orange}p(x,z|θ^\star_t)}/p(z|θ)$
  - $𝜆 \cdot p(z|θ) = \sum_x {\color{orange}p(x,z|θ^\star_t)}$
  - $\sum_z 𝜆 \cdot p(z|θ) = \sum_z \sum_x {\color{orange}p(x,z|θ^\star_t)}$
  - $𝜆 \cdot \sum_z p(z|θ) = \sum_x \sum_z {\color{orange}p(z|x,θ_t)}/|X|$
  - $𝜆 \cdot 1 = \sum_x 1/|X|$
  - $𝜆 = 1$
  - $\sum_x {\color{orange}p(x,z|θ^\star_t)}/p(z|θ) - 1 = 0$
  - $1 = \sum_x {\color{orange}p(x,z|θ^\star_t)}/p(z|θ)$
  - $p(z|θ) = \sum_x {\color{orange}p(x,z|θ^\star_t)}$
  - $= \sum_x {\color{orange}p(x,z|θ^\star_t)}$
- $p(x|z,θ_{t+1})$
  - $:= \arg\max_{p(x|z,θ)} L(θ|θ_t,X)$
  - $\frac \partial {\partial p(x|z,θ)} L(θ|θ_t,X) = 0$
    - $= \frac \partial {\partial p(x|z,θ)}( L(θ|θ_t,X) - 𝜆 \cdot (\sum_z p(x|z,θ)-1))$
      - $𝜆$ is the _Lagrange multiplier_.
    - $= {\color{orange}p(x,z|θ^\star_t)}/p(x|z,θ) - 𝜆$
  - ${\color{orange}p(x,z|θ^\star_t)}/p(x|z,θ) - 𝜆 = 0$
  - $𝜆 = {\color{orange}p(x,z|θ^\star_t)}/p(x|z,θ)$
  - $𝜆 \cdot p(x|z,θ) = {\color{orange}p(x,z|θ^\star_t)}$
  - $\sum_x 𝜆 \cdot p(x|z,θ) = \sum_x {\color{orange}p(x,z|θ^\star_t)}$
  - $𝜆 \cdot \sum_x p(x|z,θ) = p(z|θ)$
  - $𝜆 \cdot 1 = p(z|θ)$
  - $𝜆 = p(z|θ)$
  - ${\color{orange}p(x,z|θ^\star_t)} / p(x|z,θ) - p(z|θ) = 0$
  - $p(x|z,θ) = {\color{orange}p(x,z|θ^\star_t)} / p(z|θ)$
  - $= {\color{orange}p(x,z|θ^\star_t)} / p(z|θ_{t+1})$
- $\mu_{z|θ_{t+1}}$
  - $:= \arg\max_{\mu_{z|θ}} L(θ|θ_t,X)$
  - $\frac \partial {\partial \mu_{z|θ}} L(θ|θ_t,X) = 0$
    - $= \sum_x \frac \partial {\partial p(x|z,θ)} L(θ|θ_t,X) \cdot \frac {\partial p(x|z,θ)} {\partial \mu_{z|θ}}$
  - $\frac \partial {\partial p(x|z,θ)} L(θ|θ_t,X)$
    - $= {\color{orange}p(x,z|θ^\star_t)}/p(x|z,θ)$
  - $\frac {\partial p(x|z,θ)} {\partial \mu_{z|θ}}$
    - $= \frac 1 {\sigma_{z|θ} {\sqrt{2\pi}}} e^{- \frac {(x-\mu_{z|θ})^2} {2 \sigma_{z|θ}^2}} \cdot (-\frac 1 {2\sigma_{z|θ}^2} \cdot 2 \cdot (x-\mu_{z|θ}) \cdot (-1))$
    - $= p(x|z,θ) \cdot (x-\mu_{z|θ})/\sigma_{z|θ}^2$
  - $\sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot (x-\mu_{z|θ})/\sigma_{z|θ}^2 = 0$
  - $\sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot (x-\mu_{z|θ}) = 0$
  - $\sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot \mu_{z|θ} = \sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot x$
  - $\mu_{z|θ} \cdot \sum_x {\color{orange}p(x,z|θ^\star_t)} = \sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot x$
  - $\mu_{z|θ} \cdot p(z|θ) = \sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot x$
  - $\mu_{z|θ} = \sum_x {\color{orange}p(x,z|θ^\star_t)} / p(z|θ) \cdot x$
  - $\mu_{z|θ} = \sum_x p(x|z,θ) \cdot x$
  - $= \sum_x p(x|z,θ_{t+1}) \cdot x$
- $\sigma_{z|θ_{t+1}}$
  - $:= \arg\max_{\sigma_{z|θ}} L(θ|θ_t,X)$
  - $\frac \partial {\partial \sigma_{z|θ}} L(θ|θ_t,X) = 0$
    - $= \sum_x \frac \partial {\partial p(x|z,θ)} L(θ|θ_t,X) \cdot \frac {\partial p(x|z,θ)} {\partial \sigma_{z|θ}}$
  - $\frac {\partial p(x|z,θ)} {\partial \sigma_{z|θ}}$
    - $= \frac 1 {\sigma_{z|θ} {\sqrt{2\pi}}} e^{- \frac {(x-\mu_{z|θ})^2} {2 \sigma_{z|θ}^2}} \cdot \frac {-(x-\mu_{z|θ})^2} 2 \frac {-2} {\sigma_{z|θ}^3} - \frac 1 {\sigma_{z|θ} {\sqrt{2\pi}}} e^{- \frac {(x-\mu_{z|θ})^2} {2 \sigma_{z|θ}^2}} \cdot \frac 1 \sigma_{z|θ}$
    - $= p(x|z,θ) \cdot (x-\mu_{z|θ})^2\sigma_{z|θ}^{-3} - p(x|z,θ) \cdot \sigma_{z|θ}^{-1}$
    - $= p(x|z,θ) \cdot \sigma_{z|θ}^{-3} \left( (x-\mu_{z|θ})^2 - \sigma_{z|θ}^2 \right)$
  - $\sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot \sigma_{z|θ}^{-3} \left( (x-\mu_{z|θ})^2 - \sigma_{z|θ}^2 \right) = 0$
  - $\sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot \left( (x-\mu_{z|θ})^2 - \sigma_{z|θ}^2 \right) = 0$
  - $\sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot \sigma_{z|θ}^2 = \sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot (x-\mu_{z|θ})^2$
  - $\sigma_{z|θ}^2 \cdot \sum_x {\color{orange}p(x,z|θ^\star_t)} = \sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot (x-\mu_{z|θ})^2$
  - $\sigma_{z|θ}^2 \cdot p(z|θ) = \sum_x {\color{orange}p(x,z|θ^\star_t)} \cdot (x-\mu_{z|θ})^2$
  - $\sigma_{z|θ}^2 = \sum_x {\color{orange}p(x,z|θ^\star_t)} / p(z|θ) \cdot (x-\mu_{z|θ})^2$
  - $\sigma_{z|θ}^2 = \sum_x p(x|z,θ) \cdot (x-\mu_{z|θ})^2$
  - $= \sqrt{ \sum_x p(x|z,θ_{t+1}) \cdot (x-\mu_{z|θ_{t+1}})^2 }$
