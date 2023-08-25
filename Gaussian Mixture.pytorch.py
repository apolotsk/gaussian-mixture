"""
Gaussian mixture with 2 normal distribution.
Uses _Stochastic Gradient Descent_, and not _Expectation-Maximization_ as mathematical optimization.
"""
import numpy as np
np.random.seed(0)

def gauss(x, mean, stdev):
  from numpy import pi, sqrt, exp
  return 1/sqrt(2*pi*stdev**2) * exp(-(x-mean)**2/(2*stdev**2))

import torch
from torch import tensor

from typing import Iterable, Union
def numpy(tensor:Union[torch.Tensor, Iterable])->np.ndarray:
  return tensor.detach().numpy() if isinstance(tensor, torch.Tensor) else [numpy(t) for t in tensor]

def target_θ():
  p_z = tensor(np.expand_dims([0.3, 0.7], axis=1))
  print(f'The real probability probability of selecting Gaussian 1 is {p_z[0,0]:0.2f} and Gaussian 2 is {p_z[1,0]:0.2f}.')
  means = tensor(np.expand_dims([13.0, 8.0], axis=1))
  print(f'The real mean of Gaussian 1 is {means[0,0]:0.2f} and Gaussian 1 is {means[1,0]:0.2f}.')
  stdevs = tensor(np.expand_dims([1.0, 1.4], axis=1))
  print(f'The real standard deviation of Gaussian 1 is {stdevs[0,0]:0.2f} and Gaussian 1 is {stdevs[1,0]:0.2f}.')
  return p_z, means, stdevs
target_θ = target_θ()

z_length, x_length = 2, 1000
def x(target_θ):
  p_z, means, stdevs = target_θ
  p_z, means, stdevs = p_z.numpy(), means.numpy(), stdevs.numpy()
  z = np.random.choice(z_length, size=x_length, p=p_z[:,0])
  print(f'The real sample count of Gaussian 1 is {(z==0).sum()} and Gaussian 2 is {(z==1).sum()}.')
  samples = np.random.normal(means, stdevs, [z_length, x_length])
  x = np.choose(z, samples)
  from show import show_observable_and_latent_data, show_observable_data
  show_observable_and_latent_data(x, z)
  show_observable_data(x)
  return tensor(x)
x = x(target_θ)

def find_parameters(x):
  print()
  print('Predicting the parameters given only the samples.')

  def θ():
    p_z = tensor(np.ones([z_length, 1])/z_length, requires_grad=True)
    means = tensor(np.random.rand(z_length, 1)*20, requires_grad=True)
    stdevs = tensor(np.random.rand(z_length, 1)*5, requires_grad=True)
    return p_z, means, stdevs
  θ = θ()

  def optimizer(parameters):
    from torch.optim import SGD
    return SGD(parameters, lr=1.0)
  optimizer = optimizer(θ)

  images = []
  log_likelihood1, log_likelihood0 = float('inf'), float('-inf')
  while abs(log_likelihood1-log_likelihood0)>1e-6:
    def p_x(x, θ):
      p_z, means, stdevs = θ
      p_z = p_z/p_z.sum(dim=0, keepdim=True)
      def gauss(x, mean, stdev):
        from math import pi
        from torch import sqrt, exp
        return 1/sqrt(2*pi*stdev**2) * exp(-(x-mean)**2/(2*stdev**2))
      p_x_given_z = gauss(x, means, stdevs)
      p_z_and_x = p_z*p_x_given_z
      p_x = p_z_and_x.sum(dim=0)
      return p_x
    def log_likelihood(x, θ):
      return torch.log(p_x(x, θ)).mean(dim=0)
    log_likelihood1, log_likelihood0 = log_likelihood(x, θ), log_likelihood1

    def update_parameters():
      optimizer.zero_grad()
      (-log_likelihood1).backward()
      optimizer.step()
    update_parameters()

    def _p_x(x, θ):
      with torch.no_grad(): return p_x(tensor(x), tuple(map(tensor,θ))).numpy()
    from show import show_inference
    show_inference(_p_x, numpy(x), numpy(θ), numpy(target_θ))

    from show import plot_to_image
    images.append(plot_to_image())

  from imageio import mimsave
  mimsave('Find parameters.gif', images)

  return θ
θ = find_parameters(x)

x = numpy(x)
p_z, means, stdevs = numpy(θ)
print(f'The predicted probability probability of selecting Gaussian 1 is {p_z[0,0]:0.2f} and Gaussian 2 is {p_z[1,0]:0.2f}.')
print(f'The predicted mean of Gaussian 1 is {means[0,0]:0.2f} and Gaussian 2 is {means[1,0]:0.2f}.')
print(f'The predicted standard deviation of Gaussian 1 is {stdevs[0,0]:0.2f} and Gaussian 2 is {stdevs[1,0]:0.2f}.')

p_z_and_x = p_z * gauss(x, means, stdevs)
z = p_z_and_x/p_z_and_x.sum(axis=0)
print(f'The predicted sample count of Gaussian 1 is {z[0].sum():.1f} and Gaussian 2 is {z[1].sum():.1f}.')
