"""
Gaussian mixture with 2 normal distribution.
Uses _Stochastic Gradient Descent_, and not _Expectation-Maximization_ as mathematical optimization.
"""
import numpy as np
np.random.seed(0)
import torch
from torch import tensor

z_length, x_length = 2, 1000
def target_θ():
  p_z = tensor(np.expand_dims([0.7, 0.3], axis=1))
  means = tensor(np.expand_dims([8.0, 13.0], axis=1))
  stdevs = tensor(np.expand_dims([1.4, 1.0], axis=1))
  return p_z, means, stdevs
target_θ = target_θ()

def x(target_θ):
  p_z, means, stdevs = target_θ
  p_z, means, stdevs = p_z.numpy(), means.numpy(), stdevs.numpy()
  z = np.random.choice(z_length, size=x_length, p=p_z[:,0])
  samples = np.random.normal(means, stdevs, [z_length, x_length])
  x = np.choose(z, samples)
  return tensor(x)
x = x(target_θ)

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

  def show(x):
    from matplotlib import pyplot
    pyplot.clf()
    pyplot.xlabel('x')
    pyplot.ylabel('p(x|θ)')
    bin_size = 0.5
    bins = np.arange(x.min(), x.max(), bin_size)
    weights = np.ones_like(x)/len(x)/bin_size
    x = x.numpy()
    pyplot.hist(x, bins=bins, weights=weights, color='g', alpha=0.3, label='Samples')

    x = np.linspace(x.min(), x.max(), 1000)
    def _p_x(x, θ):
      with torch.no_grad(): return p_x(tensor(x), θ).numpy()
    y = _p_x(x, target_θ)
    pyplot.plot(x, y, 'g-', alpha=0.3, label='p(x)')

    y = _p_x(x, θ)
    pyplot.plot(x, y, 'b-', label='p(x|θ)')

    p_z, means, stdevs = θ
    means, stdevs = means.T, stdevs.T
    with torch.no_grad(): means, stdevs = means.numpy(), stdevs.numpy()
    y = _p_x(means, θ)
    pyplot.plot(means + [[-1],[1]]*stdevs, y/[[2],[2]], '|-b', alpha=0.3, linewidth=1)
    pyplot.plot([[1],[1]]*means, [[0],[1]]*y, '|-b', alpha=0.3, linewidth=1)

    pyplot.show(block=False)
    pyplot.pause(0.01)
  show(x)
