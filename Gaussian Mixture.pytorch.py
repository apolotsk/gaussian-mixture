"""
Gaussian mixture with 2 normal distribution.
Uses _Stochastic Gradient Descent_, and not _Expectation-Maximization_ as mathematical optimization.
"""
import numpy as np
np.random.seed(0)
import torch
from torch import tensor

x_length, z_length = 1000, 2
def target_θ():
  P_z = tensor(np.expand_dims([0.7, 0.3], axis=0))
  means = tensor(np.expand_dims([8.0, 13.0], axis=0))
  stdevs = tensor(np.expand_dims([1.4, 1.0], axis=0))
  return P_z, means, stdevs
target_θ = target_θ()

def x(target_θ):
  P_z, means, stdevs = target_θ
  P_z, means, stdevs = P_z.numpy(), means.numpy(), stdevs.numpy()
  z = np.random.choice(z_length, size=x_length, p=P_z[0])
  samples = np.random.normal(means, stdevs, [x_length, 1, z_length])
  x = np.choose(z, samples.T).T
  return tensor(x)
x = x(target_θ)

def θ():
  P_z = tensor(np.ones([1, z_length])/z_length, requires_grad=True)
  means = tensor(np.random.rand(1, z_length)*20, requires_grad=True)
  stdevs = tensor(np.random.rand(1, z_length)*5, requires_grad=True)
  return P_z, means, stdevs
θ = θ()

def optimizer(parameters):
  from torch.optim import SGD
  return SGD(parameters, lr=0.001)
optimizer = optimizer(θ)

log_likelihood1, log_likelihood0 = float('inf'), float('-inf')
while abs(log_likelihood1-log_likelihood0)>1e-6:
  '''
  def P_x(x, θ):
    P_z, means, stdevs = θ
    from torch.distributions import MixtureSameFamily, Categorical, Independent, Normal
    distribution = MixtureSameFamily(Categorical(P_z), Independent(Normal(means.T, stdevs.T), 1))
    return torch.exp(distribution.log_prob(x))
  '''
  def P_x(x, θ):
    P_z, means, stdevs = θ
    P_z = P_z/P_z.sum(dim=1, keepdim=True)
    def gauss(x, mean, stdev):
      from math import pi
      from torch import sqrt, exp
      return 1/sqrt(2*pi*stdev**2) * exp(-(x-mean)**2/(2*stdev**2))
    P_x_given_z = gauss(x, means, stdevs)
    P_x_and_z = P_z*P_x_given_z
    P_x = P_x_and_z.sum(dim=1, keepdim=True)
    return P_x
  def log_likelihood(x, θ):
    return torch.log(P_x(x, θ)[:,0]).sum(dim=0)
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
    pyplot.ylabel('P(x|θ)')
    bin_size = 0.5
    bins = np.arange(x.min(), x.max(), bin_size)
    weights = np.ones_like(x)/len(x)/bin_size
    x = x.numpy()
    pyplot.hist(x, bins=bins, weights=weights, color='g', alpha=0.3, label='Samples')

    x = np.linspace([x.min()], [x.max()], 1000)
    def _P_x(x, θ):
      with torch.no_grad(): return P_x(tensor(x), θ).numpy()
    y = _P_x(x, target_θ)
    pyplot.plot(x, y, 'g-', alpha=0.3, label='P(x)')

    y = _P_x(x, θ)
    pyplot.plot(x, y, 'b-', label='P(x|θ)')

    P_z, means, stdevs = θ
    means, stdevs = means.detach().numpy(), stdevs.detach().numpy()
    y = _P_x(means.T, θ).T
    pyplot.plot(means + [[-1],[1]]*stdevs, y/[[2],[2]], '|-b', alpha=0.3, linewidth=1)
    pyplot.plot([[1],[1]]*means, [[0],[1]]*y, '|-b', alpha=0.3, linewidth=1)

    pyplot.show(block=False)
    pyplot.pause(0.01)
  show(x)
