"""
Gaussian mixture with 2 normal distribution.
http://courses.cs.washington.edu/courses/cse312/11wi/slides/12em.pdf
"""
import numpy as np
np.random.seed(0)

def gauss(x, mean, stdev):
  from numpy import pi, sqrt, exp
  return 1/sqrt(2*pi*stdev**2) * exp(-(x-mean)**2/(2*stdev**2))

z_length, x_length = 2, 1000
def target_θ():
  p_z = np.expand_dims([0.7, 0.3], axis=1)
  means = np.expand_dims([8.0, 13.0], axis=1)
  stdevs = np.expand_dims([1.4, 1.0], axis=1)
  return p_z, means, stdevs
target_θ = target_θ()

def x(target_θ):
  p_z, means, stdevs = target_θ
  z = np.random.choice(z_length, size=x_length, p=p_z[:,0])
  samples = np.random.normal(means, stdevs, [z_length, x_length])
  return np.choose(z, samples)
x = x(target_θ)

def θ():
  p_z = np.ones([z_length, 1])/z_length
  means = np.random.rand(z_length, 1)*20
  stdevs = np.random.rand(z_length, 1)*5
  return p_z, means, stdevs
θ = θ()

log_likelihood1, log_likelihood0 = float('inf'), float('-inf')
while abs(log_likelihood1-log_likelihood0)>1e-6:
  def expectation_step(x, θ):
    p_z, means, stdevs = θ
    p_x_given_z = gauss(x, means, stdevs)
    p_z_and_x = p_z*p_x_given_z
    p_x = p_z_and_x.sum(axis=0, keepdims=True)
    p_z_given_x = p_z_and_x/p_x
    return p_z_given_x
  p_z_given_x = expectation_step(x, θ)

  def maximization_step(p_z_given_x, x):
    w = p_z_given_x.sum(axis=1, keepdims=True)
    p_z = w/len(x)
    means = (p_z_given_x*x).sum(axis=1, keepdims=True)/w
    stdevs = np.sqrt((p_z_given_x*(x-means)**2).sum(axis=1, keepdims=True)/w)
    return p_z, means, stdevs
  θ = maximization_step(p_z_given_x, x)

  def p_x(x, θ):
    p_z, means, stdevs = θ
    p_x_given_z = gauss(x, means, stdevs)
    p_z_and_x = p_z*p_x_given_z
    p_x = p_z_and_x.sum(axis=0)
    return p_x

  def show(x):
    from matplotlib import pyplot
    pyplot.clf()
    pyplot.xlabel('$x$')
    pyplot.ylabel('$p(x|θ)$')
    bin_size = 0.5
    bins = np.arange(x.min(), x.max(), bin_size)
    weights = np.ones_like(x)/len(x)/bin_size
    pyplot.hist(x, bins=bins, weights=weights, color='g', alpha=0.3, label='Samples')

    x = np.linspace(x.min(), x.max(), 1000)
    y = p_x(x, target_θ)
    pyplot.plot(x, y, 'g-', alpha=0.3, label='$p(x)$')

    y = p_x(x, θ)
    pyplot.plot(x, y, 'b-', label='$p(x|θ)$')

    p_z, means, stdevs = θ
    means, stdevs = means.T, stdevs.T
    y = p_x(means, θ)
    pyplot.plot(means + [[-1],[1]]*stdevs, y/[[2],[2]], '|-b', alpha=0.3, linewidth=1)
    pyplot.plot([[1],[1]]*means, [[0],[1]]*y, '|-b', alpha=0.3, linewidth=1)

    pyplot.show(block=False)
    pyplot.pause(0.01)
  show(x)

  def log_likelihood(x, θ):
    return np.log(p_x(x, θ)).mean(axis=0)
  log_likelihood1, log_likelihood0 = log_likelihood(x, θ), log_likelihood1
