"""
Gaussian mixture with 2 normal distribution.
http://courses.cs.washington.edu/courses/cse312/11wi/slides/12em.pdf
"""
import numpy as np
np.random.seed(0)

def gauss(x, mean, stdev):
  from numpy import pi, sqrt, exp
  return 1/sqrt(2*pi*stdev**2) * exp(-(x-mean)**2/(2*stdev**2))

x_length, z_length = 1000, 2
def target_θ():
  P_z = np.expand_dims([0.7, 0.3], axis=0)
  means = np.expand_dims([8.0, 13.0], axis=0)
  stdevs = np.expand_dims([1.4, 1.0], axis=0)
  return P_z, means, stdevs
target_θ = target_θ()

def x(target_θ):
  P_z, means, stdevs = target_θ
  z = np.random.choice(z_length, size=x_length, p=P_z[0])
  samples = np.random.normal(means, stdevs, [x_length, 1, z_length])
  return np.choose(z, samples.T).T
x = x(target_θ)

def θ():
  def normalize(x, axis=1): return x/x.sum(axis=axis, keepdims=True)
  P_z = normalize(np.random.rand(1, z_length)*0.5+0.25)
  means = np.random.rand(1, z_length)*20-10
  stdevs = np.random.rand(1, z_length)*5
  return P_z, means, stdevs
θ = θ()

log_likelihood1, log_likelihood0 = float('inf'), float('-inf')
while abs(log_likelihood1-log_likelihood0)>1e-6:
  def expectation_step(x, θ):
    P_z, means, stdevs = θ
    P_x_given_z = gauss(x, means, stdevs)
    P_x_and_z = P_z*P_x_given_z
    P_x = P_x_and_z.sum(axis=1, keepdims=True)
    P_z_given_x = P_x_and_z/P_x
    return P_z_given_x
  P_z_given_x = expectation_step(x, θ)

  def maximization_step(P_z_given_x, x):
    w = P_z_given_x.sum(axis=0, keepdims=True)
    P_z = w/len(x)
    means = (P_z_given_x*x).sum(axis=0, keepdims=True)/w
    stdevs = np.sqrt((P_z_given_x*(x-means)**2).sum(axis=0, keepdims=True)/w)
    return P_z, means, stdevs
  θ = maximization_step(P_z_given_x, x)

  def show(x):
    import numpy as np
    from matplotlib import pyplot
    pyplot.clf()
    pyplot.xlabel('x')
    pyplot.ylabel('P(x|θ)')
    bin_size = 0.5
    bins = np.arange(x.min(), x.max(), bin_size)
    weights = np.ones_like(x)/len(x)/bin_size
    pyplot.hist(x, bins=bins, weights=weights, color='g', alpha=0.3, label='Samples')

    x = np.linspace([x.min()], [x.max()], 1000)
    def P_x(x, θ):
      P_z, means, stdevs = θ
      P_x_given_z = gauss(x, means, stdevs)
      P_x_and_z = P_z*P_x_given_z
      P_x = P_x_and_z.sum(axis=1, keepdims=True)
      return P_x
    y = P_x(x, target_θ)
    pyplot.plot(x, y, 'g-', alpha=0.3, label='P(x)')

    y = P_x(x, θ)
    pyplot.plot(x, y, 'b-', label='P(x|θ)')

    P_z, means, stdevs = θ
    y = P_x(means.T, θ).T
    pyplot.plot(means + [[-1],[1]]*stdevs, y/[[2],[2]], '|-b', alpha=0.3, linewidth=1)
    pyplot.plot([[1],[1]]*means, [[0],[1]]*y, '|-b', alpha=0.3, linewidth=1)
    pyplot.show(block=False)
    pyplot.pause(0.01)
  show(x)

  def log_likelihood(x, θ):
    P_z, means, stdevs = θ
    P_x_given_z = gauss(x, means, stdevs)
    P_x_and_z = P_z*P_x_given_z
    P_x = P_x_and_z.sum(axis=1)
    return np.log(P_x).sum(axis=0)
  log_likelihood1, log_likelihood0 = log_likelihood(x, θ), log_likelihood1
