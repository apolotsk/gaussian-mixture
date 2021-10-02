"""
Gaussian mixture with 2 normal distribution.
http://courses.cs.washington.edu/courses/cse312/11wi/slides/12em.pdf
"""

import numpy as np
np.random.seed(1)

def true_parameters():
  p_z0, mean0, variance0 = 0.7, 8.0, 2.0
  p_z1, mean1, variance1 = (1-p_z0), 13.0, 1.0
  return (p_z0, mean0, variance0), (p_z1, mean1, variance1)
true_parameters = true_parameters()

def observed_data(true_parameters):
  (p_z0, mean0, variance0), (p_z1, mean1, variance1) = true_parameters
  def sample():
    import numpy as np
    if np.random.rand()<p_z0: return np.random.normal(mean0, variance0**0.5)
    else: return np.random.normal(mean1, variance1**0.5)
  return [sample() for _ in range(1000)]
observed_data = observed_data(true_parameters)

def parameters():
  from numpy.random import rand
  p_z0, mean0, variance0 = rand()*0.5+0.25, rand()*20-10, rand()*10+1
  p_z1, mean1, variance1 = (1-p_z0), rand()*20-10, rand()*10+1
  return (p_z0, mean0, variance0), (p_z1, mean1, variance1)
parameters = parameters()

def gauss(x, mean, variance):
  from numpy import pi, sqrt, exp
  return 1/sqrt(2*pi*variance) * exp(-(x-mean)**2/(2*variance))
def p_xi_given_zj(x, mean, variance):
  return gauss(x, mean, variance)
def p_xi_and_zj(x, p_z, mean, variance):
  return p_z*p_xi_given_zj(x, mean, variance)
def p_xi(x, parameters):
  return sum(p_xi_and_zj(x, p_z, mean, variance) for p_z, mean, variance in parameters)

def show(block=True):
  import numpy as np
  from matplotlib import pyplot
  pyplot.clf()
  pyplot.xlabel('x')
  pyplot.ylabel('P(x|θ)')
  x = np.linspace(0, 20, 1000)
  bin_size = 0.5
  bins = np.arange(x.min(), x.max(), bin_size)
  weights = np.ones_like(observed_data)/len(observed_data)/bin_size
  pyplot.hist(observed_data, bins=bins, weights=weights, color='r', alpha=0.3, label='Samples')
  y = p_xi(x, true_parameters)
  pyplot.plot(x, y, 'g-', alpha=0.3, label='P(x)')
  y = p_xi(x, parameters)
  pyplot.plot(x, y, 'b-', label='P(x|θ)')
  for p_zi, mean, variance in parameters:
    y = p_xi(mean, parameters)
    pyplot.plot([mean-variance**0.5, mean+variance**0.5], [y/2, y/2], '|-b', alpha=0.3, linewidth=1)
    pyplot.plot([mean, mean], [0, y], '|-b', alpha=0.3, linewidth=1)
  pyplot.show(block=block)
  pyplot.pause(0.01)

log_likelihood1, log_likelihood0 = float('inf'), float('-inf')
while abs(log_likelihood1-log_likelihood0)>1e-6:
  def expectation_step(observed_data, parameters):
    p_x_and_z = [[p_xi_and_zj(x, p_z, mean, variance) for p_z, mean, variance in parameters] for x in observed_data]
    p_x = [sum(p_xi_and_z) for p_xi_and_z in p_x_and_z]
    p_z_given_x = [[p_xi_and_zj/p_xi for p_xi_and_zj in p_xi_and_z] for p_xi_and_z, p_xi in zip(p_x_and_z, p_x)]
    return p_z_given_x
  p_z_given_x = expectation_step(observed_data, parameters)

  def maximization_step(p_z_given_x, observed_data):
    p_z_given_x = list(zip(*p_z_given_x))
    def update(p_zi_given_x):
      p_zi = sum(p_zi_given_x)/len(observed_data)
      mean = sum(p_zi_given_xj*x for p_zi_given_xj, x in zip(p_zi_given_x, observed_data))/sum(p_zi_given_x)
      variance = sum(p_zi_given_xj*(x-mean)**2 for p_zi_given_xj, x in zip(p_zi_given_x, observed_data))/sum(p_zi_given_x)
      return p_zi, mean, variance
    parameters = [update(p_zi_given_x) for p_zi_given_x in p_z_given_x]
    return parameters
  parameters = maximization_step(p_z_given_x, observed_data)
  show(block=False)

  def log_likelihood(observed_data, parameters):
    p_x_and_z = [[p_xi_and_zj(x, p_z, mean, variance) for p_z, mean, variance in parameters] for x in observed_data]
    from math import log
    return sum([log(sum(p_xi_and_z)) for p_xi_and_z in p_x_and_z])
  log_likelihood1, log_likelihood0 = log_likelihood(observed_data, parameters), log_likelihood1
  print(log_likelihood1)
print(parameters)
show()
