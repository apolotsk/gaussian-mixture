"""
Gaussian mixture with 2 normal distribution.
http://courses.cs.washington.edu/courses/cse312/11wi/slides/12em.pdf
"""
import numpy as np
np.random.seed(1)

def gauss(x, mean, variance):
  from numpy import pi, sqrt, exp
  return 1/sqrt(2*pi*variance) * exp(-(x-mean)**2/(2*variance))

def true_parameters():
  P_z0, mean0, variance0 = 0.7, 8.0, 2.0
  P_z1, mean1, variance1 = (1-P_z0), 13.0, 1.0
  return (P_z0, mean0, variance0), (P_z1, mean1, variance1)
true_parameters = true_parameters()

def observed_data(true_parameters):
  (P_z0, mean0, variance0), (P_z1, mean1, variance1) = true_parameters
  def sample():
    import numpy as np
    if np.random.rand()<P_z0: return np.random.normal(mean0, variance0**0.5)
    else: return np.random.normal(mean1, variance1**0.5)
  return np.array([sample() for _ in range(1000)])
observed_data = observed_data(true_parameters)

def parameters():
  from numpy.random import rand
  P_z0, mean0, variance0 = rand()*0.5+0.25, rand()*20-10, rand()*10+1
  P_z1, mean1, variance1 = (1-P_z0), rand()*20-10, rand()*10+1
  return (P_z0, mean0, variance0), (P_z1, mean1, variance1)
parameters = parameters()

def P_xi_given_zj(x, mean, variance):
  return gauss(x, mean, variance)
def P_xi_and_zj(x, P_z, mean, variance):
  return P_z*P_xi_given_zj(x, mean, variance)
def P_xi(x, parameters):
  return sum(P_xi_and_zj(x, P_z, mean, variance) for P_z, mean, variance in parameters)

log_likelihood1, log_likelihood0 = float('inf'), float('-inf')
while abs(log_likelihood1-log_likelihood0)>1e-6:
  def expectation_step(observed_data, parameters):
    P_x_and_z = [[P_xi_and_zj(x, P_z, mean, variance) for P_z, mean, variance in parameters] for x in observed_data]
    P_x = [sum(P_xi_and_z) for P_xi_and_z in P_x_and_z]
    P_z_given_x = [[P_xi_and_zj/P_xi for P_xi_and_zj in P_xi_and_z] for P_xi_and_z, P_xi in zip(P_x_and_z, P_x)]
    return P_z_given_x
  P_z_given_x = expectation_step(observed_data, parameters)

  def maximization_step(P_z_given_x, observed_data):
    P_z_given_x = list(zip(*P_z_given_x))
    def update(P_zi_given_x):
      P_zi = sum(P_zi_given_x)/len(observed_data)
      mean = sum(P_zi_given_xj*x for P_zi_given_xj, x in zip(P_zi_given_x, observed_data))/sum(P_zi_given_x)
      variance = sum(P_zi_given_xj*(x-mean)**2 for P_zi_given_xj, x in zip(P_zi_given_x, observed_data))/sum(P_zi_given_x)
      return P_zi, mean, variance
    parameters = [update(P_zi_given_x) for P_zi_given_x in P_z_given_x]
    return parameters
  parameters = maximization_step(P_z_given_x, observed_data)

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
    x = np.linspace(x.min(), x.max(), 1000)
    y = P_xi(x, true_parameters)
    pyplot.plot(x, y, 'g-', alpha=0.3, label='P(x)')
    y = P_xi(x, parameters)
    pyplot.plot(x, y, 'b-', label='P(x|θ)')
    for P_zi, mean, variance in parameters:
      y = P_xi(mean, parameters)
      pyplot.plot([mean-variance**0.5, mean+variance**0.5], [y/2, y/2], '|-b', alpha=0.3, linewidth=1)
      pyplot.plot([mean, mean], [0, y], '|-b', alpha=0.3, linewidth=1)
    pyplot.show(block=False)
    pyplot.pause(0.01)
  show(observed_data)

  def log_likelihood(observed_data, parameters):
    P_x_and_z = [[P_xi_and_zj(x, P_z, mean, variance) for P_z, mean, variance in parameters] for x in observed_data]
    from math import log
    return sum([log(sum(P_xi_and_z)) for P_xi_and_z in P_x_and_z])
  log_likelihood1, log_likelihood0 = log_likelihood(observed_data, parameters), log_likelihood1
