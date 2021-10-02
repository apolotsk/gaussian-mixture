''' Example of Gaussian mixture with 2 normal distribution.
http://courses.cs.washington.edu/courses/cse312/11wi/slides/12em.pdf '''
from math import pi, sqrt, exp, log, hypot

gauss = lambda x, m, v: 1/sqrt(2*pi*v) * exp(-(x-m)**2/(2*v))

epsilon = 10**-6

observed_data = -6, -5, -4, 0, 4, 5, 6
# Initial (random) values of parameters.
weights, m, v = (.5, .5), (-20, 6), (1, 1)


def plot_likelihood():
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  import matplotlib.pyplot as plt
  import numpy as np
  
  def log_likelihood(m):
    a = 1
    for i, x in enumerate(observed_data):
      for z in range(2):
        a *= ( weights[z] * gauss(x, m[z], v[z]) ) ** expectations_of_groups[z][i]
    return log(a) if a else float('-inf')
  
  range_x = np.arange(-20, 20, 1)
  #TODO: delete `y`. We use 1D gaussians.
  range_y = np.arange(-20, 20, 1)
  # Pringle surface
  z = [[log_likelihood((x, y)) for x in range_x] for y in range_y]
  x, y = np.meshgrid(range_x, range_y)
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  plt.show()
  

distance = float('inf')
while True:
  # Compute a probability distribution over completions of missing data given the current model.
  probs_of_observation_and_group = [ [ gauss(x, m[i], v[i])*weights[i] for i in range(2) ] for x in observed_data ]
  probs_of_group_given_observation = [ [ p/sum(probs_of_group) for p in probs_of_group ] for probs_of_group in probs_of_observation_and_group ]
  expectations_of_groups = list(zip(*probs_of_group_given_observation))
  
  # Reestimate the model parameters using the completions.
  # Maximum likelihood estimation of the expected log-likelihood of the data.
  sum_of_obserations_per_group = [ sum(observed_data[i]*e[i] for i in range(len(e))) for e in expectations_of_groups ]
  observation_count_per_group = [ sum(e) for e in expectations_of_groups ]
  m = [ sum_of_obserations_per_group[i]/observation_count_per_group[i] for i in range(2) ]
  # TODO: update `weights` and `v`.
  print(*m)
  plot_likelihood()

  # If converged then break.
  distance2 = hypot(*m)
  if abs(distance-distance2)<epsilon: break
  distance = distance2
