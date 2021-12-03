import numpy as np
np.random.seed(0)

z_length, x_length = 2, 10000
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
  x = np.choose(z, samples)
  return x, x[z==0], x[z==1]
x, x1, x2 = x(target_θ)

from matplotlib import pyplot
pyplot.clf()
pyplot.xlabel('$x$')
pyplot.ylabel('Count of $x$')
pyplot.ylim(0, 1200)
bin_size = 0.5
bins = np.arange(x.min(), x.max(), bin_size)

def a():
  pyplot.hist(x, bins=bins, color='black', alpha=0.4, label='Samples')
  return 'Histogram of the observable data'
#title = a()

def b():
  pyplot.hist([x1, x2], bins=bins, color=['red', 'blue'], alpha=0.4, stacked=True, label='Samples')
  pyplot.text(8, 200, '1', horizontalalignment='center', size=30, color='white')
  pyplot.text(13, 100, '2', horizontalalignment='center', size=30, color='white')
  return 'Histogram of the observable data and the latent data'
title = b()

pyplot.title(title)
pyplot.savefig(f'{title}.svg')
