def show_inference(p_x, x, θ, target_θ):
  from matplotlib import pyplot
  pyplot.clf()
  pyplot.xlabel('$x$')
  pyplot.ylabel('$p(x|θ)$')
  bin_size = 0.5
  import numpy as np
  bins = np.arange(x.min(), x.max(), bin_size)
  weights = np.ones_like(x)/len(x)/bin_size
  pyplot.hist(x, bins=bins, weights=weights, color='g', alpha=0.3, label='Samples')

  x = np.linspace(x.min(), x.max(), 1000)
  y = p_x(x, target_θ)
  pyplot.plot(x, y, 'g-', alpha=0.3, label='Target (real) distribution $p(x)$')

  y = p_x(x, θ)
  pyplot.plot(x, y, 'b-', label='Inferred (predicted) distribution $p(x|θ)$')

  p_z, means, stdevs = θ
  means, stdevs = means.T, stdevs.T
  y = p_x(means, θ)
  pyplot.plot(means + [[-1],[1]]*stdevs, y/[[2],[2]], '|-b', alpha=0.3, linewidth=1)
  pyplot.plot([[1],[1]]*means, [[0],[1]]*y, '|-b', alpha=0.3, linewidth=1)

  pyplot.legend(loc='upper right')
  pyplot.show(block=False)
  pyplot.pause(0.01)
