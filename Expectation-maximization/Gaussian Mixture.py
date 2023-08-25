import numpy as np
np.random.seed(0)

def gauss(x, mean, stdev):
  from numpy import pi, sqrt, exp
  return 1/sqrt(2*pi*stdev**2) * exp(-(x-mean)**2/(2*stdev**2))

def target_θ():
  p_z = np.expand_dims([0.3, 0.7], axis=1)
  print(f'The real probability probability of selecting Gaussian 1 is {p_z[0,0]:0.2f} and Gaussian 2 is {p_z[1,0]:0.2f}.')
  means = np.expand_dims([13.0, 8.0], axis=1)
  print(f'The real mean of Gaussian 1 is {means[0,0]:0.2f} and Gaussian 1 is {means[1,0]:0.2f}.')
  stdevs = np.expand_dims([1.0, 1.4], axis=1)
  print(f'The real standard deviation of Gaussian 1 is {stdevs[0,0]:0.2f} and Gaussian 1 is {stdevs[1,0]:0.2f}.')
  return p_z, means, stdevs
target_θ = target_θ()

z_length, x_length = 2, 1000
def x(target_θ):
  p_z, means, stdevs = target_θ
  z = np.random.choice(z_length, size=x_length, p=p_z[:,0])
  print(f'The real sample count of Gaussian 1 is {(z==0).sum()} and Gaussian 2 is {(z==1).sum()}.')
  samples = np.random.normal(means, stdevs, [z_length, x_length])
  x = np.choose(z, samples)
  from show import show_observable_and_latent_data, show_observable_data
  show_observable_and_latent_data(x, z)
  show_observable_data(x)
  return x
x = x(target_θ)

def find_parameters(x):
  print()
  print('Finding the parameters given only the samples.')

  def θ():
    p_z = np.ones([z_length, 1])/z_length
    means = np.random.rand(z_length, 1)*20
    stdevs = np.random.rand(z_length, 1)*5
    return p_z, means, stdevs
  θ = θ()

  from show import Animation
  animation = Animation()

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
      p_x_and_z = p_z_given_x/len(x)
      p_z = p_x_and_z.sum(axis=1, keepdims=True)
      p_x_given_z = p_x_and_z/p_z
      means = (p_x_given_z * x).sum(axis=1, keepdims=True)
      stdevs = np.sqrt((p_x_given_z * (x-means)**2).sum(axis=1, keepdims=True))
      return p_z, means, stdevs
    θ = maximization_step(p_z_given_x, x)

    def p_x(x, θ):
      p_z, means, stdevs = θ
      p_x_given_z = gauss(x, means, stdevs)
      p_z_and_x = p_z*p_x_given_z
      p_x = p_z_and_x.sum(axis=0)
      return p_x

    from show import show_inference
    show_inference(p_x, x, θ, target_θ)
    animation.add_plot()

    def log_likelihood(x, θ):
      return np.log(p_x(x, θ)).mean(axis=0)
    log_likelihood1, log_likelihood0 = log_likelihood(x, θ), log_likelihood1

  animation.save('Find parameters.gif')
  return θ
θ = find_parameters(x)

p_z, means, stdevs = θ
print(f'The predicted probability probability of selecting Gaussian 1 is {p_z[0,0]:0.2f} and Gaussian 2 is {p_z[1,0]:0.2f}.')
print(f'The predicted mean of Gaussian 1 is {means[0,0]:0.2f} and Gaussian 2 is {means[1,0]:0.2f}.')
print(f'The predicted standard deviation of Gaussian 1 is {stdevs[0,0]:0.2f} and Gaussian 2 is {stdevs[1,0]:0.2f}.')

p_z_and_x = p_z * gauss(x, means, stdevs)
z = p_z_and_x/p_z_and_x.sum(axis=0)
print(f'The predicted sample count of Gaussian 1 is {z[0].sum():.1f} and Gaussian 2 is {z[1].sum():.1f}.')
