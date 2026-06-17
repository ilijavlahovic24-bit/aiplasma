import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num=50)
test = np.linspace(0, 2)

# Proper uniform distribution over our mu range
uniform_dist = sts.uniform.pdf(mu, loc=1.65, scale=0.15)  # 1.8-1.65=0.15

# Beta distribution (already properly normalized)
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.15)  # scale should match range

plt.plot(mu, beta_dist, label='Beta Dist')
plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.xlabel(r"Value of $\mu$ in meters")
plt.ylabel("Probability density")
plt.legend()
plt.savefig('distributions_plot.png')
plt.clf()
print("Plot saved as 'distributions_plot.png'")

def likelihood_func(datum, mu):
  likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1) #Note that mu here is an array of values, so the output is also an array!
  return likelihood_out/likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title(r"Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel(r"Value of $\mu$")
plt.savefig('likelihood_plot.png')
print("Plot saved as 'likelihood_plot.png'")
plt.clf()