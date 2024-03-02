import numpy as np
import matplotlib.pyplot as plt

def energy_dist(E):
        #E In units of GeV
        #Parameterise the distribution.

        E_0 = 4.29
        eps = 854
        n = 3.01

        #Energy dist from paper.
        p = ((E_0+E)**(-n))* ((1+ E / eps)**(-1))
    
        return p

def mean(energy,prob):
    E_mean = 0
    for x,i in enumerate(energy):
        E_mean += i *prob[x]
        
    return E_mean

mean_E = []
E_cuts = []

for E_cut in np.linspace(100,600,num=50):
    E_cuts.append(E_cut)
    energy_vals = np.linspace(0.1057,E_cut,10000)
    energy_probs = [energy_dist(x) for x in energy_vals]
    norm_energy_probs = np.multiply(1/(np.sum(energy_probs)),energy_probs)
    mean_E.append(mean(energy_vals,norm_energy_probs))

fig = plt.figure()

plt.scatter(E_cuts,mean_E)

plt.xlabel('E')
plt.ylabel('E_Mean')

plt.show()