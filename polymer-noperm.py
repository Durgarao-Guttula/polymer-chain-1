import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from collections import defaultdict
from scipy.optimize import curve_fit
import math

epsilon = 0.25
sigma = 0.8
sigma6 = sigma**6
sigma12 = sigma**12

T = 1
N = 150
num_runs = 100

@jit
def lennard_jones(r):
	r2 = np.sum(r*r)
	r6 = r2*r2*r2
	r12 = r6*r6
	Vij = 4 * epsilon * ( (sigma12/r12) - (sigma6/r6) )
	return Vij

def E(polymer, new_bead):
	Vj = 0
	for bead in polymer:
		Vij = lennard_jones(new_bead-bead)
		Vj += Vij
	return Vj

def plot_polymer(polymer):
	b_ary = np.array(polymer)
	plt.plot(b_ary[:,0], b_ary[:,1], 'o-')
	plt.axis('equal')
	plt.title('Polymer chain')
	plt.grid(True)
	plt.show()

def add_bead(polymer, L):
	global R2s

	angle_offset = np.random.uniform(0, 2/6*np.pi)
	angles = np.arange(0, 2*np.pi, 2/6*np.pi).reshape(6, 1) + angle_offset
	delta_pos = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
	new_pos = polymer[-1] + delta_pos
	w_l = np.ndarray(6)
	for j in range(0,6):
		w_l[j] = np.exp(-E(polymer,new_pos[j])/T)
	W_l = np.sum(w_l)
	if W_l > 0:
		p_l = w_l/W_l
		j = np.random.choice(6, p=p_l)
	else:
		print("all options impossible, choosing at random")
		j = np.random.choice(6)
	polymer.append(new_pos[j])

	L += 1
	R2s[L].append(np.sum(polymer[-1]*polymer[-1]))
	
	if L < N:
		add_bead(polymer, L)
	else:
		print("reached maximum length (L={})".format(L))
		#plot_polymer(polymer)

################################################

R2s = defaultdict(list)

for i in range(0,num_runs):
	pop = 1
	# IC: first two beads
	polymer = []
	polymer.append(np.array([0.0, 0.0]))
	polymer.append(np.array([1.0, 0.0]))
	L = 2
	#
	R2s[2].append(1.0)
	
	print("run {} of {}".format(i, num_runs))
	# run the simulation
	add_bead(polymer, L)

plt.figure()
Ls = []
R2s_mean = []
R2s_std = []
for L, R2vals in sorted(R2s.items()):
	Ls.append(L)
	R2s_mean.append(np.mean(R2vals))
	R2s_std.append(np.std(R2vals))

Ls = np.array(Ls)
R2s_mean = np.array(R2s_mean)

def fitfunc(N, a):
	return a*(N-1)**1.5
popt, pcov = curve_fit(fitfunc, Ls, R2s_mean)
a_fit = popt[0]
print("fitted a = {}".format(a_fit))

plt.loglog(Ls, R2s_mean, '.', Ls, fitfunc(Ls, a_fit), '-')
plt.hold(True)
plt.xlim(xmin=2) #match book
plt.ylim(ymin=1)
plt.xlabel('$N$')
plt.ylabel('$R^2$')
plt.show()
