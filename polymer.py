import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from collections import defaultdict
from scipy.optimize import curve_fit
import time
import math
import sys

epsilon = 0.25
sigma = 0.8
sigma6 = sigma**6
sigma12 = sigma**12

T = 1
N = 250
num_runs = 1000

DEBUG = False

@jit
def lennard_jones(r):
	r2 = np.sum(r*r)
	r6 = r2*r2*r2
	r12 = r6*r6
	Vij = 4 * epsilon * ( (sigma12/r12) - (sigma6/r6) )
	return Vij

@jit
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

def add_bead(polymer, pol_weight, weight3, use_perm):
	global pop
	print("pop: {}    ".format(pop), end="\r", file=sys.stderr)

	angle_offset = np.random.uniform(0, 2/6*np.pi)
	angles = np.arange(0, 2*np.pi, 2/6*np.pi).reshape(6, 1) + angle_offset
	delta_pos = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
	new_pos = polymer[-1] + delta_pos
	w_l = np.ndarray(6)
	for j in range(0,6):
		w_l[j] = np.exp(-E(polymer,new_pos[j])/T)
	W_l = np.sum(w_l)
	if W_l > 0:
		# choose one of the six new bead positions,
		# with probability based on the weight (energy) of the new position
		p_l = w_l/W_l
		j = np.random.choice(6, p=p_l)
	else:
		# if all weights are zero (caused by high energy/curling up),
		# there is no use continuing with this polymer,
		# since it (and all offspring) will have a pol_weight of 0,
		# so will not be used later on because of the weighing process
		if DEBUG: print("all options impossible, aborting")
		pop -= 1
		return
	polymer.append(new_pos[j])
	pol_weight *= W_l
	pol_weight *= 1/(0.75*6)

	pol_weights[len(polymer)].append(pol_weight)
	R2s[len(polymer)].append(np.sum(polymer[-1]*polymer[-1]))

	if len(polymer) == 3:
		weight3 = pol_weight

	av_weight = np.mean(pol_weights[len(polymer)])
	up_lim = 3 * av_weight / weight3
	low_lim = 1.2 * av_weight / weight3

	if len(polymer) < N:
		if use_perm and pol_weight > up_lim:
			if DEBUG: print("enriching polymer (L={})".format(len(polymer)))
			pop += 1
			new_polymer1 = polymer[:]
			new_polymer2 = polymer[:]
			add_bead(new_polymer1, 0.5*pol_weight, weight3, use_perm)
			add_bead(new_polymer2, 0.5*pol_weight, weight3, use_perm)
		elif use_perm and pol_weight < low_lim:
			if np.random.rand() < 0.5:
				add_bead(polymer, 2*pol_weight, weight3, use_perm)
			else:
				if DEBUG: print("pruning polymer (L={})".format(len(polymer)))
				pop -= 1
		else:
			add_bead(polymer, pol_weight, weight3, use_perm)
	else:
		if DEBUG: print("reached maximum length (L={})".format(len(polymer)))
		pop -= 1
		#plot_polymer(polymer)



################################################

R2s = defaultdict(list)
pol_weights = defaultdict(list)

for i in range(0,num_runs):
	pop = 1
	# IC: first two beads
	polymer = []
	polymer.append(np.array([0.0, 0.0]))
	polymer.append(np.array([1.0, 0.0]))
	pol_weight = 1.0
	# only use PERM after a few runs, 
	# when the average weight has stabilized
	use_perm = (len(pol_weights[N]) > 100)
	
	print("run {} of {} (PERM = {})".format(i, num_runs,use_perm))
	# run the simulation
	add_bead(polymer, pol_weight, None, use_perm)
	print ("\n\r", end="", file=sys.stderr)

Ls = []
R2s_count = []
R2s_avg = []
R2s_std = []
av_weights = []
for L in sorted(R2s.keys()):
	Ls.append(L)
	R2s_count.append(len(R2s[L]))
	average = np.average(R2s[L],weights=pol_weights[L])
	variance = np.average((R2s[L]-average)**2, weights=pol_weights[L])/len(R2s[L])
	R2s_avg.append(average)
	R2s_std.append(np.sqrt(variance))
	av_weights.append(np.mean(pol_weights[L]))

Ls = np.array(Ls)
R2s_avg = np.array(R2s_avg)
R2s_std = np.array(R2s_std)

def fitfunc(N, a):
	return a*(N-1)**1.5
popt, pcov = curve_fit(fitfunc, Ls, R2s_avg)
a_fit = popt[0]
print("fitted a = {}".format(a_fit))

plt.figure()
plt.loglog(Ls, fitfunc(Ls, a_fit), '-')
plt.hold(True)
plt.errorbar(Ls, R2s_avg, R2s_std, fmt='.')
plt.xlim(xmin=2) #match book
plt.ylim(ymin=1)
plt.xlabel('$N$')
plt.ylabel('$R^2$')


plt.plot(Ls, R2s_count, 'ko')
plt.show()