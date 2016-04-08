import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from collections import defaultdict
from scipy.optimize import curve_fit
import time
import math
import pickle

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

def add_bead(polymer, pol_weight, L, weight3, use_perm):
	global pop
	global R2s
	#print(pop)

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
	pol_weight *= W_l
	# pol_weight *= 1/(0.75*6)
	# print(max(w_l[j],0.1))
	# if pol_weight < 1e-10:
	# 	print('pol_weight: {}, w_l = {}'.format(pol_weight,max(w_l[j],0.1)))

	L += 1
	if L == 3:
		weight3 = pol_weight
	pol_weights[L].append(pol_weight)
	R2s[L].append(np.sum(polymer[-1]*polymer[-1]))
	av_weight = np.mean(pol_weights[L])
	up_lim = 3.5 * av_weight / weight3
	low_lim = 2 * av_weight / weight3

	if(use_perm):
		print(w_l[j])
		print('L: {} pol_weight: {} av_weight: {} up: {} lo: {}'.format(L, pol_weight, av_weight,up_lim,low_lim))
	if L < N:
		if use_perm and pol_weight > up_lim:
			print("enriching polymer (L={})".format(L))
			pop += 1
			new_polymer1 = polymer[:]
			new_polymer2 = polymer[:]
			add_bead(new_polymer1, 0.5*pol_weight, L, weight3, use_perm)
			add_bead(new_polymer2, 0.5*pol_weight, L, weight3, use_perm)
		elif use_perm and pol_weight < low_lim:
			if np.random.rand() < 0.5:
				add_bead(polymer, 2*pol_weight, L, weight3, use_perm)
			else:
				print("pruning polymer (L={})".format(L))
				pop -= 1
		else:
			add_bead(polymer, pol_weight, L, weight3, use_perm)
	else:
		print("reached maximum length (L={})".format(L))
		#plot_polymer(polymer)
		pop -= 1

def storvar(vardict):
	f = open('Length_weights.txt', 'wb')
	pickle.dump(vardict,f,)
	f.close()
	return



################################################

R2s = defaultdict(list)
pol_weights = defaultdict(list)

for i in range(0,num_runs):
	pop = 1
	# IC: first two beads
	polymer = []
	polymer.append(np.array([0.0, 0.0]))
	polymer.append(np.array([1.0, 0.0]))
	L = 2
	#
	pol_weight = 1.0
	R2s[2].append(1.0)
	pol_weights[2].append(1.0)

	use_perm = False # (len(R2s[N]) > 100)
	
	print("run {} of {} (PERM = {})".format(i, num_runs,use_perm))
	# run the simulation
	add_bead(polymer, pol_weight, L, None, use_perm)

Ls = []
R2s_count = []
R2s_avg = []
av_weights = []
for L, R2vals in sorted(R2s.items()):
	Ls.append(L)
	R2s_count.append(len(R2vals))
	R2s_avg.append(np.average(R2vals,weights=pol_weights[L]))
	av_weights.append(np.mean(pol_weights[L]))

Ls = np.array(Ls)
R2s_avg = np.array(R2s_avg)

def fitfunc(N, a):
	return a*(N-1)**1.5
popt, pcov = curve_fit(fitfunc, Ls, R2s_avg)
a_fit = popt[0]
print("fitted a = {}".format(a_fit))

plt.figure()
plt.semilogy(Ls, R2s_count, '.', Ls, av_weights, '.')
plt.xlabel('$L$')
plt.legend(['count','av_weight'])
plt.show(block=False)

plt.figure()
plt.loglog(Ls, R2s_avg, '.', Ls, fitfunc(Ls, a_fit), '-')
plt.hold(True)
plt.xlim(xmin=2) #match book
plt.ylim(ymin=1)
plt.xlabel('$N$')
plt.ylabel('$R^2$')
plt.show()

vardict = {'L': Ls, 'R2': R2s_avg}
storvar(vardict)
