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
N = 250

PERM = False

@jit
def lennard_jones(r):
	r2 = np.sum(r*r)
	r6 = r2*r2*r2
	r8 = r6*r2
	r12 = r6*r6
	r14 = r8*r6
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
	global a
	global R2s
	#print(a)

	angles = np.arange(0, 2*np.pi, 2/6*np.pi).reshape(6, 1) + (2/6*np.pi)*np.random.rand()
	delta_pos = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
	new_pos = polymer[-1] + delta_pos
	w_l = np.ndarray(6)
	for j in range(0,6):
		w_l[j] = np.exp(-E(polymer,new_pos[j])/T)
	if np.count_nonzero(w_l) == 0:
		#print("all options impossible (L={})".format(L))
		# b_ary = np.array(polymer)
		# plt.plot(b_ary[:,0], b_ary[:,1], 'o-')
		# plt.plot(new_pos[:,0],new_pos[:,1], 'r.')
		# plt.axis('equal')
		# plt.title('All options impossible!')
		# plt.grid(True)
		# plt.show()
		a -= 1
		return
	W_l = np.sum(w_l)
	p_l = w_l/W_l
	j = np.random.choice(6, p=p_l)
	polymer.append(new_pos[j])
	pol_weight *= w_l[j]
	# pol_weight *= 1/(0.75*6)
	# print(max(w_l[j],0.1))
	# if pol_weight < 1e-10:
	# 	print('pol_weight: {}, w_l = {}'.format(pol_weight,max(w_l[j],0.1)))

	L += 1
	if L == 3:
		weight3 = pol_weight # is this correct??!?!
	pol_weights[L].append(pol_weight)
	R2s[L].append(np.sum(polymer[-1]*polymer[-1]))
	av_weight = np.mean(pol_weights[L])
	up_lim = 3.5 * av_weight / weight3
	low_lim = 2 * av_weight / weight3

	if(use_perm):
		print(w_l[j])
		print('L: {} pol_weight: {} av_weight: {} up: {} lo: {}'.format(L, pol_weight, av_weight,up_lim,low_lim))
	#input()
	if L < N:
		if use_perm and pol_weight > up_lim:
			#print("enriching polymer (L={})".format(L))
			a += 1
			new_polymer1 = polymer[:]
			new_polymer2 = polymer[:]
			add_bead(new_polymer1, 0.5*pol_weight, L, weight3, use_perm)
			add_bead(new_polymer2, 0.5*pol_weight, L, weight3, use_perm)
		elif use_perm and pol_weight < low_lim:
			if np.random.rand() < 0.5:
				add_bead(polymer, 2*pol_weight, L, weight3, use_perm)
			else:
				#print("pruning polymer (L={})".format(L))
				a -= 1
		else:
			add_bead(polymer, pol_weight, L, weight3, use_perm)
	else:
		#print("reached maximum length (L={})".format(L))
		#plot_polymer(polymer)
		a -= 1

def storvar(vardict):
	f = open('Length_weights.txt', 'wb')
	pickle.dump(vardict,f,)
	f.close()
	return



################################################

R2s = defaultdict(list)
pol_weights = defaultdict(list)

num_runs = 30

for i in range(0,num_runs):
	a = 1
	print("run {} of {}".format(i, num_runs))
	# IC: first two beads
	polymer = []
	polymer.append(np.array([0.0, 0.0]))
	polymer.append(np.array([1.0, 0.0]))
	pol_weight = 1
	R2s[2].append(1)
	L = 2

	use_perm = (len(R2s[N]) > 100)
	print(use_perm)
	# run the simulation
	add_bead(polymer, pol_weight, L, None, use_perm)

plt.figure()
Ls = []
R2s_count = []
av_weights = []
for L, R2vals in sorted(R2s.items()):
	Ls.append(L)
	R2s_count.append(len(R2vals))
	av_weights.append(np.mean(pol_weights[L]))
plt.semilogy(Ls, R2s_count, '.', Ls, av_weights, '.')
plt.xlabel('$L$')
plt.legend(['count','av_weight'])
plt.show(block=False)


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

print('Ls= {} R2smean= {}'.format(Ls, R2s_mean))

# popt, pcov = curve_fit(fitfunc, Ls, R2s_mean, sigma=R2s_std)
# a_fit = popt[0]
# print("fitted a = {}".format(a_fit))

vardict = {'L': Ls, 'R2': R2s_mean}
storvar(vardict)

plt.loglog(Ls, R2s_mean, '.')
plt.hold(True)
plt.xlim(xmin=2) #match book
plt.ylim(ymin=1)
plt.xlabel('$N$')
plt.ylabel('$R^2$')
plt.show()
