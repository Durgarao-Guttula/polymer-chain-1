import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from collections import defaultdict

epsilon = 0.25
sigma = 0.8
sigma6 = sigma**6
sigma12 = sigma**12

T = 1
N = 250

PERM = True

R2s = defaultdict(list)
pol_weights = defaultdict(list)

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

def add_bead(polymer, pol_weight, L, weight3):
	angles = np.arange(0, 2*np.pi, 2/6*np.pi).reshape(6, 1) + (2/6*np.pi)*np.random.rand()
	delta_pos = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
	new_pos = polymer[-1] + delta_pos
	w_l = np.ndarray(6)
	for j in range(0,6):
		w_l[j] = np.exp(-E(polymer,new_pos[j])/T)
	if np.count_nonzero(w_l) == 0:
		print("all options impossible (L={})".format(L))
		R2s[L].append(np.sum(polymer[-1]*polymer[-1]))
		# b_ary = np.array(polymer)
		# plt.plot(b_ary[:,0], b_ary[:,1], 'o-')
		# plt.plot(new_pos[:,0],new_pos[:,1], 'r.')
		# plt.axis('equal')
		# plt.title('All options impossible!')
		# plt.grid(True)
		# plt.show()
		return
	W_l = np.sum(w_l)
	p_l = w_l/W_l
	j = np.random.choice(6, p=p_l)
	polymer.append(new_pos[j])
	pol_weight *= w_l[j]
	L += 1
	if L == 3:
		weight3 = pol_weight # is this correct??!?!
	pol_weights[L].append(pol_weight)
	av_weight = np.mean(pol_weights[L])
	up_lim = 2 * av_weight / weight3
	low_lim = 1.2 * av_weight / weight3

	print("up_lim: {} low_lim: {}".format(up_lim,low_lim))

	#print("num concurrent polymers: {}, pol_weight: {}".format(a, pol_weight))
	if L < N:
		if PERM and pol_weight > up_lim:
			new_polymer1 = polymer[:]
			new_polymer2 = polymer[:]
			# weight3 =
			add_bead(new_polymer1, 0.5*pol_weight, L, weight3)
			add_bead(new_polymer2, 0.5*pol_weight, L, weight3)
		elif PERM and pol_weight < low_lim:
			if np.random.rand() < 0.5:
				add_bead(polymer, 2*pol_weight, L, weight3)
		else:
			add_bead(polymer, pol_weight, L, weight3)
	else:
		print("reached maximum length (L={})".format(L))
		R2s[L].append(np.sum(polymer[-1]*polymer[-1]))
		#plot_polymer(polymer)

################################################

num_runs = 100
for i in range(0,num_runs):
	print("run {} of {}".format(i, num_runs))

	# IC: first two beads
	polymer = []
	polymer.append(np.array([0.0, 0.0]))
	polymer.append(np.array([1.0, 0.0]))
	pol_weight = 1
	L = 2

	# run the simulation
	add_bead(polymer, pol_weight, L, 0)

Ls = []
R2s_mean = []
R2s_std = []
for L, R2vals in sorted(R2s.items()):
	Ls.append(L)
	R2s_mean.append(np.mean(R2vals))
	R2s_std.append(np.std(R2vals))

plt.loglog(Ls, R2s_mean)
#plt.errorbar(Ls, R2s_mean, yerr=R2s_std/np.sqrt(num_runs), fmt='.')
plt.xlabel('$L$')
plt.ylabel('$R^2$')
plt.show()

