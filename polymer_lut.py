import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from collections import defaultdict
import copy

epsilon = 0.25
sigma = 0.8
sigma6 = sigma**6
sigma12 = sigma**12

T = 1
N = 40

PERM = True

R2s = defaultdict(list)

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
	# print('Vj=',Vj)
	return Vj

def plot_polymer(polymer):
	b_ary = np.array(polymer)
	plt.plot(b_ary[:,0], b_ary[:,1], 'o-')
	plt.axis('equal')
	plt.title('Polymer chain')
	plt.grid(True)
	plt.show()

def add_bead3(polymer, pol_weight, L):
	angles = np.arange(0, 2*np.pi, 2/6*np.pi).reshape(6, 1) + (2/6*np.pi)*np.random.rand()
	delta_pos = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
	new_pos = polymer[-1] + delta_pos
	w_l = np.ndarray(6)
	for j in range(0,6):
		w_l[j] = np.exp(-E(polymer,new_pos[j])/T)

	W_l = np.sum(w_l)
	p_l = w_l/W_l
	j = np.random.choice(6, p=p_l)
	
	polymer.append(new_pos[j])
	pol_weight1 = copy.copy(pol_weight)
	pol_weight *= w_l[j]
	sum_pol_weight = pol_weight1 + pol_weight
	R2s[L].append(np.sum(polymer[-1]*polymer[-1]))

	return polymer, pol_weight, pol_weight1, sum_pol_weight,L+1

def add_bead(polymer, pol_weight, pol_weight1, pol_weight2, pol_weight3, sum_pol_weight, L):
	angles = np.arange(0, 2*np.pi, 2/6*np.pi).reshape(6, 1) + (2/6*np.pi)*np.random.rand()
	
	delta_pos = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
	
	new_pos = polymer[-1] + delta_pos
	
	w_l = np.ndarray(6)
	for j in range(0,6):
		w_l[j] = np.exp(-E(polymer,new_pos[j])/T)
	if np.count_nonzero(w_l) == 0:
		print("all options impossible")
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
	pol_weight_pre = copy.copy(pol_weight)
	pol_weight *= w_l[j]
	########This if function is annoying beacuse i think it slows the system down considerably###########
	if pol_weight3 == 0:
		pol_weight3 = copy.copy(pol_weight)
	#############################################

	sum_pol_weight += pol_weight
	avg_pol_weight = sum_pol_weight/L

	up_lim = 2*avg_pol_weight/((pol_weight1+pol_weight2+pol_weight3)/3)
	low_lim = 1.2*avg_pol_weight/((pol_weight1+pol_weight2+pol_weight3)/3)
	print('up_lim={0} low_lim={1}'.format(up_lim, low_lim))
	
	

	# print('polymer weight =',pol_weight)
	# print('pol_weight3=' ,pol_weight3)
	# print()
	# print()
	# print('up_lim=',up_lim)
	# print('low_lim=',low_lim)
	# print('average pol weight= ', avg_pol_weight)
	# print('weight3= ', ((pol_weight1+pol_weight2+pol_weight3)/3))
	# print('pol_weight > up_lim',pol_weight>up_lim)
	# print('pol_weight < low_lim',pol_weight<low_lim)
	# input()

	if L < N:
		if PERM and pol_weight > up_lim:
			new_polymer1 = polymer[:]
			new_polymer2 = polymer[:]
			add_bead(new_polymer1, 0.5*pol_weight, 0.5*pol_weight_pre, copy.copy(0.5*pol_weight), 0 , sum_pol_weight, L+1)
			add_bead(new_polymer2, 0.5*pol_weight, 0.5*pol_weight_pre, copy.copy(0.5*pol_weight), 0 , sum_pol_weight, L+1)
		elif PERM and pol_weight < low_lim:
			if np.random.rand() < 0.5:
				add_bead(polymer, 2*pol_weight, 2*pol_weight_pre , copy.copy(2*pol_weight), 0 , sum_pol_weight, L+1)
		else:
			add_bead(polymer, pol_weight, pol_weight1, pol_weight2, pol_weight3, sum_pol_weight, L+1)
	else:
		print("reached maximum length")
		R2s[L].append(np.sum(polymer[-1]*polymer[-1]))
		# plot_polymer(polymer)

################################################

num_runs = 10
for i in range(0,num_runs):
	print("run {} of {}".format(i, num_runs))

	# IC: first two beads
	polymer = []
	polymer.append(np.array([0.0, 0.0]))
	polymer.append(np.array([1.0, 0.0]))
	print('polymer= ', polymer)
	pol_weight = 1
	L = 2
	a=0
	pol_weight3 = 0

	# run the simulation
	[polymer, pol_weight, pol_weight1, sum_pol_weight, L] = add_bead3(polymer,  pol_weight, L)
	pol_weight2 = copy.copy(pol_weight)
	add_bead(polymer, pol_weight, pol_weight1, pol_weight2, pol_weight3, sum_pol_weight, L)

Ls = []
R2s_mean = []
R2s_std = []
for L, R2vals in sorted(R2s.items()):
	Ls.append(L)
	R2s_mean.append(np.mean(R2vals))
	R2s_std.append(np.std(R2vals))

plt.errorbar(Ls, R2s_mean, yerr=R2s_std, fmt='.')
plt.xlabel('$L$')
plt.ylabel('$R^2$')
plt.show()

