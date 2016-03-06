import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit
def lennard_jones(r):
	r2 = np.sum(r*r)
	r6 = r2*r2*r2
	r8 = r6*r2
	r12 = r6*r6
	r14 = r8*r6
	Fij = r * 4 * ( 12*(1/r14) -6*(1/r8) )
	Vij = 4 * ( (1/r12) - (1/r6) )
	return Fij, Vij

def E(beads, new_bead):
	Vj = 0
	for bead in beads:
		Fij, Vij = lennard_jones(new_bead-bead)
		Vj += Vij
	return Vj

def plot_beads(beads):
	b_ary = np.array(beads)
	plt.plot(b_ary[:,0], b_ary[:,1], 'o-')
	plt.title('Polymer chain')
	plt.grid(True)
	plt.show()


def add_bead(beads, T):
	angles = np.arange(0, 2*np.pi, 2/6*np.pi).reshape(6, 1) + (2/6*np.pi)*np.random.rand()
	delta_pos = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
	new_pos = beads[-1] + delta_pos
	w_l = np.ndarray(6)
	for j in range(0,6):
		w_l[j] = np.exp(-E(beads,new_pos[j])/T)
	#debug
	if(np.count_nonzero(w_l)==0):
		print(w_l)
		for j in range(0,6):
			print('E=',E(beads,new_pos[j]))
			print('w=',np.exp(-E(beads,new_pos[j])/T))
		b_ary = np.array(beads)
		plt.plot(b_ary[:,0], b_ary[:,1], 'o-')
		plt.plot(new_pos[:,0],new_pos[:,1], 'r.')
		plt.title('All options impossible!')
		plt.grid(True)
		plt.show()
		exit()
	#end debug
	W_l = np.sum(w_l)
	p_l = w_l/W_l
	j = np.random.choice(6, p=p_l)
	beads.append(new_pos[j])

################################################
T = 1
beads = []
N = 1000

# IC: first two beads
beads.append(np.array([0.0, 0.0]))
beads.append(np.array([1.0, 0.0]))

# run the simulation
for i in range(0, N):
	add_bead(beads, T)

# display results
plot_beads(beads)
