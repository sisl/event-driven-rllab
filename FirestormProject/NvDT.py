# Explores relationship between number of agents and required timestep

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pickle
import pdb
import progressbar


def min_rel_distance(X):
	X = X.tolist()
	rel_dists = [ np.abs(x-y) for xi,x in enumerate(X) for yi,y in enumerate(X) if xi != yi ]
	return min(rel_dists)

def find_DT(min_rel_dists, gamma):
	min_rel_dists = sorted(min_rel_dists)
	if isinstance(gamma,list):
		ind_gammas = [int(g * len(min_rel_dists)) for g in gamma]
		return [min_rel_dists[i] for i in ind_gammas]
	else:
		ind_gamma = int(gamma * len(min_rel_dists))
		return min_rel_dists[ind_gamma]

if __name__ == '__main__':

	N = 30
	DT = 0.1

	num_samps = 100000

	if(False):
		# sample N times from the distribtion, num_samps times
		
		print('Sampling...')
		samps = np.random.normal(0,1, (N,num_samps))

		# for each set of N, compute the min rel distance between two of the N samples
		print('Computing minimum relative distances...')
		bar = progressbar.ProgressBar()
		min_rel_dists = [ [min_rel_distance(samps[:n, i]) for i in range(num_samps)] for n in bar(range(2,N,3))]

		print('Sorting...')
		sorted_min_rel_dists = np.sort(np.array(min_rel_dists), axis = 1)

		dump = {'sorted_min_rel_dists': sorted_min_rel_dists, 'Ns': range(2,N,3)}

		print('Saving...')
		pickle.dump(dump, open('sortedminreldists.pkl','wb'))

		print('Saved!')

	else:
		dump = pickle.load(open('sortedminreldists.pkl','rb'))
		sorted_min_rel_dists = dump['sorted_min_rel_dists']

	# [plt.plot(sorted_min_rel_dists[i,:]) for i in range(0,sorted_min_rel_dists.shape[0])]
	# plt.show()

	# [plt.plot(sorted_min_rel_dists[:,i]) for i in [int(x*num_samps) for x in [0.01,0.03,0.1,0.3]]]
	# plt.show()

	axes = [plt.plot(list(range(2,N,3)),1./(sorted_min_rel_dists[:,i]**.5)) for i in [int(x*num_samps) for x in [0.003, 0.01,0.03,0.1,0.3]]]
	axes = [ax[0] for ax in axes ]
	plt.xlabel('N')
	plt.ylabel(r'$\frac{1}{\sqrt{\Delta T}}$')
	plt.legend(axes, map(str,[0.003, 0.01,0.03,0.1,0.3]),loc='best')
	plt.title(r'$\Delta T$ vs $N$ for various $\delta$')
	plt.grid()
	plt.savefig('NvsDT.png', bbox_inches='tight', dpi = 300)
	plt.show()










