import pickle
import matplotlib.pyplot as plt
import pdb
import numpy as np
import glob


def main1():
	filenames = glob.glob('./data/data/ckpt*')
	for fn in filenames:
		print(fn)
		results = pickle.load(open(fn,'rb'))

		means = [0]*5
		stds = [0]*5

		fig = plt.figure(0)

		i = 0

		all_drs = []

		pol_means = []
		pol_stds = []

		for key,val in results.items():
			drs = [val2 for _,val2 in val.items()]
			pol_means.append([np.mean(v) for v in drs])
			pol_stds.append([np.std(v)/np.sqrt(len(v)) for v in drs])
			drs = [item for sublist in drs for item in sublist]
			all_drs.append(drs)
			means[i] = np.mean(drs)
			stds[i] = np.std(drs) / np.sqrt(len(drs))
			i += 1

		all_drs = [item for sublist in all_drs for item in sublist]
		total_mean = np.mean(all_drs)
		total_std = np.std(all_drs) / np.sqrt(len(all_drs))

		start_ind = 0
		order = 0
		plt.rc('lines', **{'linestyle': 'None'})
		for i, pol_mean in enumerate(pol_means):
			plt.errorbar(range(start_ind, start_ind + len(pol_mean)),
				pol_mean, yerr = pol_stds[i], marker = 'o', fmt='', zorder = 100-order)
			start_ind += len(pol_mean)
			order -= 5

		plt.plot(range(0, start_ind), [total_mean]*start_ind, marker='_', color='0.6', zorder = 100-order )
		order -= 5
		plt.plot(range(0, start_ind), [total_mean+total_std]*start_ind, '--', color='0.6', zorder = 100-order )
		order -= 5
		plt.plot(range(0, start_ind), [total_mean-total_std]*start_ind, '--', color='0.6', zorder = 100-order )
		order -= 5

		start_ind = 0
		for i in range(len(means)):
			plt.plot(range(start_ind, start_ind + len(pol_means[i])),
			 [means[i]]*len(pol_means[i]), marker='_', color = 'k',zorder = 100-order )
			order -= 5
			plt.plot(range(start_ind, start_ind + len(pol_means[i])),
			 [means[i]+stds[i]]*len(pol_means[i]), '--', color = 'k',zorder = 100-order )
			order -= 5
			plt.plot(range(start_ind, start_ind + len(pol_means[i])),
			 [means[i]-stds[i]]*len(pol_means[i]), '--', color = 'k',zorder = 100-order )
			order -= 5
			start_ind += len(pol_means[i])
			

		plt.show()


def main2():

	show_new_uavsec = False

	filenames = ['ckpt_-1.pkl', 'ckpt_0.1.pkl', 'ckpt_0.316.pkl', 'ckpt_1.pkl', 'ckpt_3.162.pkl', 'ckpt_10.pkl', 'ckpt_1.00_2.999uavsec.pkl']
	filenames = ['./data/data/' + fn for fn in filenames]

	curve_means = []
	curve_stds = []

	total_means = []
	total_stds = []

	for fn in filenames:
		#print('Opening '+fn)
		results = pickle.load(open(fn,'rb'))

		means = []
		stds = []

		fig = plt.figure(0)

		i = 0

		all_drs = []

		pol_means = []
		pol_stds = []

		for local_opt,local_opt_dict in results.items():
			drs = [pol_dict for _,pol_dict in local_opt_dict.items()]
			pol_means.append([np.mean(v) for v in drs])
			pol_stds.append([np.std(v)/np.sqrt(len(v)) for v in drs])
			#drs = [item for sublist in drs for item in sublist]
			#all_drs.append(drs)
			#means.append(np.mean(pol_means))
			#stds.append(np.std(pol_means) / np.sqrt(len(pol_means)))
			i += 1

		means = [np.mean(pol_mean) for pol_mean in pol_means]
		stds = [np.std(pol_mean) / np.sqrt(len(pol_mean)) for pol_mean in pol_means]


		all_drs = [item for sublist in all_drs for item in sublist]
		total_mean = np.mean(all_drs)
		total_std = np.std(all_drs) / np.sqrt(len(all_drs))

		curve_means.append(means)
		curve_stds.append(stds)
		total_means.append(np.mean(means))
		total_stds.append(np.std(means) / np.sqrt(len(means)))

	start_ind = 0
	order = 0
	plt.rc('lines', **{'linestyle': 'None'})
	for i, curve_mean in enumerate(curve_means[:-1]):
		h2 = plt.errorbar( range(start_ind, start_ind + len(curve_mean)), #[start_ind+len(curve_mean)/2]*len(curve_mean), #
			curve_mean, yerr = curve_stds[i], marker = 'o', fmt='', zorder = 100-order)
		start_ind += len(curve_mean)
		order -= 5

	labels = ['ED', r'$10^{-1}$', r'$10^{-0.5}$', r'$10^{0}$', r'$10^{0.5}$', r'$10^{1}$']
	plt.xticks([2.5,7.5,12.5,17.5,22.5,27.5], labels, rotation='vertical')

	if(show_new_uavsec):
		plt.errorbar( range(16,19), #[15+5/2]*len(curve_means[-1]), 
				curve_means[-1], yerr = curve_stds[-1], marker = 'o', fmt='', zorder = 100-order)

	start_ind = 0
	for i in range(len(total_means)-1):
		h3, = plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]]*len(curve_means[i]), marker='_', color = 'k',zorder = 100-order )
		order -= 5
		plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]+total_stds[i]]*len(curve_means[i]), '--', color = 'k',zorder = 100-order )
		order -= 5
		plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]-total_stds[i]]*len(curve_means[i]), '--', color = 'k',zorder = 100-order )
		order -= 5
		start_ind += len(curve_means[i])

	plt.plot(range(0, start_ind),
			 [2.83440175693 + 0.0212387516683]*start_ind, '--', color = 'r',zorder = 0 )
	order -= 5
	h1, = plt.plot(range(0, start_ind),
		 [2.83440175693 ]*start_ind, marker='_', color = 'r',zorder = 0 )
	order -=5
	plt.plot(range(0, start_ind),
		 [2.83440175693 - 0.0212387516683]*start_ind, '--', color = 'r',zorder = 0 )

	if(show_new_uavsec):
		plt.plot(range(15, 20),
			 [total_means[-1]+total_stds[-1]]*5, '--', color = 'k',zorder = 100-order )
		order -= 5
		plt.plot(range(15, 20),
			 [total_means[-1]]*5, marker='_', color = 'k',zorder = 100-order )
		order -=5
		plt.plot(range(15, 20),
			 [total_means[-1]-total_stds[-1]]*5, '--', color = 'k',zorder = 100-order )


	plt.legend([h1,h2,h3], ['Nearest Live Fire Policy', 'Training Simulation (TS)', 'Average over TS'])
	plt.ylabel('Average Discounted Return')
	plt.xlabel('Training Simulation Time-step')
	plt.title('Estimated Learned Policy Performance on ED Simulator')
	plt.grid()
	plt.tight_layout()
	if(show_new_uavsec):
		plt.savefig('PolicyPerformance.png', bbox_inches='tight', dpi = 300)
	else:
		plt.savefig('PolicyPerformanceNoNewUS.png', bbox_inches='tight', dpi = 300)
	plt.show()


def main3():
	# Plots for NASA Presentation

	show_new_uavsec = True

	filenames = ['ckpt_0.1.pkl', 'ckpt_0.316.pkl', 'ckpt_1.pkl', 'ckpt_3.162.pkl', 'ckpt_10.pkl', 'ckpt_1.00_2.999uavsec.pkl']
	filenames = ['./data/data/' + fn for fn in filenames]

	curve_means = []
	curve_stds = []

	total_means = []
	total_stds = []

	for fn in filenames:
		#print('Opening '+fn)
		results = pickle.load(open(fn,'rb'))

		means = []
		stds = []

		fig = plt.figure(0)

		i = 0

		all_drs = []

		pol_means = []
		pol_stds = []

		for local_opt,local_opt_dict in results.items():
			drs = [pol_dict for _,pol_dict in local_opt_dict.items()]
			pol_means.append([np.mean(v) for v in drs])
			pol_stds.append([np.std(v)/np.sqrt(len(v)) for v in drs])
			#drs = [item for sublist in drs for item in sublist]
			#all_drs.append(drs)
			#means.append(np.mean(pol_means))
			#stds.append(np.std(pol_means) / np.sqrt(len(pol_means)))
			i += 1

		means = [np.mean(pol_mean) for pol_mean in pol_means]
		stds = [np.std(pol_mean) / np.sqrt(len(pol_mean)) for pol_mean in pol_means]


		all_drs = [item for sublist in all_drs for item in sublist]
		total_mean = np.mean(all_drs)
		total_std = np.std(all_drs) / np.sqrt(len(all_drs))

		curve_means.append(means)
		curve_stds.append(stds)
		total_means.append(np.mean(means))
		total_stds.append(np.std(means) / np.sqrt(len(means)))

	start_ind = 0
	order = 0

	labels = [r'$10^{-1}$', r'$10^{-0.5}$', r'$10^{0}$', r'$10^{0.5}$', r'$10^{1}$']
	plt.xticks([2.5,7.5,12.5,17.5,22.5], labels, rotation='vertical')

	start_ind = 0
	for i in range(len(total_means)-1):
		h3, = plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]]*len(curve_means[i]), marker='_', color = 'b',zorder = 100-order )
		order -= 5
		plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]+total_stds[i]]*len(curve_means[i]), '--', color = 'b',zorder = 100-order )
		order -= 5
		plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]-total_stds[i]]*len(curve_means[i]), '--', color = 'b',zorder = 100-order )
		order -= 5
		start_ind += len(curve_means[i])


	if(show_new_uavsec):
		plt.plot(range(10, 15),
			 [total_means[-1]+total_stds[-1]]*5, '--', color = 'r',zorder = 100-order )
		order -= 5
		h4, = plt.plot(range(10, 15),
			 [total_means[-1]]*5, marker='_', color = 'r',zorder = 100-order )
		order -=5
		plt.plot(range(10, 15),
			 [total_means[-1]-total_stds[-1]]*5, '--', color = 'r',zorder = 100-order )


	plt.legend([h4], ['Slightly Modified Environment Parameter'])
	plt.ylabel('Average Discounted Return')
	plt.xlabel('Training Simulation Time-step')
	plt.title('Estimated Learned Policy Performance on "Ground Truth" Simulator')
	plt.grid()
	plt.tight_layout()
	# if(show_new_uavsec):
	# 	plt.savefig('PolicyPerformance.png', bbox_inches='tight', dpi = 300)
	# else:
	plt.savefig('PolicyPerformanceNoED.png', bbox_inches='tight', dpi = 300)
	plt.show()


def main4():

	# NASA Presentation plots, including ED

	show_new_uavsec = False

	filenames = ['ckpt_-1.pkl', 'ckpt_0.1.pkl', 'ckpt_0.316.pkl', 'ckpt_1.pkl', 'ckpt_3.162.pkl', 'ckpt_10.pkl', 'ckpt_1.00_2.999uavsec.pkl']
	filenames = ['./data_firestorm/data/' + fn for fn in filenames]

	curve_means = []
	curve_stds = []

	total_means = []
	total_stds = []

	for fn in filenames:
		#print('Opening '+fn)
		results = pickle.load(open(fn,'rb'))

		means = []
		stds = []

		fig = plt.figure(0)

		i = 0

		all_drs = []

		pol_means = []
		pol_stds = []

		for local_opt,local_opt_dict in results.items():
			drs = [pol_dict for _,pol_dict in local_opt_dict.items()]
			pol_means.append([np.mean(v) for v in drs])
			pol_stds.append([np.std(v)/np.sqrt(len(v)) for v in drs])
			#drs = [item for sublist in drs for item in sublist]
			#all_drs.append(drs)
			#means.append(np.mean(pol_means))
			#stds.append(np.std(pol_means) / np.sqrt(len(pol_means)))
			i += 1

		means = [np.mean(pol_mean) for pol_mean in pol_means]
		stds = [np.std(pol_mean) / np.sqrt(len(pol_mean)) for pol_mean in pol_means]


		all_drs = [item for sublist in all_drs for item in sublist]
		total_mean = np.mean(all_drs)
		total_std = np.std(all_drs) / np.sqrt(len(all_drs))

		curve_means.append(means)
		curve_stds.append(stds)
		total_means.append(np.mean(means))
		total_stds.append(np.std(means) / np.sqrt(len(means)))

	start_ind = 0
	order = 0


	labels = ['ED', r'$10^{-1}$', r'$10^{-0.5}$', r'$10^{0}$', r'$10^{0.5}$', r'$10^{1}$']
	plt.xticks([2.5,7.5,12.5,17.5,22.5,27.5], labels, rotation='vertical')

	for i in range(len(total_means)):
		print(total_means[i], total_stds[i])


	for i in range(len(total_means)-1):
		clr = 'm' if i == 0 else 'b'
		h3, = plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]]*len(curve_means[i]), marker='_', color = clr,zorder = 100-order )
		order -= 5
		plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]+total_stds[i]]*len(curve_means[i]), '--', color = clr,zorder = 100-order )
		order -= 5
		plt.plot(range(start_ind, start_ind + len(curve_means[i])),
		 [total_means[i]-total_stds[i]]*len(curve_means[i]), '--', color = clr,zorder = 100-order )
		order -= 5
		start_ind += len(curve_means[i])

	# plt.plot(range(0, start_ind),
	# 		 [2.83440175693 + 0.0212387516683]*start_ind, '--', color = 'r',zorder = 0 )
	# order -= 5
	# h1, = plt.plot(range(0, start_ind),
	# 	 [2.83440175693 ]*start_ind, marker='_', color = 'r',zorder = 0 )
	# order -=5
	# plt.plot(range(0, start_ind),
	# 	 [2.83440175693 - 0.0212387516683]*start_ind, '--', color = 'r',zorder = 0 )

	if(True):
		plt.plot(range(15, 20),
			 [total_means[-1]+total_stds[-1]]*5, '--', color = 'r',zorder = 100-order )
		order -= 5
		h4, = plt.plot(range(15, 20),
			 [total_means[-1]]*5, marker='_', color = 'r',zorder = 100-order )
		order -=5
		plt.plot(range(15, 20),
			 [total_means[-1]-total_stds[-1]]*5, '--', color = 'r',zorder = 100-order )


	plt.legend([h4], ['Slightly Modified Environment Parameter'])
	plt.ylabel('Average Discounted Return')
	plt.xlabel('Training Simulation Time-step')
	plt.title('Estimated Learned Policy Performance on ED Simulator')
	plt.grid()
	plt.tight_layout()
	# plt.savefig('PolicyPerformanceWithED.png', bbox_inches='tight', dpi = 300)
	plt.show()






if __name__ == '__main__':
	main4()



