import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import FormatStrFormatter



if __name__ == "__main__":

	filenames = ['experiment_2017_05_30_gae_1',
		'experiment_2017_05_30_gae_2',
		'experiment_2017_05_30_gae_4',
		'experiment_2017_05_30_gae_8',
		'experiment_2017_05_30_gae_16',
		'experiment_2017_06_01_gae_2_with_featnet' ]

	fig, ax = plt.subplots()
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

	for filename in filenames:
		reader = csv.DictReader(open('./data/'+filename+'/progress.csv'))

		result = {}
		for row in reader:
		    for column, value in row.items():
		        result.setdefault(column, []).append(value)
		
		for key,val in result.items():
			result[key] = [float(v) for v in val]

		plt.plot(result['Iteration'], result['AverageReturn'])
		

	plt.legend([r'$\lambda=1.0$', r'$\lambda=2.0$', r'$\lambda=4.0$', r'$\lambda=8.0$', r'$\lambda=16.0$', r'$\lambda=2.0+$FeatNet'])
	plt.show()

	