import matplotlib.pyplot as plt
import numpy as np
import csv



if __name__ == "__main__":

	filenames = ['experiment_2017_05_29_16_40_10_663501_PDT_dt_-1.000',
		'experiment_2017_05_29_17_20_20_049848_PDT_dt_-1.000',
		'experiment_2017_05_29_17_33_39_626926_PDT_dt_-1.000',
		'experiment_2017_05_29_18_32_44_154121_PDT_dt_-1.000' ]

	for filename in filenames:
		reader = csv.DictReader(open('./data/'+filename+'/progress.csv'))

		result = {}
		for row in reader:
		    for column, value in row.items():
		        result.setdefault(column, []).append(value)
		
		for key,val in result.items():
			result[key] = [float(v) for v in val]

		plt.plot(result['Iteration'], result['AverageReturn'])

	plt.legend(['Default', 'LargerStep', 'GAE 2.0', 'GAE 4.0'])
	plt.show()

	