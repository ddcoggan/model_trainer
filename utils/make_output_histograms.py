import os.path as op
import matplotlib.pyplot as plt

def activation_histograms(activations, filepath):
	a = activations.detach().cpu().flatten()
	n, bins, patches = plt.hist(x=a, bins=25, color='#0504aa',
		                    alpha=1, rwidth=0.85)
	plt.grid(axis='y', alpha=1)
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	if not op.exists(op.dirname(filepath)):
	    os.make_dirs(op.dirname(filepath))
	plt.savefig(filepath)
	plt.close()
