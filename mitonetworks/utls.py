import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pdb import set_trace

from matplotlib import cm

def reset_plots():
	plt.close('all')
	fontsize = 20
	plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	plt.rc('text', usetex=True)
	font = {'size' : fontsize}
	plt.rc('font', **font)
	mpl.rc('lines', markersize=10)
	plt.rcParams.update({'axes.labelsize': fontsize})
	mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amsfonts}']


####################################################################
# Useful plots
####################################################################


def make_heatmap(matrix,
		xlabel, 
		ylabel, 
		zlabel, 
		xticklabels, 
		yticklabels, 
		plotextensions = ['svg','png'],
		colorbar_heatmap=cm.jet,
		figname = None,
		vmin = None, vmax = None,out_dir=os.getcwd()):	
		"""Make a heatmap of a matrix where rows are ratios and columns are magnitudes of the fusion/fission rate
		
			:param matrix: A numpy matrix, intensities for heatmap. Expect a square matrix of dimension n_points x n_points
			:param zlabel: A string, colorbar label of heatmap
			:param figname: A string, prefix to figure name. If none, do not write to file
			:param xlabel: A string, x-label of heatmap
			:param ylabel: A string, y-label of heatmap
			:param vmin: A float, minimum value for colorbar
			:param vmax: A float, maximum value for colorbar			
		"""
		plt.close('all')
		nan_col = 0.4
		nan_rgb = nan_col*np.ones(3)
		colorbar_heatmap.set_bad(nan_rgb)

		fig, ax = plt.subplots(1,1, figsize = (9,9))		
		im = ax.imshow(np.flipud(matrix), cmap = colorbar_heatmap, vmin=vmin, vmax=vmax)
		plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=zlabel)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_xticks(np.arange(len(xticklabels)))
		ax.set_yticks(np.arange(len(yticklabels)))
		ax.set_xticklabels(xticklabels)
		ax.set_yticklabels(yticklabels[::-1])

		if figname is not None:
			for p in plotextensions:
				plt.savefig(out_dir+'/'+figname+'.'+p, bbox_inches='tight')
