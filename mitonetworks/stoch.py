import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import getcwd
from warnings import warn
import functools 
from matplotlib import cm
from ipywidgets import interact

#from pdb import set_trace

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

class SubmitHPCSoluableFeedbackControl(object):
	""" Create submission scripts for fusion/fission parameter sweep of a feedback control on the HPC

	Usage:: 
		>>> import mitonetworks.det as mtd
		>>> import mitonetworks.stoch as mts 
		>>> #... define parameters ...
		>>> x = mts.SubmitHPCSoluableFeedbackControl(mtd.B_differential_control_ss, nominal_params, 
				hpc_workdir,
				c_filename,
				c_file_param_ordering_convention,
				hpc_jobname)

	:param ss_definition: A function, steady state definition of the form function(ss_indep_variable, params)
	:param nominal_params: A dict, nominal parameters of the steady state 

	:param hpc_workdir: A string, working directory for stochastic simulation on the HPC
	:param c_filename: A string, name of the .c script for stochastic simulation (neglect file extension and assume prefix is same for .c and .ce files)
	:param c_file_param_ordering_convention: A list of strings, parameters to be received by the .c file from the command line
	:param hpc_jobname: A string, name of the job as it appears in Torque on the HPC
	
	:param h_target: A float, target heteroplasmy for heteroplasmy sweep

	:param n_points: An int, number of points in the scan space
	:param lg_low_mag: A float, log10 lower bound magnitudes of fusion & fission rate
	:param lg_high_mag: A float, log10 upper bound magnitudes of fusion & fission rate
	:param lg_low_rat: A float, log10 lower bound ratio of fusion / fission rate
	:param lg_high_rat: A float, log10 upper bound ratio of fusion / fission rate

	:param initial_seed: An int, seed which is incremented upon for each stochastic simulation
	:param time_per_job: A string, maximum amount of time per job in the form "HH:MM:SS"
	:param maximum_species_number: An int, maximum copy number for any single species for defining steady state
	"""
	
	def __init__(self,ss_definition, nominal_params, 
				hpc_workdir,
				c_filename,
				c_file_param_ordering_convention,
				hpc_jobname,
				h_target=0.3,
				n_points=11, lg_low_mag=-2, lg_high_mag=2,lg_low_rat=-2, lg_high_rat=2,
				n_iters_per_block=1000,
				initial_seed=None,
				time_per_job='20:00:00',
				out_dir=getcwd(),
				maximum_species_number = 5000,
				):
		
		self.ss_definition = ss_definition 
		self.nominal_params = nominal_params 
		
		self.hpc_workdir = hpc_workdir 
		self.c_filename = c_filename 
		self.c_file_param_ordering_convention = c_file_param_ordering_convention # 
		self.hpc_jobname = hpc_jobname 

		self.h_target = h_target 

		self.n_points = n_points 

		self.mags = 10**np.linspace(lg_low_mag, lg_high_mag, n_points)  
		self.rats = 10**np.linspace(lg_low_rat, lg_high_rat, n_points) 
		self.rat_nominal = nominal_params['gamma']/nominal_params['beta']

		self.n_iters_per_block = n_iters_per_block # number of iterations to run within multiple serial on HPC
		
		
		if initial_seed is None:
			self.initial_seed = np.round(np.random.uniform()*1e6).astype(int)
		else:
			self.initial_seed = initial_seed

		self.time_per_job = time_per_job
		self.out_dir = out_dir

		self.param_convention = ['ws', 'wf', 'ms', 'mf'] # global ordering convention of parameters
		self.param_convention_map = {}
		for i, s in enumerate(self.param_convention):
			self.param_convention_map[s] = i

		self.maximum_species_number = maximum_species_number 

	def find_ss_h_target(self, new_params):
		"""
		Find the steady state which is closest to the heteroplasmy target given 
		ss_definition
		"""
		indep_var_sweep = np.linspace(0,2000,2001) # sweep over independent variable
		ws, wf, ms, mf = self.ss_definition(indep_var_sweep, new_params) # get steady state for each value of the independent parameter

		# strip negative values and nans
		stacked_space = np.vstack((ws,ms,mf,wf))

		# Check all copy numbers are between 0 and self.maximum_species_number and are not nan
		good_elements = functools.reduce(np.logical_and, (stacked_space>=0, stacked_space<=self.maximum_species_number, ~np.isnan(stacked_space)))
		good_cols = np.all(good_elements,axis=0)
		stacked_space_strip_neg = stacked_space[:,good_cols]
		ws,ms,mf,wf = np.vsplit(stacked_space_strip_neg,4)
		ws = ws.ravel()
		ms = ms.ravel()
		wf = wf.ravel()
		mf = mf.ravel()

		n = ms+mf+ws+wf
		n = n.astype(float)
		h = (ms+mf)/n
		
		try:		
			ind = np.argmin(np.abs(h-self.h_target))
		except ValueError: # steady state does not exist
			return np.nan*np.zeros(4), np.nan, -1

		ics = [ws[ind], wf[ind],  ms[ind], mf[ind]]
		ics = np.round(ics).astype(int)

		h_opt = (ms[ind]+mf[ind])/np.sum(ics).astype(float)

		return ics, h_opt, 1

	def make_submission_string(self, param_block, ics, new_params):
		submission_string = './{0}.ce {1} {2} {3} $PBS_ARRAY_INDEX'.format(self.c_filename,self.initial_seed,
																								param_block,
																								self.n_iters_per_block)     
		for ic in ics:
			submission_string+=' '+str(ic)

		for p in self.c_file_param_ordering_convention:
			submission_string+=' {}'.format(new_params[p])

		submission_string+='\n'

		return submission_string

	def make_submission_script(self, param_block, ics, new_params):
		"""
		A PBS script for each parametrization to run .c script
		"""
		f = open(self.out_dir+'/submission_{}.pbs'.format(param_block),'w')
		f.write('#PBS -N {0}_{1}\n'.format(self.hpc_jobname, param_block))
		f.write('#PBS -l walltime={}\n'.format(self.time_per_job))
		f.write('#PBS -l select=1:ncpus=1:mem=1gb\n')
		f.write('#PBS -J 0-{}\n'.format(self.n_iters_per_block - 1))

		f.write('cp $PBS_O_WORKDIR/{}.ce $TMPDIR\n'.format(self.c_filename))

		submission_string = self.make_submission_string(param_block, ics, new_params)
		  
		f.write(submission_string) 
		f.write('cp output_{0}_$PBS_ARRAY_INDEX.txt $PBS_O_WORKDIR/p{0}\n'.format(param_block))

		# clean up -- have a file number quota
		f.write('sleep 10\n')
		f.write('rm $PBS_O_WORKDIR/{}_*\n'.format(self.hpc_jobname)) # clear up .o and .e files to save space
		f.close()




	def make_submission_scripts(self):
		"""
		Make submission scripts for the HPC for a feedback control
		"""
		master_script = open(self.out_dir+"/master.sh","w")
		# Add the .c compilation line to the master script
		master_script.write('gcc -o3 {0}.c -lm -o {0}.ce\n'.format(self.c_filename))

		# Output dataframe with parametriations
		param_df =  []

		for i, ratio in enumerate(self.rats):
			for j, mag in enumerate(self.mags):

				param_block = self.n_points*i+j 

				# Make the new parametrization
				new_params = self.nominal_params.copy()
				new_params['beta'] = self.nominal_params['beta']*mag
				new_params['gamma'] = self.nominal_params['gamma']*mag*ratio

				# Find the steady state initial condition for a particular target heteroplasmy
				ics, h_param, ss_flag = self.find_ss_h_target(new_params)
				
				# Save the parameters and initial conditions
				param_dict = new_params.copy()
				param_dict['mag'] = mag; param_dict['rat'] = ratio; param_dict['h'] = h_param; 
				for k, param in enumerate(self.param_convention): param_dict[param+'_init'] = ics[k]
				param_df.append(pd.DataFrame(param_dict, index=[0]))

				if ss_flag == 1: # if a steady state exists, continue
					if abs((self.h_target - h_param)/h_param) > 0.01:
						warn("Heteroplasmy target not reached, i={0}, j={1}, h={2}".format(i,j,h_param), UserWarning)

					# Make a submission script
					self.make_submission_script(param_block, ics, new_params)

					# Add the submission to a master script
					master_script.write('echo "{}"\n'.format(param_block))
					master_script.write('mkdir -p {0}/p{1}\n'.format(self.hpc_workdir,param_block))
					master_script.write("qsub submission_{}.pbs\n".format(param_block))
					master_script.write('sleep 0.5\n')
				else:
					warn("SS not found, i={0}, j={1}".format(i,j))



		master_script.close()
		param_df = pd.concat(param_df, ignore_index=True)
		self.param_df = param_df
		param_df.to_csv(self.out_dir+'/param_sweep_vals.csv',index=False)



class AnalyseDataFeedbackControl(object):
	""" Analyse processed data from HPC for parameter sweep of feedback control
	
	:param dir_df_params: A string, the directory of the dataframe containing parameter information for each parametrization
	:param dir_data: A string, directory containing ensemble statistics of a parametrization of a feedback control
	:param ctrl_name: A string, name of the control for file extensions

	:param out_dir: A string, output directory for files
	:param n_points: An int, number of points in the scan space
	:param lg_low_mag: A float, log10 lower bound magnitudes of fusion & fission rate
	:param lg_high_mag: A float, log10 upper bound magnitudes of fusion & fission rate
	:param lg_low_rat: A float, log10 lower bound ratio of fusion / fission rate
	:param lg_high_rat: A float, log10 upper bound ratio of fusion / fission rate
	:param nan_col: A float, color to plot nan in heatmap
	:param colorbar_heatmap: A matplotlib colormap, for heatmaps
	:param plotextensions: A list of strings, extensions for heatmaps
	:param ansatz_is_ajhg: A bool, if true use AJHG as ansatz, otherwise use network ansatz
	"""
	def __init__(self, dir_df_params, dir_data, ctrl_name,
		out_dir=getcwd(),
		n_points=11, lg_low_mag=-2, lg_high_mag=2,lg_low_rat=-2, lg_high_rat=2,
		nan_col = 0.4, colorbar_heatmap=cm.jet, plotextensions = ['svg','png'],
		istransposed=False,
		ansatz_is_ajhg=False,
		):
		self.df_params = pd.read_csv(dir_df_params)
		self.dir_data = dir_data
		self.ctrl_name = ctrl_name
		self.istransposed = istransposed

		self.param_convention = ['ws', 'wf', 'ms', 'mf']
		self.out_dir = out_dir
		self.n_points = n_points 
		self.mags = 10**np.linspace(lg_low_mag, lg_high_mag, n_points)  
		self.rats = 10**np.linspace(lg_low_rat, lg_high_rat, n_points) 

		self.nan_col = nan_col
		self.colorbar_heatmap = colorbar_heatmap
		self.plotextensions = plotextensions
		self.ansatz_is_ajhg = ansatz_is_ajhg


	def het_var_ansatz(self,steady_state, tmax, dt, mu):
	    t = np.linspace(0,tmax,int((tmax+dt)/dt))
	    ws_sol = steady_state[0]
	    wf_sol = steady_state[1]
	    ms_sol = steady_state[2]
	    mf_sol = steady_state[3]

	    n_sol = ws_sol + wf_sol + ms_sol + mf_sol
	    meanh = (mf_sol+ms_sol)/float(n_sol)
	    fsingleton = (ws_sol + ms_sol)/float(n_sol)

	    return 2*mu/n_sol*t*meanh*(1-meanh)*fsingleton

	def het_var_ajhg(self,steady_state, tmax, dt, mu):
	    t = np.linspace(0,tmax,int((tmax+dt)/dt))
	    ws_sol = steady_state[0]
	    wf_sol = steady_state[1]
	    ms_sol = steady_state[2]
	    mf_sol = steady_state[3]

	    n_sol = ws_sol + wf_sol + ms_sol + mf_sol
	    meanh = (mf_sol+ms_sol)/float(n_sol)

	    return 2*mu/n_sol*t*meanh*(1-meanh)

	def make_gradients(self):
		"""Compute the gradient of heteroplasmy for every parametrization"""
		
		vh = [] # heteroplasmy variance
		grad_vh = [] # mean gradient heteroplasmy variance

		for row in self.df_params.iterrows():
			idx = row[0] # block index
			vals = row[1]

			try:
				stats_data = pd.read_csv(self.dir_data+'/online_stats_{}.csv'.format(idx))
			except IOError:
				continue # go to next parameter block

			TFinal = stats_data['t'].iloc[-1]
			dt = stats_data.iloc[1]['t']-stats_data.iloc[0]['t']
			grad_vh_sim = np.mean(np.gradient(stats_data['var_h'],dt))

			ss_vals = [] # steady state values
			for p in self.param_convention:
				ss_vals.append(vals[p+'_init']) # simulations are initialised at deterministic steady state
			if self.ansatz_is_ajhg:
				vh_ansatz = self.het_var_ajhg(ss_vals, TFinal, dt, vals['mu']) # heteroplasmy variance ansatz
			else:
				vh_ansatz = self.het_var_ansatz(ss_vals, TFinal, dt, vals['mu']) # heteroplasmy variance ansatz

			grad_vh_ansatz = np.mean(np.gradient(vh_ansatz,dt)) 
			grad_vh.append(pd.DataFrame([[idx,grad_vh_sim,grad_vh_ansatz]],columns=['block_idx','grad_vh_sim','grad_vh_ansatz']))

			vh.append(pd.DataFrame(np.transpose(np.vstack((idx*np.ones(len(stats_data)),stats_data['t'],stats_data['var_h'],vh_ansatz))),
							columns = ['block_idx','t','vh_sim','vh_ansatz'])) 
		self.grad_vh = pd.concat(grad_vh, ignore_index=True)
		self.grad_vh.to_pickle(self.out_dir+'/{}_grad_vh'.format(self.ctrl_name))

		self.vh = pd.concat(vh, ignore_index=True)
		self.vh.to_pickle(self.out_dir+'/{}_vh'.format(self.ctrl_name))

	def compute_errors(self):
		"""Compute the error between the ansatz and stochastic simulation for heteroplasmy gradient"""		
		grad_vh = pd.read_pickle(self.out_dir+'/{}_grad_vh'.format(self.ctrl_name))		
		self.errors = grad_vh
		self.errors['error_sim_ansatz'] = np.abs(1. - grad_vh['grad_vh_ansatz']/grad_vh['grad_vh_sim'])

		self.error_ansatz_beta_gamma = np.zeros((self.n_points,self.n_points))	
		for i in range(self.n_points): # rats
			for j in range(self.n_points): # mags
				param_block = self.n_points*i+j 			
				row = self.errors[self.errors.block_idx==param_block]
				if len(row)==0:
					self.error_ansatz_beta_gamma[i,j] = np.nan
				else:
					self.error_ansatz_beta_gamma[i,j] = np.log10(row['error_sim_ansatz'].values[0])

	def make_summary_matrices(self, t_eval):
		"""
		:param t_eval: A float, time to evaluate summary stats where appropriate
		Returns:

			Matrices (self.n_points x self.n_points) for every parametrization:

				self.vh_array: Heteroplasmy variance
				self.mfs_array: Mean fraction singletons
				self.mn_array: Mean copy number
				self.h_cdf_4_array: Probability h <= 0.4
				self.eh_array: Mean heteroplasmy
				self.count_array: Counts
				self.vn_array: Heteroplasmy variance
		"""
		self.vh_array = np.nan*np.zeros((self.n_points,self.n_points)) 
		self.mfs_array = np.nan*np.zeros((self.n_points,self.n_points)) 
		self.mn_array = np.nan*np.zeros((self.n_points,self.n_points))
		self.vn_array = np.nan*np.zeros((self.n_points,self.n_points))
		self.h_cdf_4_array = np.nan*np.zeros((self.n_points,self.n_points)) # this is a bit arbitrary. Assume h_target was h = 0.3 and all params are well-constrained thereabouts
		self.eh_array = np.nan*np.zeros((self.n_points,self.n_points)) 
		self.count_array = np.nan*np.zeros((self.n_points,self.n_points)) 
		self.p_h_fix_0_array = np.nan*np.zeros((self.n_points,self.n_points)) 
		self.p_h_fix_1_array = np.nan*np.zeros((self.n_points,self.n_points)) 

		for i in range(self.n_points): # mags
			for j in range(self.n_points): # ratios
				param_block = self.n_points*i+j

				try:
					stats_data = pd.read_csv(self.dir_data+'/online_stats_{}.csv'.format(param_block))
					data_present = True
				except IOError:
					data_present = False
				
				if data_present:
					self.vh_array[i,j] = self.errors.loc[param_block]['grad_vh_sim']
					self.mfs_array[i,j] = stats_data[stats_data['t']==t_eval].mean_fs
					self.mn_array[i,j] = stats_data[stats_data['t']==t_eval].mean_n
					self.h_cdf_4_array[i,j] = 1. - (stats_data[stats_data['t']==t_eval].cdf_h_0_4/stats_data[stats_data['t']==t_eval].counts)
					self.p_h_fix_0_array[i,j] = 1. - (stats_data[stats_data['t']==t_eval].p_h_fix_0/stats_data[stats_data['t']==t_eval].counts)
					self.p_h_fix_1_array[i,j] = 1. - (stats_data[stats_data['t']==t_eval].p_h_fix_1/stats_data[stats_data['t']==t_eval].counts)
					self.eh_array[i,j] = stats_data[stats_data['t']==t_eval].mean_h
					self.vn_array[i,j] = stats_data[stats_data['t']==t_eval].var_n
					self.count_array[i,j] = stats_data.counts.sum()
				else:
					self.vh_array[i,j] = np.nan
					self.mfs_array[i,j] = np.nan
					self.mn_array[i,j] = np.nan
					self.h_cdf_4_array[i,j] = np.nan
					self.p_h_fix_0_array[i,j] = np.nan
					self.p_h_fix_1_array[i,j] = np.nan
					self.eh_array[i,j] = np.nan
					self.count_array[i,j] = np.nan
					self.vn_array[i,j] = np.nan


	def make_heatmap(self, matrix, zlabel, figname = None,
		xlabel = r'Log10 relative churn, $M$', ylabel =  r'Log10 relative fusion/fission rate, $R$', 
		vmin = None, vmax = None):	
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
		self.colorbar_heatmap.set_bad(nan_rgb)

		if self.istransposed:
			matrix = np.transpose(matrix)

		fig, ax = plt.subplots(1,1, figsize = (9,9))		
		im = ax.imshow(np.flipud(matrix), cmap = self.colorbar_heatmap, vmin=vmin, vmax=vmax)
		plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=zlabel)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_xticks(np.arange(self.n_points));
		ax.set_yticks(np.arange(self.n_points));
		ax.set_xticklabels(np.log10(self.mags));
		ax.set_yticklabels(np.log10(self.rats)[::-1]);

		if figname is not None:
			for p in self.plotextensions:
				plt.savefig(self.out_dir+'/'+self.ctrl_name+'_'+figname+'.'+p, bbox_inches='tight')

	def get_vh_plotting(self, block_idx):
		""" Get a particular mean heteroplasmy variance trajectory, loading from disk if necessary
		
		:param block_idx: An int, block of the index to recall 
		:rtype vh_plotting:
		"""
		try:
			vh_plotting = self.vh[self.vh['block_idx']==block_idx]
		except AttributeError:
			try:
				self.vh = pd.read_pickle(self.out_dir+'/{}_vh'.format(self.ctrl_name))
				vh_plotting = self.vh[self.vh['block_idx']==block_idx]
			except IOError:
				raise Exception('Run AnalyseDataSoluableFeedbackControl.make_gradients()')
		return vh_plotting
	
	def plt_hvar(self, i):
		vh_plotting = self.get_vh_plotting(i)

		plt.plot(vh_plotting['t'],vh_plotting['vh_sim'],'.k',label='Sim', markersize=8)
		plt.plot(vh_plotting['t'],vh_plotting['vh_ansatz'],'-g',label='Ansatz')
		plt.xlabel('Time')
		plt.ylabel('Heteroplasmy variance')
		plt.legend()

	def make_sweep_widget(self, slidermin = 0, slidermax = None):
		if slidermax is None:
			slidermax = len(self.df_params)-1
		interact(self.plt_hvar, i = (slidermin, slidermax));

	# def add_vh_to_ax(self, ax, vh_plotting):
	# 	ax.plot(vh_plotting['t'],vh_plotting['vh_sim'],'kx',label='Simulation')
	# 	ax.plot(vh_plotting['t'],vh_plotting['vh_ansatz'],'go',mfc='none',label='Ansatz')
	# 	ax.legend(prop={'size':12})
	# 	ax.set_xlabel('Time, (days)')
	# 	ax.set_ylabel('Heteroplasmy variance, $\mathbb{V}(h)$')

	# def plot_best_worst(self, best_coord, worst_coord):
	# 	i_b, j_b = best_coord
	# 	p_b = 11*i_b + j_w
	# 	vh_plotting = self.get_vh_plotting(p_b)
	# 	i_w, j_w = worst_coord

	

						
	def plot_h_n_t(self,param_block, leg_size=10, figname=None):
		""" Plot mean/var heteroplasmy/copy number for a particular parametrization
		:param param_block: An int indicating the parameterization index to be plotted
		"""

		stats_data = pd.read_csv(self.dir_data+'/online_stats_{}.csv'.format(param_block))

		params = dict(self.df_params.iloc[param_block])

		steady_state = [params['ws_init'],params['wf_init'],params['ms_init'],params['mf_init']]
		n_ss = np.sum(steady_state)
		h_ss = (params['ms_init']+params['mf_init'])/float(n_ss)
		
		tmax = stats_data.iloc[-1]['t']
		dt = stats_data.iloc[1]['t'] - stats_data.iloc[0]['t']
		mu = params['mu']
		t = stats_data.t
		
		if self.ansatz_is_ajhg:
			vh_an = self.het_var_ajhg(steady_state, tmax, dt, mu)
		else:
			vh_an = self.het_var_ansatz(steady_state, tmax, dt, mu)


		fig, axs = plt.subplots(2,2,figsize=(2*5,2*5))

		ax = axs[0,0]
		ax.plot(t,stats_data.mean_h,'kx', label='Simulation')
		ax.fill_between(t, stats_data.mean_h + 2.0*np.sqrt(stats_data.var_h/stats_data.counts),
						   stats_data.mean_h - 2.0*np.sqrt(stats_data.var_h/stats_data.counts),
						   color = 'red',
						   alpha = 0.3, label = '$\pm2$ SEM')
		ax.plot(t,h_ss*np.ones(len(t)),'-r',label='Deterministic')
		ax.plot()
		ax.set_xlabel('Time (days)')
		ax.set_ylabel('Mean heteroplasmy')
		ax.legend(prop={'size':leg_size})

		ax = axs[1,0]
		ax.plot(t,stats_data.var_h,'kx', label='Simulation')
		ax.plot(t,vh_an,'og',mfc='none', label='Ansatz');
		ax.set_xlabel('Time (days)')
		ax.set_ylabel('Heteroplasmy variance')
		ax.legend(prop={'size':leg_size})

		ax = axs[0,1]
		ax.plot(t,stats_data.mean_n,'kx', label='Simulation')
		ax.fill_between(t, stats_data.mean_n + 2.0*np.sqrt(stats_data.var_n/stats_data.counts),
						   stats_data.mean_n - 2.0*np.sqrt(stats_data.var_n/stats_data.counts),
						   color = 'red',
						   alpha = 0.3, label = '$\pm2$ SEM')
		ax.plot(t,n_ss*np.ones(len(t)),'-r',label='Deterministic')
		ax.set_xlabel('Time (days)')
		ax.set_ylabel('Mean copy number')
		ax.legend(prop={'size':leg_size})

		ax = axs[1,1]
		ax.plot(t,stats_data.var_n,'kx', label='Simulation')
		ax.set_xlabel('Time (days)')
		ax.set_ylabel('Copy number variance')
		ax.legend(prop={'size':leg_size})

		plt.tight_layout()

		if figname is not None:
			for p in self.plotextensions:
				plt.savefig(self.out_dir+'/'+self.ctrl_name+'_'+figname+'_'+str(param_block)+'.'+p, bbox_inches='tight')








		










			
			
			






		












