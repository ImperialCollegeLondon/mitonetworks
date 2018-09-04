import numpy as np
from scipy.integrate import ode 
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
from pdb import set_trace

import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter


'''
parameter ordering convention: ['ws','wf','ms','mf']
'''

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

def hello_world():
	print('Hello world!')

def add_arrow(line, position=None, direction='right', size=15, color=None, alpha=0.5):
		    """
		    Add an arrow to a line.

		    line:       Line2D object
		    position:   x-position of the arrow. If None, mean of xdata is taken
		    direction:  'left' or 'right'
		    size:       size of the arrow in fontsize points
		    color:      if None, line color is taken.
		    """
		    if color is None:
		        color = line.get_color()

		    xdata = line.get_xdata()
		    ydata = line.get_ydata()

		    if position is None:
		        position = (xdata[-1] + xdata[0])/2		        
		    # find closest index
		    start_ind = np.argmin(np.absolute(xdata - position))
		    if direction == 'right':
		        end_ind = start_ind + 1
		    else:
		        end_ind = start_ind - 1

		    line.axes.annotate('',
		        xytext=(xdata[start_ind], ydata[start_ind]),
		        xy=(xdata[end_ind], ydata[end_ind]),
		        arrowprops=dict(arrowstyle="->", color=color,alpha=alpha),
		        size=size
		    )

#######################
# Feedback controls 
#######################

def network_system(ws, wf, ms, mf, gamma, beta, rep_rate, degrate, xi, epsilon_pcp=1.0, Q_f=1.0):
	wsdot = -1.0 * gamma * ws * ws - gamma * ws * wf + beta * wf - rep_rate * ws - degrate * ws - epsilon_pcp * gamma*Q_f * mf * ws - epsilon_pcp * gamma * ws * ms
	msdot = -1.0 * gamma*Q_f * ms * ms - gamma*Q_f * ms * mf + beta * mf - rep_rate * ms - degrate * ms  - epsilon_pcp * gamma*Q_f * wf * ms - epsilon_pcp * gamma * ws * ms

	wfdot = 1.0 * gamma * ws * ws + gamma * ws * wf - beta * wf + rep_rate * (2.0*ws + wf) - xi * degrate * wf + epsilon_pcp * gamma*Q_f * mf * ws + epsilon_pcp * gamma * ws * ms
	mfdot = 1.0 * gamma*Q_f * ms * ms + gamma*Q_f * ms * mf - beta * mf + rep_rate * (2.0*ms + mf) - xi * degrate * mf + epsilon_pcp * gamma*Q_f * wf * ms + epsilon_pcp * gamma * ws * ms

	return [wsdot, wfdot, msdot, mfdot]

def A_relaxed_replication(t, y, params):    
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	alpha = params['alpha']
	mu = params['mu']
	w_opt = params['w_opt']
	delta  = params['delta']

	ws, wf, ms, mf = y

	w = ws + wf
	m = ms + mf

	rep_rate = alpha*mu*(w_opt - w - delta*m)/(w+m)

	if rep_rate < 0:
		rep_rate = 0

	return network_system(ws, wf, ms, mf, gamma, beta, rep_rate, mu, xi)


def B_differential_control(t, y, params):    
    beta = params['beta']
    gamma = params['gamma']
    xi = params['xi']
    alpha = params['alpha']
    mu = params['mu']
    w_opt  = params['w_opt']

    ws, wf, ms, mf = y

    w = ws + wf
    m = ms + mf

    rep_rate = alpha*(w_opt - w)

    if rep_rate < 0:
		rep_rate = 0

    return network_system(ws, wf, ms, mf, gamma, beta, rep_rate, mu, xi)

def C_ratiometric_control(t, y, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	alpha = params['alpha']
	mu = params['mu']
	w_opt = params['w_opt']

	ws, wf, ms, mf = y

	w = ws + wf
	m = ms + mf

	rep_rate = alpha*(w_opt/w - 1.0)

	if rep_rate < 0:
		rep_rate = 0

	return network_system(ws, wf, ms, mf, gamma, beta, rep_rate, mu, xi)

def D_no_control(t, y, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	mu = params['mu']

	ws, wf, ms, mf = y

	w = ws + wf
	m = ms + mf

	return network_system(ws, wf, ms, mf, gamma, beta, mu, mu, xi)

def E_linear_feedback_control(t, y, params):
	xi = params['xi']
	gamma = params['gamma']
	beta = params['beta']
	kappa = params['kappa']
	b = params['b']
	mu = params['mu']
	delta = params['delta']

	if 'epsilon_pcp' in params:
		epsilon_pcp = params['epsilon_pcp']
	else:
		epsilon_pcp = 1.0
	if 'Q_f' in params:
		Q_f = params['Q_f']
	else:
		Q_f = 1.0

	ws, wf, ms, mf = y

	w = ws + wf
	m = ms + mf

	rep_rate = (mu+b*(kappa-(w + delta*m)))

	if rep_rate < 0:
		rep_rate = 0

	return network_system(ws, wf, ms, mf, gamma, beta, rep_rate, mu, xi, epsilon_pcp=epsilon_pcp, Q_f=Q_f)



def F_production_indep_wt(t, y, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	alpha = params['alpha']
	mu = params['mu']

	ws, wf, ms, mf = y

	w = ws + wf
	m = ms + mf

	rep_rate = alpha/w

	return network_system(ws, wf, ms, mf, gamma, beta, rep_rate, mu, xi)

def G_ratiometric_deg(t, y, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	mu = params['mu']
	w_opt = params['w_opt']

	ws, wf, ms, mf = y

	w = ws + wf
	m = ms + mf

	deg_rate = mu*w/w_opt

	return network_system(ws, wf, ms, mf, gamma, beta, mu, deg_rate, xi)

def Z_differential_deg(t, y, params):    
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	alpha = params['alpha']
	mu = params['mu']
	w_opt  = params['w_opt']

	ws, wf, ms, mf = y

	w = ws + wf
	m = ms + mf

	deg_rate = alpha*(w -  w_opt)

	if deg_rate < 0:
		deg_rate = 0

	return network_system(ws, wf, ms, mf, gamma, beta, mu, deg_rate, xi)

def Y_linear_feedback_deg(t, y, params):    
	xi = params['xi']
	gamma = params['gamma']
	beta = params['beta']
	kappa = params['kappa']
	b = params['b']
	mu = params['mu']
	delta = params['delta']
	#c = params['c']

	ws, wf, ms, mf = y

	w = ws + wf
	m = ms + mf

	deg_rate = mu + b*(w + delta*m - kappa)

	if deg_rate < 0:
		deg_rate = 0

	return network_system(ws, wf, ms, mf, gamma, beta, mu, deg_rate, xi)

def X_general_linear_feedback_control(t, y, params):
	xi = params['xi']
	gamma = params['gamma']
	beta = params['beta']
	kappa = params['kappa']
	b = params['b']
	mu = params['mu']
	deltas = params['deltas']

	if 'epsilon_pcp' in params:
		epsilon_pcp = params['epsilon_pcp']
	else:
		epsilon_pcp = 1.0

	ws, wf, ms, mf = y

	rep_rate = (mu+b*(kappa-(deltas[0]*ws+deltas[1]*wf+deltas[2]*ms+deltas[3]*mf)))

	if rep_rate < 0:
		rep_rate = 0

	return network_system(ws, wf, ms, mf, gamma, beta, rep_rate, mu, xi, epsilon_pcp=epsilon_pcp)

#####################################################
# Deterministic steady states (self.ss_definition)
#####################################################

"""
These must be of the form:
	
	def ss_definition(ss_indep_variable, params):
		#... definition ...
		return ws, wf, ms, mf

"""

def B_differential_control_ss(ws, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	alpha = params['alpha']
	mu = params['mu']
	w_opt  = params['w_opt']

	wf = (alpha*w_opt - 2*alpha*ws - mu*xi + np.sqrt(alpha**2*w_opt**2 - 2*alpha*mu*w_opt*xi + mu*(4*alpha*ws*(-1 + xi) + mu*xi**2)))/(2.*alpha)
	mf = -((alpha*w_opt - 2*alpha*ws - mu*xi + np.sqrt(alpha**2*w_opt**2 - 2*alpha*mu*w_opt*xi + mu*(4*alpha*ws*(-1 + xi) + mu*xi**2)))*
		(-(alpha**2*w_opt**2) + beta*(2*mu - mu*xi +\
		np.sqrt(alpha**2*w_opt**2 - 2*alpha*mu*w_opt*xi + mu*(4*alpha*ws*(-1 + xi) + mu*xi**2))) +\
		alpha*w_opt*(-beta - mu + mu*xi + np.sqrt(alpha**2*w_opt**2 - 2*alpha*mu*w_opt*xi + mu*(4*alpha*ws*(-1 + xi) + mu*xi**2))) +\
		mu*(2*alpha*ws - 2*gamma*ws + mu*xi - 2*alpha*ws*xi + 2*gamma*ws*xi +\
		np.sqrt(alpha**2*w_opt**2 - 2*alpha*mu*w_opt*xi + mu*(4*alpha*ws*(-1 + xi) + mu*xi**2)))))/(4.*alpha*gamma*mu*ws*(-1 + xi))
	ms = (-2*beta*mu + alpha*beta*w_opt + alpha*mu*w_opt + alpha**2*w_opt**2 - 2*alpha*mu*ws + 2*gamma*mu*ws + beta*mu*xi - mu**2*xi - 
		alpha*mu*w_opt*xi + 2*alpha*mu*ws*xi - 2*gamma*mu*ws*xi -\
		(beta + mu + alpha*w_opt)*np.sqrt(alpha**2*w_opt**2 - 2*alpha*mu*w_opt*xi + mu*(4*alpha*ws*(-1 + xi) + mu*xi**2)))/\
		(2.*gamma*mu*(-1 + xi))

	return ws, wf, ms, mf

def C_ratiometric_control_ss(ws, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	alpha = params['alpha']
	mu = params['mu']
	w_opt = params['w_opt']

	wf = (alpha*w_opt - (alpha + mu)*ws)/(alpha + mu*xi)
	mf = ((-(alpha*w_opt) + (alpha + mu)*ws)*(alpha**2*w_opt**2*(-beta + gamma*ws) - \
           mu*ws**2*(-1 + xi)*(alpha**2 - alpha*(beta + mu - mu*xi) - mu*(beta + gamma*ws*(-1 + xi) + mu*xi)) + \
           alpha*w_opt*ws*(alpha*(beta + mu + mu*xi) + mu*(-(beta*(-2 + xi)) + 2*gamma*ws*(-1 + xi) + mu*xi*(1 + xi)))))/\
       (gamma*ws*(alpha*w_opt + mu*ws*(-1 + xi))**2*(alpha + mu*xi))
	ms = (alpha**2*w_opt**2*(beta - gamma*ws) + mu*ws**2*(-1 + xi)*\
          (alpha**2 - alpha*(beta + mu - mu*xi) - mu*(beta + gamma*ws*(-1 + xi) + mu*xi)) - \
         alpha*w_opt*ws*(alpha*(beta + mu + mu*xi) + mu*(-(beta*(-2 + xi)) + 2*gamma*ws*(-1 + xi) + mu*xi*(1 + xi))))/\
       (gamma*(alpha*w_opt + mu*ws*(-1 + xi))**2)
	return ws, wf, ms, mf

def E_linear_feedback_control_ss(ms, params):
	xi = params['xi']
	gamma = params['gamma']
	beta = params['beta']
	kappa = params['kappa']
	b = params['b']
	mu = params['mu']
	delta = params['delta']

	ws =  (b*beta**2 + b**2*beta*kappa + b*beta*gamma*kappa + b**2*gamma*kappa**2 + 5*b*beta*mu - beta*gamma*mu + b**2*kappa*mu + \
         3*b*gamma*kappa*mu + 2*b**2*delta*ms*mu - 2*b*gamma*ms*mu - 2*b*delta*gamma*ms*mu + 2*gamma**2*ms*mu + 2*b*mu**2 + \
         2*gamma*mu**2 - b*beta*mu*xi + beta*gamma*mu*xi + b**2*kappa*mu*xi - b*gamma*kappa*mu*xi - 2*b**2*delta*ms*mu*xi + \
         2*b*gamma*ms*mu*xi + 2*b*delta*gamma*ms*mu*xi - 2*gamma**2*ms*mu*xi + 2*b*mu**2*xi - 2*gamma*mu**2*xi - \
         (beta + b*kappa + 2*mu)*np.sqrt(gamma**2*mu**2*(-1 + xi)**2 - \
            2*b*gamma*mu*(-1 + xi)*(-beta + gamma*kappa - 2*(-1 + delta)*gamma*ms + mu + mu*xi) + \
            b**2*(beta**2 - 4*(-1 + delta)*gamma*ms*mu*(-1 + xi) + 2*beta*(gamma*kappa + 3*mu - mu*xi) + (gamma*kappa + mu + mu*xi)**2)))/\
       (2.*(b - gamma)**2*mu*(-1 + xi))

	x1 = np.sqrt(gamma**2*mu**2*(-1 + xi)**2 - 2*b*gamma*mu*(-1 + xi)*(-beta + gamma*kappa - 2*(-1 + delta)*gamma*ms + mu + mu*xi) + \
        b**2*(beta**2 - 4*(-1 + delta)*gamma*ms*mu*(-1 + xi) + 2*beta*(gamma*kappa + 3*mu - mu*xi) + (gamma*kappa + mu + mu*xi)**2))
	
	wf = ((beta*x1)/(b*gamma*(beta + (-1 + delta)*gamma*ms)) - (ms*x1)/(b*(beta + (-1 + delta)*gamma*ms)) - \
         (gamma*(beta + gamma*kappa + 2*mu)**2)/((b - gamma)**2*mu*(-1 + xi)) + (2*x1)/((b - gamma)**2*(-1 + xi)) + \
         x1/((b - gamma)*gamma*(-1 + xi)) + (beta*x1)/((b - gamma)**2*mu*(-1 + xi)) + (kappa*x1)/((b - gamma)*mu*(-1 + xi)) + \
         (gamma*kappa*x1)/((b - gamma)**2*mu*(-1 + xi)) + \
         (-beta**2 - 2*(gamma*kappa + mu)*(gamma*kappa + 2*mu) + beta*(-3*gamma*kappa + 2*mu*(-3 + xi)) + \
            2*(-1 + delta)*gamma*ms*mu*(-1 + xi))/((b - gamma)*mu*(-1 + xi)) - \
         ((beta - gamma*ms)*mu*(-1 + xi))/(b*(beta + (-1 + delta)*gamma*ms)) - (x1*xi)/((b - gamma)*gamma*(-1 + xi)) - \
         (beta*kappa*(beta + gamma*kappa - mu*(-3 + xi)) - 2*(-1 + delta)*delta*gamma*ms**2*mu*(-1 + xi) + \
            ms*((-1 + delta)*gamma**2*kappa**2 + beta*((-1 + delta)*gamma*kappa - delta*mu*(-1 + xi)) + \
               gamma*kappa*mu*(-3 + 2*delta + xi) + delta*mu**2*(-1 + xi**2)))/((beta + (-1 + delta)*gamma*ms)*mu*(-1 + xi)))/2.
	 
	mf =  -(ms*(gamma*mu*(-1 + xi) + b*(beta - gamma*(kappa + 2*ms - 2*delta*ms) - mu*(1 + xi)) - \
            np.sqrt(gamma**2*mu**2*(-1 + xi)**2 - 2*b*gamma*mu*(-1 + xi)*(-beta + gamma*kappa - 2*(-1 + delta)*gamma*ms + mu + mu*xi) + \
              b**2*(beta**2 - 4*(-1 + delta)*gamma*ms*mu*(-1 + xi) + 2*beta*(gamma*kappa + 3*mu - mu*xi) + (gamma*kappa + mu + mu*xi)**2))\
            ))/(2.*b*(beta + (-1 + delta)*gamma*ms))
	return ws, wf, ms, mf

def F_production_indep_wt_ss(wf, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	alpha = params['alpha']
	mu = params['mu']

	if not hasattr(wf,"__len__"): # if scalar
		ws = alpha/mu 
	else:
		ws = alpha/mu * np.ones(len(wf))


	mf =  -((wf*(alpha**3*gamma - beta*mu**3*wf**2 + 2*alpha**2*mu*(mu + gamma*wf) + alpha*mu**2*wf*(-beta + mu + gamma*wf)))/\
         (alpha*gamma*(alpha + mu*wf)**2))

	ms = -((alpha**3*gamma - beta*mu**3*wf**2 + 2*alpha**2*mu*(mu + gamma*wf) + alpha*mu**2*wf*(-beta + mu + gamma*wf))/\
        (gamma*mu*(alpha + mu*wf)**2))

	return ws, wf, ms, mf

def G_ratiometric_deg_ss(ms, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	mu = params['mu']
	w_opt = params['w_opt']

	if not hasattr(ms,"__len__"): # if scalar
		ws = w_opt
	else:
		ws = w_opt * np.ones(len(ms))

	wf = -((w_opt*(gamma*ms + 2*mu + gamma*w_opt))/(-beta + gamma*ms + mu + gamma*w_opt))

	mf = -((ms*(gamma*ms + 2*mu + gamma*w_opt))/(-beta + gamma*ms + mu + gamma*w_opt))

	return ws, wf, ms, mf


def Z_differential_deg_ss(ws, params):
	beta = params['beta']
	gamma = params['gamma']
	xi = params['xi']
	alpha = params['alpha']
	mu = params['mu']
	w_opt  = params['w_opt']

	wf = (ws*(mu + alpha*w_opt - alpha*ws))/(-mu + alpha*ws)
	mf = -(((mu + alpha*w_opt - alpha*ws)*(-((beta + mu)*(mu - alpha*ws)) + alpha*w_opt*(-beta + mu + gamma*ws)))/
         (alpha*gamma*w_opt*(-mu + alpha*ws)))
	ms = ((beta + mu)*(mu - alpha*ws) + alpha*w_opt*(beta - mu - gamma*ws))/(alpha*gamma*w_opt)

	return ws, wf, ms, mf

def Y_linear_feedback_deg_ss(ms, params):    
	xi = params['xi']
	gamma = params['gamma']
	beta = params['beta']
	kappa = params['kappa']
	b = params['b']
	mu = params['mu']
	delta = params['delta']
	
	#c = params['c']
	# ws = (mu*(beta - beta*c + c*gamma*ms + mu + c*mu) - \
	# 	b*(-(beta*kappa) + beta*delta*ms + gamma*kappa*ms + kappa*mu + delta*ms*mu))/\
	# 	(-(c*gamma*mu) + b*(beta + gamma*kappa + mu))
	# wf = ((-((-1 + c)*gamma*mu) + b*(gamma*(kappa + ms - delta*ms) + 2*mu))*\
	# 	(mu*(beta - beta*c + c*gamma*ms + mu + c*mu) - \
	# 	b*(-(beta*kappa) + beta*delta*ms + gamma*kappa*ms + kappa*mu + delta*ms*mu)))/\
	# 	((b*(beta + (-1 + delta)*gamma*ms - mu) - gamma*mu)*(-(c*gamma*mu) + b*(beta + gamma*kappa + mu)))
	# mf = -((ms*((-1 + c)*gamma*mu - b*(gamma*(kappa + ms - delta*ms) + 2*mu)))/(b*(beta + (-1 + delta)*gamma*ms - mu) - gamma*mu))

	ws = (mu*(gamma*ms + 2*mu) - b*(-(beta*kappa) + beta*delta*ms + gamma*kappa*ms + kappa*mu + delta*ms*mu))/\
		(-(gamma*mu) + b*(beta + gamma*kappa + mu))
	wf = -((b*(gamma*(kappa + ms - delta*ms) + 2*mu)*(-(mu*(gamma*ms + 2*mu)) + \
		b*(-(beta*kappa) + beta*delta*ms + gamma*kappa*ms + kappa*mu + delta*ms*mu)))/\
		((b*(beta + (-1 + delta)*gamma*ms - mu) - gamma*mu)*(-(gamma*mu) + b*(beta + gamma*kappa + mu)))) 
	mf = -((b*ms*(-(gamma*(kappa + ms - delta*ms)) - 2*mu))/(b*(beta + (-1 + delta)*gamma*ms - mu) - gamma*mu))

	return ws, wf, ms, mf

#########################################
# Jacobian matrices
#########################################

"""
Evaluate the Jacobian matrix of a feedback control at a particular state. Of the form

def jac_matrix_fcn(state, params):
	...
	return jac

:param state: A list of doubles, using the parameter ordering convention ['ws','wf','ms','mf']
:param params: A dict of parameters of the control
"""

def E_linear_feedback_control_jac(state, params):
	xi = params['xi']
	gamma = params['gamma']
	beta = params['beta']
	kappa = params['kappa']
	b = params['b']
	mu = params['mu']
	delta = params['delta']

	ws, wf, ms, mf = state

	J=np.array(
		[[-(gamma*mf) - gamma*ms - 2*mu - gamma*wf - b*(kappa - delta*mf - delta*ms - wf - ws) + b*ws - 2*gamma*ws,beta + b*ws - gamma*ws,
		b*delta*ws - gamma*ws,
		b*delta*ws - gamma*ws],
		[gamma*mf + gamma*ms + gamma*wf + 2*(mu + b*(kappa - delta*mf - delta*ms - wf - ws)) + 2*gamma*ws - b*(wf + 2*ws),
		-beta + mu + b*(kappa - delta*mf - delta*ms - wf - ws) + gamma*ws - b*(wf + 2*ws) - mu*xi,
		gamma*ws - b*delta*(wf + 2*ws),
		gamma*ws - b*delta*(wf + 2*ws)],
		[b*ms - gamma*ms,
		b*ms - gamma*ms,
		-(gamma*mf) + b*delta*ms - 2*gamma*ms - 2*mu - gamma*wf - b*(kappa - delta*mf - delta*ms - wf - ws) - gamma*ws,
		beta + b*delta*ms - gamma*ms],
		[gamma*ms - b*(mf + 2*ms),
		gamma*ms - b*(mf + 2*ms),
		gamma*mf + 2*gamma*ms - b*delta*(mf + 2*ms) + gamma*wf + 2*(mu + b*(kappa - delta*mf - delta*ms - wf - ws)) + gamma*ws,
		-beta + gamma*ms - b*delta*(mf + 2*ms) + mu + b*(kappa - delta*mf - delta*ms - wf - ws) - mu*xi]]
		)

	return J


########################################
# Feedback Control object definitions
########################################

class FeedbackControl(object):
	""" Deterministic analysis of a feedback control with network dynamics


	:param network_defn: A function, definition of the deterministic dynamics of the form function(t, y, params)
	:param params: A dict, parameters of control law
	:param initial_state: A list of floats, copy numbers of each species at t=0. By convention [ws, wf, ms, mf]
	:param ode_integrator_method: A string, the integration method for scipy.integrate.ode
	:param max_int_steps: An int, number of steps allowed per iteration of integrator
	:param TMAX: A float, Maximum time for integrator
	:param dt: A float, time step of integrator 
	:param species_lim: A float, upper bound of initial copy number per species for trajectory plot
	:param n_traj: An int, number of trajectories to plot
	:param plotdir: A string, output directory of plots
	:param phaseportraitname: A string, name of phase portrait
	:param plotextensions: A list of strings, extensions to save the phase portrait
	"""

	def __init__(self, network_defn, params, initial_state=[0,0,0,0], 
					ode_integrator_method="vode", max_int_steps=10**10, TMAX = 200.0, dt = 1.0, 
					species_lim = 500, n_traj = 20, 
					plotdir = os.getcwd(), phaseportraitname = 'phase_portrait', plotextensions = ['svg','png']):
		self.network_defn = network_defn 
		self.params = params 
		
		self.initial_state = np.array(initial_state) 
		self.ode_integrator_method = ode_integrator_method 
		self.max_int_steps = max_int_steps  
		self.TMAX = TMAX 
		self.dt = dt 
		
		self.species_lim = species_lim 
		self.n_traj = n_traj 

		self.plotdir = plotdir 
		self.phaseportraitname = phaseportraitname 
		self.plotextensions = plotextensions 

		self.t = 0.0
		self.state = self.initial_state
		self.state_trajectory = None
		self.state_trajectory_set = None

		self.param_convention = ['ws', 'wf', 'ms', 'mf'] # global ordering convention of parameters
		self.n_species = len(self.param_convention)
		self.param_convention_map = {}
		for i, s in enumerate(self.param_convention):
			self.param_convention_map[s] = i

	def __str__(self):
		return 'General feedback control object'

	
	
	def print_state(self):
		"""
		Print the current state of the system
		"""
		for i in range(4):
			print(self.param_convention[i]+' = '+str(self.state[i]))

	def make_trajectory(self):
		"""
		Make a single trajectory of ODE system by numerical integration
		"""
		self.state_trajectory = np.nan*np.zeros((int(self.TMAX/self.dt)+1,self.n_species+1)) # preallocate trajectory to nan
		
		# Set up ODE			
		r = ode(self.network_defn).set_integrator(self.ode_integrator_method, nsteps=self.max_int_steps)
		r.set_initial_value(self.initial_state, 0.0).set_f_params(self.params)
		self.state_trajectory[0,:] = np.insert(self.initial_state, 0, 0.0) # initial condition

		i = 1 # counter

		while r.successful() and r.t < self.TMAX:
			r.integrate(r.t+self.dt)

			self.t = r.t # update instantaneous time
			self.state = r.y # update instantaneous state
			self.state_trajectory[i,:] = np.insert(r.y, 0, r.t) # append state to trajectory
			i+=1

		if r.successful() == False:
			print('integration error')
			return
	  	 

	def make_random_trajectories(self):	
		"""
		Creates a 3D array where the indicies are:
			0 : time
			1 : state
			2 : trajectory repetition
		for random initial conditions
		"""	
		self.state_trajectory_set = np.nan*np.zeros((int(self.TMAX/self.dt)+1,self.n_species+1,self.n_traj)) # Tensor of trajectories preallocate to nan
		for k in range(self.n_traj):
			print(k)
			# randomize IC
			self.initial_state = np.random.uniform(0,self.species_lim, size = self.n_species)
			self.make_trajectory()
			self.state_trajectory_set[:,:,k] = self.state_trajectory

	def get_h_from_state(self, x, state_includes_time=True):
		msi = self.param_convention_map['ms']
		mfi = self.param_convention_map['mf']

		if state_includes_time:
			return (x[msi+1]+x[mfi+1])/float(np.sum(x[1:]))
		else:
			return (x[msi]+x[mfi])/float(np.sum(x))

	def get_fs_from_state(self, x, state_includes_time=True):
		msi = self.param_convention_map['ms']
		wsi = self.param_convention_map['ws']

		if state_includes_time:
			return (x[msi+1]+x[wsi+1])/float(np.sum(x[1:]))
		else:
			return (x[msi]+x[wsi])/float(np.sum(x))

	def get_n_from_state(self, x, state_includes_time=True):
		if state_includes_time:
			return float(np.sum(x[1:]))
		else:
			return float(np.sum(x))

	def get_theta_from_state(self, x, state_includes_time=True):
		wsi = self.param_convention_map['ws']
		mfi = self.param_convention_map['mf']

		if state_includes_time:
			return (x[mfi+1]+x[wsi+1])/float(np.sum(x[1:]))
		else:
			return (x[mfi]+x[wsi])/float(np.sum(x))


	def add_to_trajectory_plot(self, axs, k):
		"""
		Add a single ODE trajectory from self.state_trajectory_set to a (1,3) plot containing
			1. w/m phase portrait
			2. s/f phase portrait
			3. copy number against time
		"""
		if self.state_trajectory_set is None:
			raise Exception('state_trajectory_set does not exist. Run FeedbackControl.make_random_trajectories()')

		final_state = self.state_trajectory_set[-1,:,k]
		init_state = self.state_trajectory_set[0,:,k]

		# Unpack to make code more readable
		t_sol = self.state_trajectory_set[:,0,k]; ws_sol = self.state_trajectory_set[:,1,k]; wf_sol = self.state_trajectory_set[:,2,k]; ms_sol = self.state_trajectory_set[:,3,k]; mf_sol = self.state_trajectory_set[:,4,k]

		ws_init = init_state[1]; wf_init = init_state[2]; ms_init = init_state[3]; mf_init = init_state[4];			
		t_final = final_state[0]; ws_final = final_state[1]; wf_final = final_state[2]; ms_final = final_state[3]; mf_final = final_state[4];	

		if k == 0:
			ax = axs[0]		
			ax.plot(ws_sol + wf_sol, ms_sol + mf_sol, '-b', alpha = 0.2, label = 'Trajectory')
			ax.plot(ws_init + wf_init, ms_init + mf_init, '.b', alpha = 0.2, label = 'Initial condition')
			ax.plot(ws_final + wf_final, ms_final + mf_final, 'sk', alpha = 0.2, label = 'Final condition')

			ax = axs[1]
			ax.plot(ws_sol + ms_sol, wf_sol + mf_sol, '-b', alpha = 0.2, label = 'Trajectory')
			ax.plot(ws_init + ms_init, wf_init + mf_init, '.b', alpha = 0.2, label = 'Initial condition')
			ax.plot(ws_final + ms_final, wf_final + mf_final, 'sk', alpha = 0.2, label = 'Final condition')

			ax = axs[2]
			ax.plot(t_sol, ws_sol + ms_sol + wf_sol + mf_sol, '-b', alpha = 0.2, label = 'Trajectory')
			ax.plot(t_sol[0], ws_init + ms_init + wf_init + mf_init, '.b', alpha = 0.2, label = 'Initial condition')
			ax.plot(t_final, ws_final + ms_final + wf_final + mf_final, 'sk', alpha = 0.2, label = 'Final condition')
		else:
			ax = axs[0]		
			ax.plot(ws_sol + wf_sol, ms_sol + mf_sol, '-b', alpha = 0.2)
			ax.plot(ws_final + wf_final, ms_final + mf_final, 'sk', alpha = 0.2)
			ax.plot(ws_init + wf_init, ms_init + mf_init, '.b', alpha = 0.2)

			ax = axs[1]
			ax.plot(ws_sol + ms_sol, wf_sol + mf_sol, '-b', alpha = 0.2)
			ax.plot(ws_final + ms_final, wf_final + mf_final, 'sk', alpha = 0.2)
			ax.plot(ws_init + ms_init, wf_init + mf_init, '.b', alpha = 0.2)

			ax = axs[2]
			ax.plot(t_sol, ws_sol + ms_sol + wf_sol + mf_sol, '-b', alpha = 0.2)
			ax.plot(t_sol[0], ws_init + ms_init + wf_init + mf_init, '.b', alpha = 0.2)
			ax.plot(t_final, ws_final + ms_final + wf_final + mf_final, 'sk', alpha = 0.2)
	

	def make_trajetory_plot(self):
		"""
		Make a plot of trajectories from self.state_trajectory_set to a (1,3) plot containing
			1. w/m phase portrait
			2. s/f phase portrait
			3. copy number against time
		"""
		if self.state_trajectory_set is None:
			raise Exception('state_trajectory_set does not exist. Run FeedbackControl.make_random_trajectories()')

		

		
		fig, axs = plt.subplots(1,3, figsize = (15,5))
		axs = axs.ravel()		

		for k in range(self.n_traj):
			self.add_to_trajectory_plot(axs, k)

		ax = axs[0]
		ax.set_xlabel('Wild-type, $w_s + w_f$')
		ax.set_ylabel('Mutants, $m_s + m_f$')
		ax.legend(prop = {'size':10})

		ax = axs[1]
		ax.set_xlabel('Singletons, $w_s + m_s$')
		ax.set_ylabel('Fused, $w_f + m_f$')
		ax.legend(prop = {'size':10})

		ax = axs[2]
		ax.set_xlabel('Time (days)')
		ax.set_ylabel('Total copy number')
		ax.legend(prop = {'size':10})

		# Make it look fancy
		fmt = matplotlib.ticker.StrMethodFormatter("{x}")
		for ax in axs:
			ax.xaxis.set_major_formatter(fmt)
			ax.yaxis.set_major_formatter(fmt) 
			ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))   
			ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))   

		plt.tight_layout()

		for extn in self.plotextensions:
			plt.savefig(self.plotdir+'/'+self.phaseportraitname+'.'+extn)

	def check_h_cons(self, plotname=None, let_h_small = False):
		if self.state_trajectory_set is None:
			raise Exception('state_trajectory_set does not exist. Run FeedbackControl.make_random_trajectories()')

		ics = self.state_trajectory_set[0,1:,:]
		fcs = self.state_trajectory_set[-1,1:,:]


		h_i = (ics[2,:]+ics[3,:])/np.sum(ics, axis=0)
		h_f = (fcs[2,:]+fcs[3,:])/np.sum(fcs, axis=0)
		max_dev = np.max(abs(h_i-h_f))
		plotting_lim = np.max([0.01,1.1*max_dev])
		fig, ax = plt.subplots(1,1, figsize=(5,5))
		ax.plot(h_i, h_f-h_i, 'kx')

		if not let_h_small:
			ax.set_ylim([-plotting_lim,plotting_lim])

		ax.set_xlabel('Initial heteroplasmy')
		ax.set_ylabel('Final heteroplasmy -\n initial heteroplasmy')

		plt.tight_layout()

		if plotname is not None:
			for extn in self.plotextensions:
				plt.savefig(self.plotdir+'/'+plotname+'.'+extn)
		
		
class SoluableFeedbackControl(FeedbackControl):
	"""
	Subclass of FeedbackControl, whose steady state is soluable

	:param ss_definition: function
	:param ss_indep_variable: str
	:param max_indep_var_val: double
	:param phys_ss_is_point: Bool
	"""

	def __str__(self):
		return 'Feedback control where SS is analytically tractible'

	def __init__(self, ss_definition, ss_indep_variable, 
		max_indep_var_val = None, phys_ss_is_point = False,
		**kwargs):
		super(SoluableFeedbackControl, self).__init__(**kwargs) # ensure child class calls the correct parent class using super()

		self.ss_definition = ss_definition # a function defining the steady state
		
		if ss_indep_variable not in self.param_convention:
			raise Exception("ss_indep_variable must be an element of ['ws', 'wf', 'ms', 'mf']")
		self.ss_indep_variable = ss_indep_variable 
		self.ss_indep_variable_index = self.param_convention_map[ss_indep_variable]

		# For plotting the SS line
		self.min_state = np.zeros(4)
		self.max_state = np.zeros(4)
		self.max_indep_var_val = max_indep_var_val # maximum value of the independent variable for plotting
		self.phys_ss_is_point = phys_ss_is_point # True if plotting fused vs singletons results in a point

	def find_extreme_state(self):
		"""
		State-wise min/max of each state variable over all trajectories at SS
		"""
		if self.state_trajectory_set is None:
			raise Exception('state_trajectory_set does not exist. Run FeedbackControl.make_random_trajectories()')

		self.min_state = np.array([self.state_trajectory_set[-1,i+1,:].min() for i in range(4)]) 
		self.max_state = np.array([self.state_trajectory_set[-1,i+1,:].max() for i in range(4)]) 
	
	
	def add_to_trajectory_plot(self, axs, k):
		"""
		Add a single ODE trajectory from self.state_trajectory_set to a (1,3) plot containing
			1. w/m phase portrait
			2. s/f phase portrait
			3. copy number against time
		"""
		if self.state_trajectory_set is None:
			raise Exception('state_trajectory_set does not exist. Run FeedbackControl.make_random_trajectories()')
		

		final_state = self.state_trajectory_set[-1,:,k]
		init_state = self.state_trajectory_set[0,:,k]

		# Unpack to make code more readable
		t_sol = self.state_trajectory_set[:,0,k]; ws_sol = self.state_trajectory_set[:,1,k]; wf_sol = self.state_trajectory_set[:,2,k]; ms_sol = self.state_trajectory_set[:,3,k]; mf_sol = self.state_trajectory_set[:,4,k]

		ws_init = init_state[1]; wf_init = init_state[2]; ms_init = init_state[3]; mf_init = init_state[4];			
		t_final = final_state[0]; ws_final = final_state[1]; wf_final = final_state[2]; ms_final = final_state[3]; mf_final = final_state[4];	

		# Get deterministic steady state
		ws_det_ss, wf_det_ss, ms_det_ss, mf_det_ss = self.ss_definition(final_state[self.ss_indep_variable_index+1], self.params) # +1 since first index is time in self.state_trajectory_set
		n_det_ss = ws_det_ss+wf_det_ss+ms_det_ss+mf_det_ss

		if k == 0:
			ax = axs[0]		
			l=ax.plot(ws_sol + wf_sol, ms_sol + mf_sol, '-b', alpha = 0.2, label = 'Trajectory')
			ax.plot(ws_init + wf_init, ms_init + mf_init, '.b', alpha = 0.2, label = 'Initial condition')
			ax.plot(ws_final + wf_final, ms_final + mf_final, 'sk', alpha = 0.2, label = 'Final condition')
			add_arrow(l[0])


			ax = axs[1]
			l=ax.plot(ws_sol + ms_sol, wf_sol + mf_sol, '-b', alpha = 0.2, label = 'Trajectory')
			ax.plot(ws_init + ms_init, wf_init + mf_init, '.b', alpha = 0.2, label = 'Initial condition')
			ax.plot(ws_final + ms_final, wf_final + mf_final, 'sk', alpha = 0.2, label = 'Final condition')
			add_arrow(l[0])

			ax = axs[2]
			ax.plot(t_sol, ws_sol + ms_sol + wf_sol + mf_sol, '-b', alpha = 0.2, label = 'Trajectory')
			ax.plot(t_sol[0], ws_init + ms_init + wf_init + mf_init, '.b', alpha = 0.2, label = 'Initial condition')
			ax.plot(t_final, ws_final + ms_final + wf_final + mf_final, 'sk', alpha = 0.2, label = 'Final condition')

			ax.plot(t_final, n_det_ss, 'xr', alpha = 0.5, label = 'Final condition theory')
		else:
			ax = axs[0]		
			l=ax.plot(ws_sol + wf_sol, ms_sol + mf_sol, '-b', alpha = 0.2)
			ax.plot(ws_final + wf_final, ms_final + mf_final, 'sk', alpha = 0.2)
			ax.plot(ws_init + wf_init, ms_init + mf_init, '.b', alpha = 0.2)
			add_arrow(l[0])

			ax = axs[1]
			l=ax.plot(ws_sol + ms_sol, wf_sol + mf_sol, '-b', alpha = 0.2)
			ax.plot(ws_final + ms_final, wf_final + mf_final, 'sk', alpha = 0.2)
			ax.plot(ws_init + ms_init, wf_init + mf_init, '.b', alpha = 0.2)
			add_arrow(l[0])

			ax = axs[2]
			ax.plot(t_sol, ws_sol + ms_sol + wf_sol + mf_sol, '-b', alpha = 0.2)
			ax.plot(t_sol[0], ws_init + ms_init + wf_init + mf_init, '.b', alpha = 0.2)
			ax.plot(t_final, ws_final + ms_final + wf_final + mf_final, 'sk', alpha = 0.2)

			ax.plot(t_final, n_det_ss, 'xr', alpha = 0.5)

		if k==self.n_traj-1:
			# Plot theory

			# If the maximum value of the independent variable is not defined (default)
			# then take the largest steady state value of the variable as a guess.
			if self.max_indep_var_val is None:
				self.find_extreme_state()
				self.max_indep_var_val = 1.1*self.max_state[self.ss_indep_variable_index]
				self.min_indep_var_val = 0.9*self.min_state[self.ss_indep_variable_index]
				indep_var_range = np.linspace(self.min_indep_var_val, self.max_indep_var_val)
			else:
				indep_var_range = np.linspace(0, self.max_indep_var_val)

			wsss_space, wfss_space, msss_space, mfss_space = self.ss_definition(indep_var_range,self.params)

			# strip negative values
			stacked_space = np.vstack((wsss_space,msss_space,mfss_space,wfss_space))
			stacked_space_strip_neg = stacked_space[:,np.all(stacked_space>=0,axis=0)]
			wsss_space,msss_space,mfss_space,wfss_space = np.vsplit(stacked_space_strip_neg,4)
			wsss_space = wsss_space.ravel()
			msss_space = msss_space.ravel()
			wfss_space = wfss_space.ravel()
			mfss_space = mfss_space.ravel()



			axs[0].plot(wsss_space+wfss_space,msss_space+mfss_space,'-r',label='Theory')

			if self.phys_ss_is_point:
				axs[1].plot(wsss_space+msss_space,wfss_space+mfss_space,'.r',label='Theory')
			else:
				axs[1].plot(wsss_space+msss_space,wfss_space+mfss_space,'-r',label='Theory')

	


	
##############################################################
# Multiple serial feedback control for running on HPC
##############################################################

def get_sec(time_str):
	h, m, s = time_str.split(':')
	return int(h)*3600 + int(m)*60 + int(s)

class FeedbackControlHPC(FeedbackControl):
	"""
	To integrate a feedback control in parallel on the Imperial College High Performance Computing facility.

	Since the amount of time it takes to integrate an ODE system is uncertain, take as much time as possible,
	then when the walltime is approached, exit.

	:param walltime: A string, of the form 'hh:mm:ss' which is the walltime for the current job
	:param starttime: A float, the time when the job started using time.time()
	:param timetol: A float between 0 and 1, denoting the fraction of the walltime the integration may use
	"""
	def __init__(self, walltime, starttime, timetol=0.9,
		**kwargs):
		super(FeedbackControlHPC, self).__init__(**kwargs) # ensure child class calls the correct parent class using super()
		self.walltime = get_sec(walltime)
		self.starttime = starttime
		self.timetol = timetol

	def make_trajectory(self):
		"""
		Make a single trajectory of ODE system by numerical integration, exiting if walltime is approached
		"""
		self.state_trajectory = np.nan*np.zeros((int(self.TMAX/self.dt)+1,5)) # preallocate trajectory to nan
		
		# Set up ODE			
		r = ode(self.network_defn).set_integrator(self.ode_integrator_method, nsteps=self.max_int_steps)
		r.set_initial_value(self.initial_state, 0.0).set_f_params(self.params)
		self.state_trajectory[0,:] = np.insert(self.initial_state, 0, 0.0) # initial condition

		i = 1 # counter

		elapsed_time = time.time() - self.starttime
		while r.successful() and r.t < self.TMAX and elapsed_time < self.timetol*self.walltime:
			elapsed_time = time.time() - self.starttime

			r.integrate(r.t+self.dt)

			self.t = r.t # update instantaneous time
			self.state = r.y # update instantaneous state
			self.state_trajectory[i,:] = np.insert(r.y, 0, r.t) # append state to trajectory
			i+=1

