import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime
import pickle
from finite_difference import *

T=5e6 # make this 2 powers above nt
Nt = int(1e4) # was 1e5
Nx=100
nframes = 200

def calc_rmse(y1,y2, relative=False):
    '''
    presume y is longer
    '''
    x1 = np.linspace(0,1,y1.shape[1], endpoint=False)
    x2 = np.linspace(0,1,y2.shape[1]+1)
    y2_ = np.append(y2, y2[:,:1], axis=1)
    f = interp1d(x2,y2_, kind='quadratic', axis=1)
    y2_int = f(x1)
    if relative:
        rmse = (((y1 - y2_int)/(0.5*(y1 + y2_int)))**2).mean()**0.5
    else:
        rmse = ((y1 - y2_int)**2).mean()**0.5
    return rmse

def calc_all_rsmes(x, base, relative=False):
    if type(base)!=list:
        base = [base for _ in range(len(x))]
    return [calc_rmse(x[i], base[i], relative) for i in range(len(x))]

def save_lists(filename,
    u_ctcs_list, h_ctcs_list, 
    u_ftbtcs_list, h_ftbtcs_list
              ):
    # save the data 
    all_lists={
        'u_ctcs_list':u_ctcs_list, 
        'h_ctcs_list':h_ctcs_list, 
        'u_ftbtcs_list':u_ftbtcs_list,
        'habs_ftbtcs_list':h_ftbtcs_list
    }

    with open(filename, 'wb') as handle:
        pickle.dump(all_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

# data stores
u_ctcs_list = []
h_ctcs_list = []
u_ftbtcs_list = []
h_ftbtcs_list = []
    

factors = [f for f in 10.**np.arange(-1,2.75, 0.25)]
# [f for f in 10.**np.arange(-1,2)] 
    
for factor in factors:
    
    # set resolution this round
    nt = int(Nt*factor)
    nx = int(Nx*factor)
    
    # derived quantities
    dtbydx = (T/nt)/(diam/nx)
    
    print('-'*10)
    print(datetime.now())
    print("dt/dx:", dtbydx)
    courant = dtbydx*np.sqrt(2000*9.81)
    print("c:", courant)
    print('nx:', nx)
    print('nt:', nt)
    
    # run simulation
    u_ctcs, h_ctcs = run_ctcs(nx, nt, nframes, g=9.81, T=T)
    u_ftbtcs, h_ftbtcs = run_ftbtcs(nx, nt, nframes, g=9.81, T=T)
    
    u_ctcs_list.append(u_ctcs)
    h_ctcs_list.append(h_ctcs)
    u_ftbtcs_list.append(u_ftbtcs)
    h_ftbtcs_list.append(h_ftbtcs)
    
    # save
    print('saving : results_data_{:.2f}.pkl'.format(factor))
    save_lists(
        'results_data_{:.2f}.pkl'.format(factor),
        u_ctcs_list, h_ctcs_list, 
        u_ftbtcs_list, h_ftbtcs_list
    )
    

nxs = [Nx*f for f in factors]
    
def plotter(ax, factors, y, label, ylabel="RMSE"):
    ax.loglog(factors,y, label=label, marker='.')
    ax.set_ylabel(ylabel)
    ax.legend()

# Consistency with self

# calculate RSME for all the runs from the best run
u_ctcs_rmse = calc_all_rsmes(u_ctcs_list[:-1], u_ctcs_list[-1])
h_ctcs_rmse = calc_all_rsmes(h_ctcs_list[:-1], h_ctcs_list[-1])
u_ftbtcs_rmse = calc_all_rsmes(u_ftbtcs_list[:-1], u_ftbtcs_list[-1])
h_ftbtcs_rmse = calc_all_rsmes(h_ftbtcs_list[:-1], h_ftbtcs_list[-1])

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(8,6))
plotter(ax1,nxs[:-1],u_ctcs_rmse, label="$u_{CTCS}$",
        ylabel="RMSE $({u_{method}^{maxRes}})$")
plotter(ax1,nxs[:-1],u_ftbtcs_rmse, label="$u_{FTBTCS}$",
       ylabel="RMSE $({u_{method}^{maxRes}})$")
plotter(ax2,nxs[:-1],h_ftbtcs_rmse, label="$h_{FTBTCS}$",
        ylabel="RMSE $({h_{method}^{maxRes}})$")
plotter(ax2,nxs[:-1],h_ctcs_rmse, label="$h_{CTCS}$",
       ylabel="RMSE $({h_{method}^{maxRes}})$")
ax2.set_xlabel("$n_x$ ($n_t = 100n_x$) c = {:.3f}".format(courant))
plt.savefig('rmse_self_logy.png', dpi=400)
plt.show()

# Consistency with each other
# calculate RMSE between methods at same resolution
u_comp_rmse = calc_all_rsmes(u_ctcs_list, u_ftbtcs_list)
h_comp_rmse = calc_all_rsmes(h_ctcs_list, h_ftbtcs_list)

fig, ax1 = plt.subplots(1,1, sharex=True, figsize=(8,6))
plotter(ax1,nxs,u_comp_rmse, label="$u$",
        ylabel="RMSE")
plotter(ax1,nxs,h_comp_rmse, label="$h$",
       ylabel="RMSE")

ax1.set_xlabel("$n_x$ ($n_t = 100n_x$) c = {:.3f}".format(courant))
plt.savefig('rmse_comparison_loglog.png', dpi=400)
plt.show()
    
        
    
    