import numpy as np
from finite_difference import *

T=5e6 # make this 2 powers above nt
Nt = int(1e4) # was 1e5
Nx=100
nframes = 200

def calc_rmse(x,y):
    '''
    presume y is longer
    '''
    minlen = min(x.shape[1], y.shape[1])
    maxlen = max(x.shape[1], y.shape[1])
    skip = maxlen//minlen
    return ((y[:,::skip][:,:len(x)] -
             x[:,:len(y[:,::skip])])**2).mean()**0.5

def calc_all_rsmes(x, base, factors):
    return [calc_rmse(x[i], base) for i in range(len(factors)-1)]

# data stores
u_ctcs_list = []
h_ctcs_list = []
u_ftbtcs_list = []
h_ftbtcs_list = []
    
factors = [f for f in 10.**np.arange(-1,3, 0.5)]
# [f for f in 10.**np.arange(-1,2)] 
    
for factor in factors:
    
    # set resolution this round
    nt = int(Nt*factor)
    nx = int(Nx*factor)
    
    # derived quantities
    dtbydx = (T/nt)/(diam/nx)
    nsave = nt//nframes
    
    print('-'*10)
    print("dt/dx:", dtbydx)
    courant = dtbydx*np.sqrt(2000*9.81)
    print("c:", courant)
    print('nx:', nx)
    print('nx:', nt)
    
    # run simulation
    u_ctcs, h_ctcs = run_ctcs(nx, nt, nsave, g=9.81, T=T)
    u_ftbtcs, h_ftbtcs = run_ftbtcs(nx, nt, nsave, g=9.81, T=T)
    
    u_ctcs_list.append(u_ctcs)
    h_ctcs_list.append(h_ctcs)
    u_ftbtcs_list.append(u_ftbtcs)
    h_ftbtcs_list.append(h_ftbtcs)

# calculate RSME for all the runs from the best run
u_ctcs_rmse = calc_all_rsmes(u_ctcs_list, u_ctcs_list[-1], factors)
h_ctcs_rmse = calc_all_rsmes(h_ctcs_list, h_ctcs_list[-1], factors)
u_ftbtcs_rmse = calc_all_rsmes(u_ftbtcs_list, u_ftbtcs_list[-1], factors)
h_ftbtcs_rmse = calc_all_rsmes(h_ftbtcs_list, h_ftbtcs_list[-1], factors)

nxs = [Nx*f for f in factors[:-1]]
    
def plotter(ax, factors, y, label, ylabel="RMSE"):
    ax.semilogx(factors,y, label=label)
    ax.set_ylabel(ylabel)
    ax.legend()

# plot the thing
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

plotter(ax1,nxs,u_ctcs_rmse, label="$u_{CTCS}$",
        ylabel="$RMSE_{u_{method}^{maxRes}}$")
plotter(ax1,nxs,u_ftbtcs_rmse, label="$u_{FTBTCS}$",
       ylabel="$RMSE_{u_{method}^{maxRes}}$")
plotter(ax2,nxs,h_ftbtcs_rmse, label="$h_{FTBTCS}$",
        ylabel="$RMSE_{h_{method}^{maxRes}}$")
plotter(ax2,nxs,h_ctcs_rmse, label="$h_{CTCS}$",
       ylabel="$RMSE_{h_{method}^{maxRes}}$")

ax2.set_xlabel("$n_x$ ($n_t = 100n_x$) c = {:.3f}".format(courant))
plt.show()
    
        
    
    