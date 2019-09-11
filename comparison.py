import numpy as np
from finite_difference import *

T=5e6 # make this 2 powers above nt
Nt = int(1e5)
Nx=100
gifname = "ctcs-ftbtcs1"

u_ctcs_list = []
h_ctcs_list = []
u_ftbtcs_list = []
h_ftbtcs_list = []
    
for factor in 10.**np.arange(-1,2):
    nt = int(Nt*factor)
    nx = int(Nx*factor)
    # derived quantities
    dtbydx = (T/nt)/(diam/nx)
    print('-'*10)
    print("dt/dx:", dtbydx)
    print("c:", dtbydx*np.sqrt(2000*9.81))
    print('nx:', nx)
    nsave = nt//nframes
    
    # run simulation
    u_ctcs, h_ctcs = run_ctcs(nx, nt, nsave, g=9.81, T=T)
    u_ftbtcs, h_ftbtcs = run_ftbtcs(nx, nt, nsave, g=9.81, T=T)
    
    u_ctcs_list.append(u_ctcs)
    h_ctcs_list.append(h_ctcs)
    
    u_ftbtcs_list.append(u_ftbtcs)
    h_ftbtcs_list.append(h_ftbtcs)