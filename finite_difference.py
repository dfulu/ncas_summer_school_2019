import numpy as np
import matplotlib.pyplot as plt
import animatplot as amp

# constants
r = 6371e3 # radius of earth in meters
diam = 2*np.pi*r


def evolve_u_ctcs(h, u, u_minus, dtbydx, g):
    """
    Args:
        h: h at current time
        u: u at current time
        u_minus : u at previous time step
    returns:
        u at next time step,
    """
    term1 = -g * (np.roll(h,-1) - np.roll(h, 1)) 
    term2 = -u * (np.roll(u,-1) - np.roll(u, 1))
    return dtbydx*(term1+term2)+u_minus

def evolve_h_ctcs(h, u, h_minus, dtbydx):
    """
    Args:
        h: h at current time
        u: u at current time
        h_minus : h at previous time step
    returns:
        h at next time step,
    """
    term1 = - h * (np.roll(u,-1) - np.roll(u, 1))
    term2 = - u * (np.roll(h,-1) - np.roll(h, 1))
    return dtbydx*(term1+term2)+h_minus


def evolve_u_ftcs(h, u, u_minus, dtbydx, g):
    """
    Args:
        h: h at current time
        u: u at current time
        u_minus : u at previous time step
    returns:
        u at next time step,
    """
    term1 = -g * (np.roll(h,-1) - np.roll(h, 1))/2 
    term2 = -u * (np.roll(u,-1) - np.roll(u, 1))/2
    return dtbydx*(term1+term2) + u

def evolve_h_btcs(h, u_plus, dtbydx):
    """
    Args:
        h: h at current time
        u: u at current time
        h_minus : h at previous time step
    returns:
        h at next time step,
    """
    term1 = - h * (np.roll(u_plus,-1) - np.roll(u_plus, 1))/2
    term2 = - u_plus * (np.roll(h,-1) - np.roll(h, 1))/2
    return dtbydx*(term1+term2) + h


def run_ctcs(nx, nt, nsave, g=9.81, T=1e4):
    """
    Runs CTCS on an A-grid
    
    Ags:
        nx: number of x steps
        nt: number of time steps
        nsave: number of timestepps to save
    """
    
    # set initial conditions
    u0 = np.cos(np.linspace(0,1,nx)*2*np.pi)
    h0 = np.cos(np.linspace(0,1,nx)*2*np.pi)+2000
    
    # set store for those values to save
    u_all = np.zeros((nt//nsave, nx))
    h_all = np.zeros((nt//nsave, nx))
    u_all[0] = u0
    h_all[0] = h0
    
    # calcululate dt by dx
    dtbydx = (T/nt)/(diam/nx) 
    
    # initiate values
    h_minus = h0.copy()
    h = h0.copy()
    h_plus = h0.copy()
    
    u_minus = u0.copy()
    u = u0.copy()
    u_plus = u0.copy()
    
    for i in range(nt):
        # evolve foreward by timestep
        u_plus, h_plus = (
            evolve_u_ctcs(h, u, u_minus, dtbydx, g), 
            evolve_h_ctcs(h, u, h_minus, dtbydx)
        )
        # update the previous value
        u_minus = u
        h_minus = h
        # update the current value
        u = u_plus
        h = h_plus
        
        # save vaklues if on save run
        if (i%nsave)==0:
            u_all[i//nsave] = u
            h_all[i//nsave] = h
            
    return u_all, h_all


def run_ftbtcs(nx, nt, nsave, g=9.81, T=1e4):
    """
    Runs semi-implicit integration on the A-grid
    Ags:
        nx: number of x steps
        nt: number of time steps
        nsave: number of timestepps to save
    """
    
    # set initial conditions
    u0 = np.cos(np.linspace(0,1,nx)*2*np.pi)
    h0 = np.cos(np.linspace(0,1,nx)*2*np.pi)+2000
    
    # set store for those values to save
    u_all = np.zeros((nt//nsave, nx))
    h_all = np.zeros((nt//nsave, nx))
    u_all[0] = u0
    h_all[0] = h0
    
    # calcululate dt by dx
    dtbydx = (T/nt)/(diam/nx) 
    
    # initiate values
    h_minus = h0.copy()
    h = h0.copy()
    h_plus = h0.copy()
    
    u_minus = u0.copy()
    u = u0.copy()
    u_plus = u0.copy()
    
    for i in range(nt):
        # evolve foreward by timestep
        u_plus = evolve_u_ftcs(h, u, u_minus, dtbydx, g)
        h_plus = evolve_h_btcs(h, u_plus, dtbydx)
        
        # update the previous value
        u_minus = u
        h_minus = h
        # update the current value
        u = u_plus
        h = h_plus
        
        # save vaklues if on save run
        if (i%nsave)==0:
            u_all[i//nsave] = u
            h_all[i//nsave] = h
            
    return u_all, h_all
     

if __name__=="__main__":
    # set parameters
    T=5e6 # make this 2 powers above nt
    nt = int(1e5)
    nframes = 200
    dtbydx=0.1
    make_plot=True
    nx=100
    gifname = "ctcs-ftbtcs1"
    
    # derived quantities
    dtbydx = (T/nt)/(diam/nx)
    print("dt/dx:", dtbydx)
    print("c:", dtbydx*np.sqrt(2000*9.81))
    #nx = int(dtbydx*diam*nt/T)
    print('nx:', nx)
    nsave = nt//nframes
    print('save every:', nsave)
    
    # run simulation
    u_ctcs, h_ctcs = run_ctcs(nx, nt, nsave, g=9.81, T=T)
    u_ftbtcs, h_ftbtcs = run_ftbtcs(nx, nt, nsave, g=9.81, T=T)
    
    u_all = u_ctcs - u_ftbtcs 
    h_all = h_ctcs - h_ftbtcs
    print('finished run')
    
    if make_plot:
        # set up fig and axes
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for ax in [ax1, ax2]:
            ax.set_xlabel('x')
        ax1.set_ylabel('u')
        ax2.set_ylabel('h')

        minu, maxu = np.nanmin(u_all), np.nanmax(u_all)
        minh, maxh = np.nanmin(h_all), np.nanmax(h_all)
        #minu = minu if minu>-np.inf else -1
        #maxu = maxu if maxu<np.inf else 1
        #minh = minh if minh>-np.inf else 0
        #maxh = maxh if maxh<np.inf else 3
        ax1.set_ylim(minu, maxu)
        ax2.set_ylim(minh, maxh)

        # time and space info for axis
        t = np.arange(nframes)
        x = np.linspace(0, diam, nx)
        X = np.repeat(x[:, np.newaxis], nt//nsave, 1).T

        # create variable blocks and timeline
        u_block = amp.blocks.Line(X, u_all, ax=ax1)
        h_block = amp.blocks.Line(X, h_all, ax=ax2)
        timeline = amp.Timeline(t, units='s', fps=20)

        # Make gif
        anim = amp.Animation([u_block, h_block], timeline = timeline)
        plt.tight_layout()
        anim.controls()
        anim.save_gif(gifname) # save animation for docs
        plt.show()
        
    print('DONE')

