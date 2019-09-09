import numpy as np
import matplotlib.pyplot as plt

# constants
r = 6371e3 # radius of earth in meters
diam = 2*np.pi*r


def evolve_u_ct(h, u_minus, dtbydx, g):
    """
    Args:
        h: h at current time
        u_minus : u at previous time step
    returns:
        u at next time step,
    """
    u_plus = -g*dtbydx * (np.roll(h,-1) - np.roll(h, 1)) + u_minus
    return u_plus

def evolve_h_ct(u, h_minus, dtbydx, H):
    h_plus = -H*dtbydx * (np.roll(u,-1) - np.roll(u, 1)) + h_minus
    return h_plus


def run(nx, nt, nsave, H=1000, g=9.81, T=1e4):
    """
    Ags:
        nx: number of x steps
        nt: number of time steps
        nsave: number of timestepps to save
    """
    
    # set initial conditions
    u0 = np.sin(np.linspace(0,1,nx)*2*np.pi)
    h0 = 100*np.cos(np.linspace(0,1,nx)*2*np.pi)
    
    # set store for those values to save
    u_all = np.zeros((nt//nsave, nx))
    h_all = np.zeros((nt//nsave, nx))
    u_all[0] = u0
    h_all[0] = h0
    
    # calcululate dt by dx
    dtbydx = (T/nt)/(diam/nx) 
    print(dtbydx)
    
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
            evolve_u_ct(h, u_minus, dtbydx, g), 
            evolve_h_ct(u, h_minus, dtbydx, H)
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
        

if __name__=="__main__":
    nx = 1000
    nt = 1000
    nsave=10
    u_all, h_all = run(nx, nt, nsave, g=9.81, H=1000, T=1e4)
    
    import numpy as np
    import matplotlib.pyplot as plt
    import animatplot as amp
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for ax in [ax1, ax2]:
        #ax.set_aspect('equal')
        ax.set_xlabel('x')

    ax1.set_ylabel('u')
    ax2.set_ylabel('h')
    
    t = np.arange(0, nt, nsave)
    x = np.linspace(0, diam, nx)
    X = np.repeat(x[:, np.newaxis], nt//nsave, 1).T

    u_block = amp.blocks.Line(X, u_all, ax=ax1)
    h_block = amp.blocks.Line(X, h_all, ax=ax2)
    
    timeline = amp.Timeline(t, units='s', fps=20)
    anim = amp.Animation([u_block, h_block], timeline = timeline)
    plt.tight_layout()
    anim.controls()
    anim.save_gif('line1') # save animation for docs
    plt.show()
    print('DONE')
    
