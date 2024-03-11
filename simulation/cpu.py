"""
CPU Driven simulation of a toy star using CUDA
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from typing import Any

sqrt_pi = np.sqrt(np.pi)


def r_exp(x: Any, y: Any, z: Any, h: float):
    """ Cached function for a common operation

    Parameters
    ----------
    x (Any):     a vector/matrix of x positions
    y (Any):     a vector/matrix of y positions
    z (Any):     a vector/matrix of z positions
    h (float):   the smoothing length

    Returns
    -------
    (float): exp(-(x^2 + y^2 + z^2) / (h^2))
    """
    return np.exp(-(x*x + y*y + z*z) / (h*h))


def w(h: float, rexp: Any) -> Any:
    """Gausssian Smoothing kernel (3D)
    (w is the evaluated smoothing function)

    Parameters
    ----------
    h (float):     the smoothing length
    rexp (Any):    is the precomputed exp(-(x^2 + y^2 + z^2) / (h^2))

    Returns:
    Any
    """
    
    tmp = h*sqrt_pi    
    return rexp / (tmp*tmp*tmp)
    

def grad_w(x: Any, y: Any, z: Any, h: float, rexp: Any):
    """Gradient of the Gausssian Smoothing kernel (3D)

    Parameters
    ----------
    x (Any):      a vector/matrix of x positions
    y (Any):      a vector/matrix of y positions
    z (Any):      a vector/matrix of z positions
    h (float):    the smoothing length
    rexp (Any):   is the precomputed exp(-(x^2 + y^2 + z^2) / (h^2))

    Returns
    -------
    wx, wy, wz:    the evaluated gradient
    """
    
    n = rexp / (-0.5*h**5 * (np.pi)**(3/2))
    wx = n * x
    wy = n * y
    wz = n * z
    
    return wx, wy, wz
    
def get_pairwise_separations(ri, rj):
    """ Get pairwise desprations between 2 sets of coordinates

    Parameters
    ----------
    ri (Any):    is an M x 3 matrix of positions
    rj (Any):    is an N x 3 matrix of positions

    Returns
    -------
    dx, dy, dz:   are M x N matrices of separations
    """
    
    m = ri.shape[0]
    n = rj.shape[0]
    
    # positions ri = (x,y,z)
    rix = ri[:,0].reshape((m, 1))
    riy = ri[:,1].reshape((m, 1))
    riz = ri[:,2].reshape((m, 1))
    
    # other set of points positions rj = (x,y,z)
    rjx = rj[:,0].reshape((n, 1))
    rjy = rj[:,1].reshape((n, 1))
    rjz = rj[:,2].reshape((n, 1))
    
    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
    
    return dx, dy, dz
    
def get_density(r: Any, mass: Any, h: float, rexp: Any):
    """ Get Density at sampling loctions from SPH particle distribution

    Parameters
    ----------
    r (Any):     is an M x 3 matrix of sampling locations
    mass (Any):  is the particle mass
    h (float):   is the smoothing length
    rexp (Any):  is the precomputed exp(-(x^2 + y^2 + z^2) / (h^2))

    Returns
    -------
    rho   is M x 1 vector of densities
    """
    
    m = r.shape[0]
        
    rho = np.sum(mass * w(h, rexp), 1).reshape((m, 1))
    
    return rho
    
def get_pressure(rho: Any, k: Any, n: Any):
    """ Equation of State

    Parameters
    ----------
    rho (Any):     vector of densities
    k (Any):       equation of state constant
    n (Any):       polytropic index

    Returns
    -------
    p (Any):       pressure
    """
    
    p = k * rho**(1+1/n)
    
    return p
    
def get_acc(pos: Any, vel: Any, mass: Any, h: float, k: Any, poly_index: Any, lmbda: Any, nu: Any):
    """ Calculate the acceleration on each SPH particle

    Parameters
    ----------
    pos (Any):            is an N x 3 matrix of positions
    vel (Any):            is an N x 3 matrix of velocities
    mass (Any):           is the particle mass
    h (float):            is the smoothing length
    k (Any):              equation of state constant
    poly_index (Any):     polytropic index
    lmbda external (Any): force constant
    nu (Any):             viscosity

    Returns
    -------
    a (Any):              is N x 3 matrix of accelerations
    """
    
    n = pos.shape[0]
    
    # Calculate densities at the position of the particles
    dx, dy, dz = get_pairwise_separations(pos, pos)
    rexp = r_exp(dx, dy, dz, h) # reuse
    rho = get_density(pos, mass, h, rexp)
    
    # Get the pressures
    p = get_pressure(rho, k, poly_index)
    
    # Get pairwise distances and gradients
    dwx, dwy, dwz = grad_w(dx, dy, dz, h, rexp)
    
    # Add Pressure contribution to accelerations
    tmp = mass * (p/rho**2 + p.T/rho.T**2)               
    ax = - np.sum(tmp * dwx, 1).reshape((n, 1))          
    ay = - np.sum(tmp * dwy, 1).reshape((n, 1))          
    az = - np.sum(tmp * dwz, 1).reshape((n, 1))          
    
    # pack together the acceleration components
    a = np.hstack((ax,ay,az))
    
    # Add external potential force
    a -= lmbda * pos
    
    # Add viscosity
    a -= nu * vel
    
    return a

def sim(plot: bool, n: int = 5000, time_end: float = 100, plot_2d : bool = False, plot_real_time : bool = True) -> None:
    """ SPH simulation 
    
    Parameters
    ----------
    plot: (bool):          Plot a simulation of a star if true
    n (int):               Number of particles
    time_end (float):      How long the simulation should run
    plot_2d (bool):        If true, plot 2d. If false, plot 3d
    plot_real_time (bool): Plot a simulation of a star in real time if true, if False, just show the last time step

    Returns
    None
    -------
    """
    
    # Simulation parameters
    t         = 0      # current time of the simulation
    dt        = 0.04   # timestep
    mass         = 2      # star mass
    radius         = 0.75   # star radius
    h         = 0.1    # smoothing length
    k         = 0.1    # equation of state constant
    poly_index         = 1      # polytropic index
    nu        = 1      # damping
    
    # Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed
    
    lmbda = 2*k*(1+poly_index)*np.pi**(-3/(2*poly_index)) * (mass*gamma(5/2+poly_index)/radius**3/gamma(1+poly_index))**(1/poly_index) / radius**2  # ~ 2.01
    m     = mass/n                    # single particle mass
    pos   = np.random.randn(n,3)   # randomly selected positions and velocities
    vel   = np.zeros(pos.shape)
    
    # calculate initial gravitational accelerations
    acc = get_acc(pos, vel, m, h, k, poly_index, lmbda, nu)
    
    # number of timesteps
    timesteps = int(np.ceil(time_end/dt))
    
    # prep figure
    if plot:
        fig = plt.figure(figsize=(4,5), dpi=80)
        grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
        ax3d = fig.add_subplot(projection='3d')
        if plot_2d:
            ax2d = plt.subplot(grid[2,0])
            rr = np.zeros((100, 3))
            rlin = np.linspace(0, 1, 100)
            rr[:,0] = rlin
            rho_analytic = lmbda/(4*k) * (radius**2 - rlin**2)

    # Simulation Main Loop
    for i in range(timesteps):
        # (1/2) kick
        vel += acc * (dt/2)
        
        # drift
        pos += vel * dt
        
        # update accelerations
        acc = get_acc(pos, vel, m, h, k, poly_index, lmbda, nu)
        
        # (1/2) kick
        vel += acc * (dt/2)
        
        # update time
        t += dt
        
        if plot:
            # get density for plotting
            dx, dy, dz = get_pairwise_separations(pos, pos)
            rexp = r_exp(dx, dy, dz, h)
            rho = get_density(pos, m, h, rexp)
        
        # plot in real time
        if plot and (plot_real_time or (i == timesteps-1)):
            plt.sca(ax3d)
            plt.cla()
            
            # 3d view
            cval = np.minimum((rho-3)/3,1).flatten()

            xs = pos[:,0]
            ys = pos[:,1]
            zs = pos[:,2]
            ax3d.scatter(xs, ys, zs, s=10, c=cval, cmap=plt.cm.autumn, alpha=0.5)
            
            ax3d.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2))
            ax3d.set_aspect('equal', 'box')
            ax3d.set_xticks([-1, 0, 1])
            ax3d.set_yticks([-1, 0, 1])
            ax3d.set_zticks([-1, 0, 1])
            
			# bottom view
            if plot_2d:
                plt.sca(ax2d)
                plt.cla()
                ax2d.set(xlim=(0, 1), ylim=(0, 3))
                ax2d.set_aspect(0.1)
                plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
                dx, dy, dz = get_pairwise_separations(rr, pos)
                rexp = r_exp(dx, dy, dz, h)
                rho_radial = get_density(rr, m, h, rexp)

                plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)
        
    if plot:
        # add labels/legend
        plt.sca(ax2d)
        plt.xlabel('radius')
        plt.ylabel('density')
        
        # Save figure
        plt.savefig('sph.png',dpi=240)
        plt.show()