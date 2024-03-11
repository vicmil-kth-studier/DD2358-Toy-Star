"""
CPU Driven simulation of a toy star
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

sqrt_pi = np.sqrt(np.pi)


def r_exp(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        h: float) -> np.ndarray:
    """ Cached function for a common operation

    Parameters
    ----------
    x (np.ndarray):     a vector/matrix of x positions
    y (np.ndarray):     a vector/matrix of y positions
    z (np.ndarray):     a vector/matrix of z positions
    h (float):          the smoothing length

    Returns
    -------
    (np.ndarray): exp(-(x^2 + y^2 + z^2) / (h^2))
    """
    return np.exp(-(x*x + y*y + z*z) / (h*h))


def w(h: float, rexp: np.ndarray) -> np.ndarray:
    """Gausssian Smoothing kernel (3D)
    (w is the evaluated smoothing function)

    Parameters
    ----------
    h (float):          the smoothing length
    rexp (np.ndarray):  is the precomputed exp(-(x^2 + y^2 + z^2) / (h^2))

    Returns:
    np.ndarray
    """

    tmp = h*sqrt_pi
    return rexp / (tmp*tmp*tmp)


def grad_w(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        h: float,
        rexp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient of the Gausssian Smoothing kernel (3D)

    Parameters
    ----------
    x (np.ndarray):      a vector/matrix of x positions
    y (np.ndarray):      a vector/matrix of y positions
    z (np.ndarray):      a vector/matrix of z positions
    h (float):           the smoothing length
    rexp (np.ndarray):   is the precomputed exp(-(x^2 + y^2 + z^2) / (h^2))

    Returns
    -------
    wx, wy, wz (np.ndarray, np.ndarray, np.ndarray): the evaluated gradient
    """

    n = rexp / (-0.5*h**5 * (np.pi)**(3/2))
    wx = n * x
    wy = n * y
    wz = n * z

    return wx, wy, wz


def get_pairwise_separations(
        ri: np.ndarray,
        rj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get pairwise desprations between 2 sets of coordinates

    Parameters
    ----------
    ri (np.ndarray):    is an M x 3 matrix of positions
    rj (np.ndarray):    is an N x 3 matrix of positions

    Returns
    -------
    dx, dy, dz (np.ndarray, np.ndarray, np.ndarray):   are M x N matrices of separations
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


def get_density(
        r: np.ndarray,
        mass: float,
        h: float,
        rexp: np.ndarray) -> np.ndarray:
    """ Get Density at sampling loctions from SPH particle distribution

    Parameters
    ----------
    r (np.ndarray):     is an M x 3 matrix of sampling locations
    mass (float):       is the particle mass
    h (float):          is the smoothing length
    rexp (np.ndarray):  is the precomputed exp(-(x^2 + y^2 + z^2) / (h^2))

    Returns
    -------
    rho (np.ndarray):   is M x 1 vector of densities
    """

    m = r.shape[0]

    rho = np.sum(mass * w(h, rexp), 1).reshape((m, 1))

    return rho


def get_pressure(rho: np.ndarray, k: float, n: int) -> np.ndarray:
    """ Equation of State

    Parameters
    ----------
    rho (np.ndarray):   vector of densities
    k (float):          equation of state constant
    n (int):            polytropic index

    Returns
    -------
    p (np.ndarray):     pressure
    """

    p = k * rho**(1+1/n)

    return p


def get_acc(
        pos: np.ndarray,
        vel: np.ndarray,
        mass: float,
        h: float,
        k: float,
        poly_index: int,
        lmbda: float,
        nu: float) -> np.ndarray:
    """ Calculate the acceleration on each SPH particle

    Parameters
    ----------
    pos (np.ndarray):       is an N x 3 matrix of positions
    vel (np.ndarray):       is an N x 3 matrix of velocities
    mass (float):           is the particle mass
    h (float):              is the smoothing length
    k (float):              equation of state constant
    poly_index (float):     polytropic index
    lmbda external (float): force constant
    nu (float):             viscosity

    Returns
    -------
    a (np.ndarray):         is N x 3 matrix of accelerations
    """

    n: int = pos.shape[0]

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


def sim(
        plot: bool,
        n: int = 5000,
        time_end: float = 100,
        plot_2d: bool = False,
        plot_real_time: bool = True) -> None:
    """ SPH simulation 

    Parameters
    ----------
    plot: (bool):          Plot a simulation of a star if true
    n (int):               Number of particles
    time_end (float):      How long the simulation should run
    plot_2d (bool):        If true, plot 2d. If false, plot 3d
    plot_real_time (bool): Plot a simulation of a star in real time if true, 
                            if False, just show the last time step

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

    lmbda = 2*k*(1+poly_index)*np.pi**(-3/(2*poly_index)) * (
        mass*gamma(5/2+poly_index)/radius**3/gamma(1+poly_index)
        )**(1/poly_index) / radius**2  # ~ 2.01
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


def main() -> None:
    """ SPH simulation """

    # Simulation parameters
    N: int           = 400    # Number of particles
    t: float         = 0      # current time of the simulation
    tEnd: float      = 12     # time at which simulation ends
    dt: float        = 0.04   # timestep
    M: float         = 2      # star mass
    R: float         = 0.75   # star radius
    h: float         = 0.1    # smoothing length
    k: float         = 0.1    # equation of state constant
    n: int           = 1      # polytropic index
    nu: float        = 1      # damping
    plotRealTime: bool = True # switch on for plotting as the simulation goes along

	# Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed

    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
    m     = M/N                    # single particle mass
    pos   = np.random.randn(N,3)   # randomly selected positions and velocities
    vel   = np.zeros(pos.shape)

	# calculate initial gravitational accelerations
    acc = get_acc( pos, vel, m, h, k, n, lmbda, nu )

	# number of timesteps
    Nt = int(np.ceil(tEnd/dt))

	# prep figure
    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])
    rr = np.zeros((100,3))
    rlin = np.linspace(0,1,100)
    rr[:,0] =rlin
    rho_analytic = lmbda/(4*k) * (R**2 - rlin**2)

	# Simulation Main Loop
    for i in range(Nt):
		# (1/2) kick
        vel += acc * dt/2

		# drift
        pos += vel * dt

		# update accelerations
        acc = get_acc( pos, vel, m, h, k, n, lmbda, nu )

		# (1/2) kick
        vel += acc * dt/2

		# update time
        t += dt

		# get density for plotting
        dx, dy, dz = get_pairwise_separations(pos, pos)
        rexp = r_exp(dx, dy, dz, h)
        rho = get_density(pos, m, h, rexp)

		# plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho-3)/3,1).flatten()
            plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
            ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-1,0,1])
            ax1.set_yticks([-1,0,1])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))

            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0, 1), ylim=(0, 3))
            ax2.set_aspect(0.1)
            plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
            dx, dy, dz = get_pairwise_separations(rr, pos)
            rexp = r_exp(dx, dy, dz, h)
            rho_radial = get_density(rr, m, h, rexp)
            plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)

	# add labels/legend
    plt.sca(ax2)
    plt.xlabel('radius')
    plt.ylabel('density')

    # Save figure
    plt.savefig('sph.png',dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
