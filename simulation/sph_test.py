import sph_orig as sph_old
import sph_cpu_v2 as sph_new
import numpy as np
from scipy.special import gamma


"""
We want to make sure that the simulation does what it should 
even after our optimizations

We add a test case with the old implementation, and then 
ensures that the newer implementetation return the same 
value

We do this using the pytest framework

Note: We only need to test function getAcc, it should cover all the code that was changed
"""


def test_getAcc():
    N = 50

    # Simulation parameters
    t         = 0      # current time of the simulation
    #tEnd      = 12     # time at which simulation ends
    dt        = 0.04   # timestep
    M         = 2      # star mass
    R         = 0.75   # star radius
    h         = 0.1    # smoothing length
    k         = 0.1    # equation of state constant
    n         = 1      # polytropic index
    nu        = 1      # damping
    plotRealTime = True # switch on for plotting as the simulation goes along
    
    # Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed
    
    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
    m     = M/N                    # single particle mass
    pos   = np.random.randn(N,3)   # randomly selected positions and velocities
    vel   = np.zeros(pos.shape)
    
    # calculate initial gravitational accelerations
    acc_old = sph_old.getAcc( pos, vel, m, h, k, n, lmbda, nu )
    acc_new = sph_new.getAcc( pos, vel, m, h, k, n, lmbda, nu )

    # Ensure that the result is the same
    assert(len(acc_new) == len(acc_old))

    for i in range(0, len(acc_new)):
        for j in range(0, len(acc_new[i])):
            new_val = acc_new[i][j]
            old_val = acc_old[i][j]
            diff = abs(new_val - old_val)
            assert(diff < 0.000000000000001)

    print("Pass!")


if __name__=="__main__":
    test_getAcc()