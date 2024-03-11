""" 
We want to make sure that the simulation does what it should 
even after our optimizations

We add a test case with the old implementation, and then 
ensures that the newer implementetation return the same 
value

We do this using the pytest framework

Note: We only need to test function getAcc, it should cover all the code that was changed
"""
import sys
from pathlib import Path
from scipy.special import gamma
import numpy as np
# Add current directory to path
# (so it can find the other files in this directory)
sys.path.append(str(Path(__file__).resolve().parents[0]))

import original as sph_old
import cpu as sph_new

def test_get_acc() -> None:
    """ Test the new getAcc to ensure it returns the 
          same result as the old getAcc function
    """
    N = 50

    # Simulation parameters
    M         = 2      # star mass
    r         = 0.75   # star radius
    h         = 0.1    # smoothing length
    k         = 0.1    # equation of state constant
    n         = 1      # polytropic index
    nu        = 1      # damping

    # Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed

    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/r**3/gamma(1+n))**(1/n) / r**2  # ~ 2.01
    m     = M/N                   # single particle mass
    pos   = np.random.randn(N,3)   # randomly selected positions and velocities
    vel   = np.zeros(pos.shape)

    # calculate initial gravitational accelerations
    acc_old = sph_old.getAcc( pos, vel, m, h, k, n, lmbda, nu )
    acc_new = sph_new.get_acc( pos, vel, m, h, k, n, lmbda, nu )

    # Ensure that the result is the same
    assert(len(acc_new) == len(acc_old))

    for i in range(0, len(acc_new)):
        for j in range(0, len(acc_new[i])):
            new_val = acc_new[i][j]
            old_val = acc_old[i][j]
            diff = abs(new_val - old_val)
            assert diff < 0.000000000000001

    print("Pass!")


if __name__=="__main__":
    test_get_acc()
