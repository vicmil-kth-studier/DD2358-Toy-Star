
# Add current directory to path(so it can find the other files in this directory)
import sys; from pathlib import Path; 
sys.path.append(str(Path(__file__).resolve().parents[0])) 

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_different_n(max_n : int, step : int, times : int = 1, show_cpu: bool = False, show_gpu: bool = False, show_original: bool = False, log : bool = False, save : bool = False) -> None:
    """ Plot performance of different optimizations
    
    Parameters
    ----------
        max_n (int): The maximum number of particles to use, the number of particles will show on the x-axis
        step (int): The number of simulation steps for each simulation
        times (int): The number of times to do the simulation for each set of arguments(Then take the avarage)
        show_cpu (bool): If true, run simulation on cpu. 
        show_gpu (bool): If true, run simulation on gpu. 
        show_original (bool): If true, run simulation without any optimizations. 
        log (bool): If true, do a log graph instead
        save (bool): If it should save to a png instead of display
        
    Returns
    -------
        None
    """
    assert step > 0
    
    functions = []
    names = []
    
    if show_cpu:
        import cpu
        functions.append(cpu.sim)
        names.append("CPU (NumPy)")
        
    if show_gpu:
        import gpu
        functions.append(gpu.sim)
        names.append("GPU (CuPy)")
    
    if show_original:
        import original
        functions.append(original.sim)
        names.append("Original (NumPy)")

    # Plot time on y axis
    # Plot n on x axis
    sizes = list(range(step,max_n,step))
    time_end = 1
    
    plots = []
    
    for fun in functions:
        avg_times = []
        for n in sizes:
            print(n)
            result = []
            for i in range(times):
                start = timer()
                fun(plot=False,plot_2d=False,plot_real_time=False,n=n,time_end=time_end)
                end = timer()
                result.append(end-start)
            avg_times.append(np.average(result))
        plots.append(avg_times)

    for i in range(len(names)):
        plt.plot(sizes, plots[i], label = names[i])
        plt.legend()
    
    if log:
        plt.yscale('log')
    
    plt.xlabel('particle count')
    plt.ylabel('time [s]')
    if save:
        plt.savefig(f"""images\\{"c" if show_cpu else ""}{"g" if show_gpu else ""}{"o" if show_original else ""}{"_log" if log else ""}_n{max_n}_s{step}_t{times}.png""")
    else:
        plt.show()


'''def main():
	""" SPH simulation """
	
	# Simulation parameters
	N         = 400    # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 12     # time at which simulation ends
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
	acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
	
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
		acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
		
		# (1/2) kick
		vel += acc * dt/2
		
		# update time
		t += dt
		
		# get density for plotting
		rho = getDensity( pos, pos, m, h )
		
		# plot in real time
		if plotRealTime or (i == Nt-1):
			plt.sca(ax1)'''
               

if __name__ == "__main__":
    args_dict = dict()
    args_dict["n"] = 100
    args_dict["times"] = 1
    args_dict["step"] = 1
    args_dict["cpu"] = "True"
    args_dict["gpu"] = "True"
    args_dict["original"] = "True"
    args_dict["plot"] = "no"
    args_dict["log"] = "False"
    args_dict["save"] = "False"
    
    for i in range(1, len(sys.argv)): # The first arg is the name of the script
        arg = sys.argv[i]
        split = arg.split("=")
        assert(split[0] in args_dict.keys())
        args_dict[split[0]] = split[1]

    if args_dict["plot"] == "no":
        plot_different_n(
            max_n=int(args_dict["n"]),
            step=int(args_dict["step"]),
            times=int(args_dict["times"]),
            show_cpu=args_dict["cpu"] == "True",
            show_gpu=args_dict["gpu"] == "True",
            show_original=args_dict["original"] == "True", 
            log=args_dict["log"] == "True",
            save=args_dict["save"] == "True")
    elif args_dict["plot"] == "gpu":
        import gpu
        gpu.sim(
            plot=True,
            plot_2d=True,
            plot_real_time=True,
            n=int(args_dict["n"]))
    elif args_dict["plot"] == "cpu":
        import cpu
        cpu.sim(
            plot=True,
            plot_2d=True,
            plot_real_time=True,
            n=int(args_dict["n"]))
    elif args_dict["plot"] == "original":
        import original
        original.sim(
            plot=True,
            plot_2d=True,
            plot_real_time=True,
            n=int(args_dict["n"]))