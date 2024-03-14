# Toy Star
## Description
This is a project for course DD2358 High performance computing.

This purpose of the project was to take initial code from 
https://github.com/pmocz/sph-python

And then to optimize the code to run faster.

## Usage and Dependencies
### Python
You need python to run this project.

Some packages may need to be downloaded using pip, so once you have python you 
can try to run the code and download missing packages using:

pip install <packagename>


### Cuda
You can run the project without Cuda. But to fully utilize the project and run it on the GPU you need to have cuda installed. This requires that you have NVIDIA GPU.

### Args
To generate graphs you can specify an amount of args, like [ARG]=[VALUE]
* "n" is the max amount of particles
* "times" is the amount of times to run the same simulation
* "step" is the range(step,n,step) between runs on the graph
* "cpu" True/False, turns on and off the CPU in the graph
* "gpu" True/False, turns on and off the GPU in the graph
* "original" True/False, turnes on and off the original in the graph 
* "plot" no/gpu/cpu/original, shows the simulation of the given arg or the preformence graph
* "log" True/False converts the linear graph to a logarithmic gragh
* "save" True/False saves the content in the images folder instead of showing it to the user

#### Examples
##### Runs the simulation on the GPU with 1000 particles
python simulation n=1000 plot=gpu

##### Plots performance of gpu, cpu and original with a step of 100 and a maximum size of 1000 particles on a logarithmic graph
python simulation n=1000 step=100 times=2 log=True

## Results
When you run the code a window should pop up, showing a simulation of a star

## Documentation
Feel free to check out the python documentation at:
[Sphinx documentation](https://vicmil-kth-studier.github.io/DD2358-Toy-Star/simulation/docs/build/html/modules.html)
