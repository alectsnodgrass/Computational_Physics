# Introduction

Numerical methods can be used to make solving ordinary differential equations much more easy and efficient. This (tool) is used to demonstrate and explore several techniques... We specifically explore the ... methods. We plot the phase space and... Energy vs Time... then we analyze the plots to determine which methods conserve energy conservation... which are more accurate/precise. Then we analyze the efficiency of the methods. WE find that... (conclusions) How many time steps to get a good enough approximation

Numerical methods can be implemented to approximate solutions for ordinary differential equations that would otherwise be a computationally labor-some task. This package was developed to demonstrate the capabilities of several approximation techniques which include Euler's method, the Runge-Kutta technique, Verlet integration, and Scipy's ODEINT. We provide phase-space and Energy vs Time plots to illustrate their approximation capabilities and ability to conserve energy. By the same token, we plot and analyze the relative error of each method. We found that the Verlet method reaches a 5% relative error within 64 time steps, whereas RK2, RK4, and ODEINT reach the target error within 128 time steps, and Euler's method requires about 2048 time steps. The pros and cons of each method depending on the situation are further discussed in the report.
## Background Theory

How does Euler's Method work?... 
[insert snippet of Euler_solver]

How does RK method work? RK2... RK4...
[Insert snippet of RK2_solver]

How does Verlet integration work? Symplectic (Verlet)...
[Insert snippet of verlet_solver]

How does ODE int work? Scipy's ODE integrator...




# Procedure

What ODE's did we test these on? Exponential decay... SHO with and without a linear dampening...

What did we look at to analyze the methods?

Phase space...

Energy (Hamiltonian)... (is energy conserved basically). State the relationship between energy conservation and phase space area.

Relative error vs time... (maybe also pick some reasonable tmax and compare the error at tmax)




## Instructions
To run the tool? use (need to change the instructions to compile now that I'm using a header file.)

`
conda activate [environment]
python P3_Code.py
`cmd



# Analysis

Import Figures and talk about precision, accuracy, (error and efficiency). Why does Euler's method drift? Error accumulates.
 [plot of Euler's method vs analytic] caption: the error accumulates with more time steps 

... RK method is a step above... higher order RK method is better but more computer work required... 
[RK2 vs RK4 plot] caption: notice that RK2 and RK4 are not very different for nts range.  

... Scipy's odeINT. Talk about ease of use. seems to yield the least relative error.

Verlet method: The special thing about the verlet method is that it conserves energy over time as shown in fig.
[insert figure of energy vs time for verlet method]

# Conclusions
This is where I want to talk about in what cases might one want to use which method like in terms of computational cost and overall efficiency.

# Extensions

# Questions

## Timekeeping

Week before spring break: 
1 hour: Tuesday in class

1 hour: Tuesday after class or Wednesday (I forgot which day)

1? hour: Thursday

1 hour: Friday

After spring break: 
3 hours: Tuesday 3/24
