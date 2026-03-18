---
meta:
    author: Alec Snodgrass
    topic:  ODE Project
    course: TN Tech PHYS 4130
    term:   Spring 2026
---
# ODE Solver Project Writeup

## Introduction
The physical world is described by differential equations. Examples include laws of motion (Newton's Laws), electrodynamics (including electric/magnetic circuits), heat transfer, wave motion, etc. Dynamical systems are described by these mathematical equations. These equations are rarely analytically tractable. In most cases, there are non-linear forces, damping, or coupled systems, which make numerical approximations the most feasible solution method. 

Computational physics uses numerical methods to approximate, to varying degrees of accuracy, the behavior of these physical systems. By discretizing time or space, algorithms are developed to step through the evolution of these systems. This is done by approximating the continuous derivative with finite difference approximations. The order of approximation and correction terms determines the accuracy and effectiveness of the solver. 

There are many different algorithms for solving ODEs. There is an inverse relationship between the simplicity of the algorithm and the accuracy of the solution. For example, the Euler method is easy to implement but is not very accurate and can diverge for certain systems. In contrast, higher-order methods like Runge-Kutta are much more accurate but require more computational time and effort. A symplectic integrator like the Verlet method has the advantage of conserving properties of conservative systems. Parameters of the problem at hand, such as time scale, stiffness, and required accuracy, factor into which solver is most appropriate.

In this project, multiple numeric differential equation solvers were implemented on a simple harmonic oscillator (SHO) system (undamped and damped). This is a simple system whose behavior is well known, making it easy to compare the numerical results to the expected results. The different algorithms were compared to evaluate their performance and accuracy. Each solver's performance can be seen in the plots of position, velocity, phase space, and energy evolution. The equation of motion for the SHO is as follows:
```math
\ddot{x} + \gamma \dot{x} + \omega^2 x = 0
```
Where the gamma, the damping coefficient, is zero for the undamped case. This equation is analytically solvable, physically ubiquitous, and is sensitive to errors in approximation. 

##  Solver Algorithms
In this section, the underlying equations that motivate the algorithms are examined and related to their corresponding implementations. Three different solvers are explored: a symplectic integrator and two different predictor-corrector methods. The explanation does not include a derivation, but it does explain how each of these solvers works in terms of the algorithm and the underlying equations. 

### Symplectic Integrator: Verlet
Verlet integration is a numerical method most commonly used to solve Newton's equations of motion for different systems. It works well for most mechanical systems, which can be described by a second-order ODE such as 
```math
\ddot{x} = f(x, \dot{x}, t)
```
The algorithm updates the position first by averaging the Taylor expansion of the position at the two neighboring time intervals. 
```math
\mathrm{Taylor\; Expand:}\; x(t + \Delta t) \\
\mathrm{Taylor\; Expand:}\; x(t - \Delta t) \\
\mathrm{Add\; them:}\; x(t + \Delta t) = 2x(t) - x(t - \Delta t) + \ddot{x}(t) \Delta t^2 + O(\Delta t^4)
```
Then, the velocity is updated using the average of the acceleration at the current and next time step.  
```math
\dot{x}(t + \Delta t) = \dot{x}(t) + \frac{\ddot{x}(t + \Delta t) - \ddot{x}(t - \Delta t)}{2} \Delta t
```

The code below shows the implementation of the equations above. 
```python
for it in range(1, nts): 
    x[it+1] = 2*x[it] - x[it-1] + (a[it] * dt**2)
    a[it+1] = deriv([x[it+1], v[it]], t[it+1])[1]       # Use the derivative function to get the acceleration
    v[it+1] = v[it] + (dt / 2)*(a[it] + a[it+1])
```
A careful observer will notice that the loop does not start at zero. This is because the algorithm requires the position at the previous time step. Therefore, the first iteration is done manually, given the initial conditions, in a manner similar to the Euler method.
```python
def Verlet_solver(coord_init, tmin, tmax, nts, deriv):
# coord_init is a list of the initial position and velocity, [x0, v0]
...
    x[0] = coord_init[0]
    v[0] = coord_init[1]
    a[0] = deriv(coord_init, tmin)[1] 

    x[1] = x[0] + (v[0]*dt) + (0.5*a[0] * dt**2)       
    a[1] = deriv([x[1], v[0]], t[1])[1]
    v[1] = v[0] + (0.5 * (a[0] + a[1]) * dt)
...
```

### SciPy Integrator: Runge-Kutta 4(5)
Runge-Kutta (RK) is a famous family of methods used to solve ODEs. The 4(5) indicates that it is a fourth-order method with a fifth-order error estimate. This is a prime example of a predictor-corrector method. The algorithm works by calculating slopes at different points within a time step and then taking the weighted average of those slopes to update the function values. What makes RK45 unique is that it is an *adaptive* method, meaning that the algorithm adjusts the time step size based on the estimated error. The complexities of the implementation of this method are overlooked for the purposes of this report, but the underlying equations are as follows:

Intermediate slopes:
```math
k_i = h * f(t_n + c_i h,\; y_n + \sum_{j=1}^{i-1} a_{ij} k_j)
```
Where h is the adjustable time step, and where the coefficients c_i and a_ij are determined by the specific RK method. For RK45, the coefficients are as follows:
```math 
\begin{aligned}
k_1 &= hf(t_n, y_n) \\
k_2 &= hf(t_n + \frac{1}{4}h, y_n + \frac{1}{4}k_1) \\
k_3 &= hf(t_n + \frac{3}{8}h, y_n + \frac{3}{32}k_1 + \frac{9}{32}k_2) \\
k_4 &= hf(t_n + \frac{12}{13}h, y_n + \frac{1932}{2197}k_1 - \frac{7200}{2197}k_2 + \frac{7296}{2197}k_3) \\
k_5 &= hf(t_n + h, y_n + \frac{439}{216}k_1 - 8k_2 + \frac{3680}{513}k_3 - \frac{845}{4104}k_4) \\
k_6 &= hf(t_n + \frac{1}{2}h, y_n - \frac{8}{27}k_1 + 2k_2 - \frac{3544}{2565}k_3 + \frac{1859}{4104}k_4 - \frac{11}{40}k_5) \\
\end{aligned}
```

Fourth-order approximation:
```math
y_{n+1} = y_n + \frac{25}{216} k_1 + \frac{1408}{2565} k_3 + \frac{2197}{4104} k_4 - \frac{1}{5} k_5 \\
```
Fifth-order approximation:
```math
z_{n+1} = y_n + \frac{16}{135} k_1 + \frac{6656}{12825} k_3 + \frac{28561}{56430} k_4 - \frac{9}{50} k_5 + \frac{2}{55} k_6
```
Update and error estimate:
```math
\mathrm{Error\; Estimate}\; = |z_{n+1} - y_{n+1}|
``` 
The RK4(5) solver was not manually implemented in the code; instead, SciPy's optimized implementation was used. There are some subtleties to passing parameters to and extracting data from the pre-made function, all of which are explained in the code. Otherwise, the solver is easy to use: just call the **solve_ivp** function with the appropriate arguments from the SciPy library.
```python
def RK45_solver(coord_init, tmin, tmax, nts, deriv):
...
    solution = solve_ivp(swapped_deriv, (tmin, tmax), coord_init, t_eval=t, method='RK45')
    # Where 'swapped_deriv' is a modified function for the differential equation that is compatible with solve_ivp.
    # And where 'solution' is an object that contains the time points and corresponding solutions. 
...
```

### SciPy Integrator: DOP853
The naming convention for this method is annoyingly cryptic, but it stands for **Do**rmand-**P**rince **8**(**5**, **3**): DOP853. Following the Runge-Kutta convention, this means that the DOP853 method is an eighth-order method with a fifth-order and third-order error estimate. This method is a member of the same family as the RK4(5) method, only more computationally intensive and more accurate. The SciPy library explains it as an eighth-order Runge-Kutta method originally from the *Fortran* library; it is particularly good for solving non-stiff ODEs with high precision. 

All the development for RK4(5) applies directly to the DOP853 method -- just more. DOP853 follows the same logic: calculate intermediate slopes, take a weighted average of slopes, update the function values, then adjust the time step using the error estimate. It is also an adaptive method, meaning the time step is adjusted based on an estimated error, which in this case is calculated using the low-order (fifth and third) approximations. Another interesting feature of this method addressed an issue caused by the adjustable time stepping. The feature is referred to as a "dense output formula," and it allows the end user to find the solution at any arbitrary point in time (even those in between the exact time steps). A seventh-order interpolation is used to construct a polynomial that can be evaluated for any time point. A very pleasant feature. 

The implementation, once again, was not done manually. A simple call to the **solve_ivp** function with the appropriate arguments and method name was all that was needed - except for some similar subtleties similar to the RK4(5) method.
```python
def DOP853_solver(coord_init, tmin, tmax, nts, deriv):
...
    sol = solve_ivp(swapped_deriv, (tmin, tmax), coord_init, t_eval=t, method='DOP853')
...
```

## Phase Space Trajectory and Energy Evolution of a Simple Harmonic Oscillator
Insert images of the phase space plots for each of the solvers. These pictures will help demonstrate their capability and a use case. 
### Phase Space Undamped SHM
### Phase Space Damped SHM
### Energy Evolution Undamped SHM
### Energy Evolution Damped SHM

## Strengths and Weaknesses of Each Solver
This section is dedicated to comparing the different solvers. I will lower the number of points each one gets until there is a difference. I will time the functions to find a difference. The two from scipy are very good, so it will take a lot to find where their limits are. 

Magic Commands: %timeit

Plots of the phase/energy for different solvers with LOW nts until there is a difference

Take the nts down and plot the same function for the different nts to compare it

There will be images of the plots that - hopefully- show the difference between the solvers. If it is not in the code yet, it will be simple to add. I will only need to write the plotting code and generate a png to include in the writeup.

## Extensions

##  Addressing Questions
### Attribution
I primarily used Wikipedia and SciPy documentation to become familiar with the algorithms and how to implement them. 

### Timekeeping
I spent several full days of work on this project. I started by comparing the Euler method, RK2, and the ODEINT solvers using the SHO system. I then use the format of those comparisons to implement and compare the Verlet, RK4(5), and DOP853 methods. I spent the first couple of days writing the code for the initial assignment and quickly setting up a rough draft of the outline. Then I spent a full day's worth of time looking into and explaining the algorithms for the three more advanced solvers. I spent the next couple of days implementing the extensions and explaining that work in my report write-up. 

### Languages, Libraries, Lessons Learned
I started with Python and stayed with Python. I considered using C++ for this project to brush up on that language, but I thought that the plotting required would be too difficult. If I can learn how to make a specific function in C++ and then use Python for simpler tasks, I would do that in the future. 

I used the following libraries in this project:
1. Numpy as np
    - Used for the usual reasons: array handling, vector operations, math functions, etc. This is pretty standard for any sciencey project.
2. Matplotlib.pyplot as plt
    - Used for plotting results and comparisons. Quite necessary for a project involving graphs. 
3. Scipy.integrate.solve_ivp
    - Finally, the most interesting library for this project was the one I used for integrators. This package offers a variety of differential equation solvers. I choose to use the RK4(5) and DOP853 methods. 

I did learn some more Python-specific syntax that will be helpful moving forward. For example, the *column_stack* function was useful for handling the 2D arrays for position and velocity. 

