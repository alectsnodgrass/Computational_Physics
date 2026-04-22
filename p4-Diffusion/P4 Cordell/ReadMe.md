In this project, we model diffusion limited aggregation for a 2D system with a uniform flux of incoming particles from an infinite radius. We evaluate the capacity dimension of our simulated aggregate at different values of stickiness and radius then compare our results to the known DLA capacity dimension value of 1.7[1]. In addition, we evaluate the consistency of the capacity across different spawn radii to evaluate the effect of the spawn radius on the accuracy of our model. Our simulation yields a capacity dimension of 1.26 which is not very consistent with the expected value of 1.7 but remains consistent with itself as stickiness is varied. We also find that the capacity dimension of our model remains consistent for radius values approaching the aggregate. This result supports the idea that the spawn radius and stickiness do not alter the accuracy of the model.

# Introduction
As particles move through a medium such as air or water, they can bounce off other particles which effects their motion. This phenomena is referred to as Brownian motion. Now, suppose a seed exists that incoming particles can stick to and once a particle sticks to the seed, it becomes a part of the seed. This process is called diffusion-limited aggregation (DLA), and it is observed in many physical systems such as crystal formation.

DLA can be modeled using a simulation in which Brownian motion is simulated as a random walk. In other words, DLA is a process in which particles take random walks in a region where there exists a seed to which the particles stick. As more particles stick to an aggregate it forms into a fractal with a certain property that pertains to its geometry. This property is known as  capacity dimension, and it can have non-integer values as opposed to topological dimension. It is defined as Eq. 1 and can be estimated by overlaying space with boxes of side length $\epsilon$ and counting the number of boxes, N, that contain part of the aggregate. This process is repeated for decreasing $\epsilon$ then $\ln N(\epsilon)$ vs. $\ln (\frac{1}{\epsilon})$ is plotted where the slope yields the capacity dimension.

$$
D_C=\lim_{\epsilon\to 0}\frac{\ln N(\epsilon)}{\ln\left(\frac{1}{\epsilon}\right)}
\qquad\text{(1)}
$$
where $D_{C}$ is the capacity dimension, $\epsilon$ is the size of boxes, and N is the number of boxes. Furthermore, the expected value of the capacity dimension of a DLA fractal is 1.7 [1]. Accordingly, the capacity dimension of a simulated aggregate can be compared with this value to evaluate the accuracy of the simulation.


\\\\\
\\\

*
The capacity dimension can additionally be used to evaluate the effects of certain aspects of the model on its accuracy. 
# Procedure 
Our model is described in a procedure below containing the main steps to simulate DLA.
1.  Initializing a 2D space array
```python
space = np.zeros((length, length), dtype=bool)            
space[length//2, length//2] = True                        
```

A boolean array is used to represent discrete 2D space (length x length blocks): An element is true or 1 if occupied and false or 0 if empty. The seed is initialized in the center.

2. Defining a function to spawn particles
```python
def spawn(radius):
    theta = np.random.uniform(0, 2 * np.pi) 
    x, y = int(radius * np.cos(theta)), int(radius * np.sin(theta))
    return x, y
```
The spawn function is called to spawn a particle at a random angle on an approximate circular parameter around the aggregate. We played around with different conditions for the radius of the parameter. Mainly we explored two conditions based on the radius of the aggregate's longest branch, $r_{max}$: radius = $r_{max}$ + 10 and radius = 2 * $r_{max}$.

3. Implementing a kill radius
```python
def kill(i, x, y):
    global length, space, heat, radius 

    r_i = int(np.sqrt((x - length//2)**2 + (y - length//2)**2)) + 1          

    if r_i > radius + 10:
        return True
    else:
        return False
```
The kill function prevents particles from wandering far away from the circle. This tremendously improves runtime as particles are not allowed to wander far away which causes them to take much longer to find the aggregate. If the particle wanders more than 10 blocks outside the spawn radius, it is killed and respawned. Additionally, a kill radius does not alter the accuracy of the model because a particle has the same probability of being killed regardless of where it spawns and the spawn probability is uniformly distributed about the spawn parameter.

4. Implementing random walks & checking neighbors
```python
# Loop until the particle sticks to aggregate
    while (True):                       
        # RANDOM WALK ALGORITHM
        direction = random.getrandbits(2)
        
        if direction == 0:        
            y += 1
        elif direction == 1:
            y -= 1
        elif direction == 2:
            x -= 1
        else:
            x += 1

            # checks 3x3 neighborhood for crystal and ensures spot is not already occupied 
            if (space[x-1, y-1] or space[x-1, y] or space[x-1, +1] or        
                space[x,   y-1]                  or space[x,   y+1] or
                space[x+1, y-1] or space[x+1, y] or space[x+1, y+1]
                and not space[x, y]):                                            
                
                # Chance of sticking to the seed is the stickiness factor
                if np.random.rand() < stickiness:                              
                    space[x, y] = seed
                    # tracks the age of the particle that sticks to the seed
                    heat[x, y] = i + 1                                  
                    resizing_square(i, x, y)
                    resizing_circle(i, x, y)
                    break 
```
The random walk algorithm chooses a random direction for the particle to walk. The random.getrandbits() function is used for optimal runtime. Then, the 8 neighboring elements are checked if they are part of the aggregate, and the current position is required to not already be occupied. The particle takes random walks until it reaches the aggregate and the if conditions are met. Once the conditions are met, the particle has a chance of sticking. Finally, once the particle sticks, the loop breaks.

# Analysis 
Fig: Capacity Dimension for N=5000

The capacity dimension is 1.28. This is not very close to the expected value of 1.7. Why?

Fig: Capacity Dimension vs S

Fig: Capacity Dimension vs R

Fig: Compares Capacity dimension for two different spawn radius conditions

Short runtime analysis demonstration
```python
```
# Conclusions

# References
[1]

# Appendix
### Changelog Summary
The code described above took a long time to optimize the runtime for large N. Below, we share the trials and errors that led up to the code above.
#### Periodic Boundary Conditions
The first big thing we implemented was periodic boundary conditions to prevent the particles from wandering too far away from the seed. We implemented this using the mod operator as shown below:
``` python
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up':
            y = (y + 1) % length
        elif direction == 'down':
            y = (y - 1) % length
        elif direction == 'left':
            x = (x - 1) % length
        elif direction == 'right':
            x = (x + 1) % length
```


#### Resizing Boundary
Initially, we also started with a 100 block square spawn parameter centered around the seed. After running a simulation with 1000? particles we observed that the particles accumulated on the edges as they spawned in which is not realistic as the spawn parameter just represents the income of the particles at that point--- they don't just spawn there out of thin air. So, we increased N to avoid this; however, increasing N resulted in a much longer run time since the particles took longer to walk further to the seed. So, then we implemented the resizing functions to keep the spawn parameter at a balanced distance from the seed. The distance was chosen to be 3 times the radius of the aggregate point furthest from the origin. 

```python
def resizing():
    ##############################################
    # Adaptive resizing
    global length, space, heat                                          # I should probably pass these later but rn Im just testing stuff
    
    r_max = np.max(np.abs(np.argwhere(space) - length//2)) + 1          # max distance from the center seed
    if length < 3 * r_max:                                              # condition to adjust length
        new_length = 3 * r_max
        if new_length % 2 == 0:                                         # ensures new length is odd so seed can remain centered
            new_length += 1

        new_space = np.zeros((new_length, new_length), dtype=bool)      # adjusts space to new length dimensions
        new_heat = np.zeros((new_length, new_length))                   # adjusts heat to new length dimensions

        shift = (new_length - length) // 2                              # shift such that the space expands outwards from center

        crystal_indices = np.argwhere(space)                            # ^ adjusts space such that it adds space outwards from center
        for x, y in crystal_indices:
            new_space[x + shift, y + shift] = True
            new_heat[x + shift, y + shift] = heat[x, y]
        
        length = new_length                                             # updates length
        space = new_space                                               # updates space
        heat = new_heat                                                 # updates heat

    ##############################################
```

#### Circular Spawn Parameter
Yet, we still weren't satisfied with the runtime, so we changed the shape of the spawn parameter to a circle instead of a square. This change allowed us to spawn particles closer to the crystal. To visualize this, imagine two concentric squares where the particles pass into the exterior square in a uniformly distributed manner and take random walks. They will not pass through the interior square in a uniformly distributed manner. Meanwhile, for a pair of concentric circles, if the particles pass into the exterior circle in a uniformly distributed manner then they will pass through the interior circle uniformly. Naturally, we also wrote a resizing function for the circle which was based of the maximum distance of a crystal point from the center plus 10.

```python
def generate_space(radius):
    num_edge_blocks = int(2 * np.pi * radius)                                                               # num edge blocks is the integer value of circumerence
    theta = np.linspace(0, 2 * np.pi, num_edge_blocks, endpoint=False)                                      # generates num_edge_blocks angles evenly spaced around the circle
    x, y = np.round(radius * np.cos(theta)).astype(int), np.round(radius * np.sin(theta)).astype(int)       # finds the x and y positions of the points on the circle
    return x, y

x, y = generate_space(radius)
circle_set = np.unique(np.column_stack((x, y)), axis=0)         # pairs x and y defines the circle (np.unique removes any duplicates due to rounding)
```


#### Spawn Function
The early versions of the spawn function would define a segmented curve and pick a random point on that curve. We later changed this to pick a random angle at the set radius which is faster.
```python
def spawn(radius):
    theta = np.random.uniform(0, 2 * np.pi)                                                                 # generates a random angle between 0 and 2pi
    x, y = int(radius * np.cos(theta)), int(radius * np.sin(theta))
    return x, y
```

#### Checking r_max
Initially, we were checking the max radius by using np.max to sift through the distance of each point on the crystal for every loop of N. We improved the efficiency of this search by storing the initial r_max and every new point we check if its greater than the current r_max, and if so, we update r_max.

#### Random Walk Algorithm
Initially, we were using np.random.choice to choose up, down, left or right. We later changed this to np.random.randint which is faster.

#### Neighborhood Check
Initially we were slicing the neighborhood with
```python
        if (space[x-1:x+2, y-1:y+2].any()):
```
yet we found that 
```python
        if 0 < x < length - 1 and 0 < y < length - 1:                           # only checks if particle is not on the edge to avoid errors (aggregate does not approach edge so this is fine)
            if (space[x-1, y-1] or space[x-1, y] or space[x-1, y+1] or          # checks 3x3 neighborhood for crystal
                space[x,   y-1] or space[x,   y] or space[x,   y+1] or
                space[x+1, y-1] or space[x+1, y] or space[x+1, y+1]):
```
is more efficient. 

#### New Boundary Conditions
We found that making the particles bounce off the walls was more efficient than using periodic boundary conditions. And they are parity symmetric so it doesn't change much. 

```python
        # RANDOM WALK ALGORITHM
        direction = np.random.randint(4)

        # particles bounce off walls instead of periodic BC but its effectively the same.
        if direction == 0:        
            if y < length - 1:
                y += 1
        elif direction == 1:
            if y > 0:
                y -= 1
        elif direction == 2:
            if x > 0:
                x -= 1
        else:
            if x < length - 1:
                x += 1
```
