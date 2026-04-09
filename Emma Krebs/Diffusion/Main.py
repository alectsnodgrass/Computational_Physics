import Diffusion_Header
import numpy as np

# Starting parameters
n = 3 # Number of iterations of our octree. Depends on our grid space.
num_particles = 100 # Number of particles used in simulation

grid_value = 8**n
grid = [grid_value, grid_value, grid_value]
center = [grid_value/2, grid_value/2, grid_value/2]
count = 0
probability = 1 # probability of something sticking when encountering a particle
current_maximum = 0 # Will be updated as our program exapnds. 
generation_distance = 5 # The distance to the surface of a sphere for generating a new particle
kill_distance = 10 # If particle wanders past this point, it is killed

# Generate the very first seed and node, known as root node
root = Diffusion_Header.Node(center, grid, 0)
seed_particle = Diffusion_Header.Particle(center, probability)
root.particles.append(seed_particle)

# Now subdivide it!
Diffusion_Header.subdivide(root, 0, center)

# Now start looping through all the particles
while count < num_particles:
    
    count += 1
    # Spawn new particle. Start by getting a random location on our generation sphere
    location = Diffusion_Header.generation_sphere(current_maximum, generation_distance, center)
    particle = Diffusion_Header.Particle(location, probability)

    while particle.stuck == False:

         # Checks to see if particle has gone too far
        test = Diffusion_Header.kill_or_be_killed(particle.location, kill_distance, current_maximum, center)
        
        if test == True:
            count -=1 # Remove the particle
            del particle # Remove the waste of space...
            continue
        else:
            particle.random_walk()
            nearby_points = Diffusion_Header.generate_nearby_point(particle.location)

            total_location_list = [] # Total list of particles we need to compare with for out point
            for point in nearby_points:
                location_particles = Diffusion_Header.find_node(root, point)
                total_location_list += location_particles

            






w = Diffusion_Header.generate_nearby_point([250, 250, 250])
print(len(w))
print(w)




