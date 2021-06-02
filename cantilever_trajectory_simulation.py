from math import pi, sqrt, pow, sin
import numpy as np
import csv 
import pandas as pd 


#import sys
import argparse
import os

#import progressbar # pip install progressbar if this does not work
parser = argparse.ArgumentParser()

parser.add_argument("--spring_constant", "-k", help="set probe stiffness (N/m)", default=0.5)
parser.add_argument("--resonant_frequency", "-f", help="set cantilever nature frequency", default=25000)
parser.add_argument("--quality_factor","-q",help="set quality factor of probe", default=3)
parser.add_argument("--temperature","-t",help="set simulation temperature (Kelvin)", default=300)
parser.add_argument("--velocity","-v",help="set simulation velocity (nm/sec)", default=1)
parser.add_argument("--force_distance","-z",help="set force distance of the measurement (m)", default=5e-9)
parser.add_argument("--force_distance_end","-e",help="set end position of force distance of the measurement (m)", default=3e-10)
parser.add_argument("--save_path","-s",help="set file path")

args = parser.parse_args()

spring_constant = float(args.spring_constant) # N/m units
resonant_frequency = int(args.resonant_frequency) 
quality_factor = float(args.quality_factor)
temperature = int(args.temperature) 
probe_vel= float(args.velocity)


mass = spring_constant / pow((2 * pi * resonant_frequency), 2)
b = sqrt(spring_constant * mass) / quality_factor
dt = 5e-7 #sampling rate = 200 kHz
#total_time = dt * wave_points

equilibrium_distance = -float(args.force_distance)
equilibrium_end_pos = float(args.force_distance_end)
equilibrium_begin_pos = equilibrium_end_pos-equilibrium_distance
equilibrium_pos = equilibrium_begin_pos

wave_points = 1e6/probe_vel*(-equilibrium_distance/5e-9)


x = np.linspace(equilibrium_end_pos, equilibrium_begin_pos, 5000)
simulation_force = []
for i in x:
    val = 1e-11/i
    sim_value = 100 * pow(val, 6) - 0.0006 * pow(val, 3)
    simulation_force.append(sim_value)

velocity = 0
acceleration = 0
position_i = equilibrium_pos

eqb_pos = [0] * int(wave_points)
force_profile = [0] * int(wave_points)
positions = [0] * int(wave_points)
scaled_position = [0] * int(wave_points)
#impacts = [0] * int(wave_points/sample_points_period)

velocity_turning = 0
oversampling = 10
oversampling_loop = 0
i = j = 0


#with progressbar.ProgressBar(max_value=wave_points) as bar:


while i < (wave_points - 1):
    initial_velocity = velocity
    min_index = np.argmin(abs(x - position_i))
    forcetrace_temp = simulation_force[min_index]
        
    equilibrium_pos = equilibrium_begin_pos + (equilibrium_distance * j/(wave_points * oversampling))
    noise = np.random.normal(0, sqrt(2 * temperature * 1.38e-23 * spring_constant / (2 * pi * quality_factor * dt * resonant_frequency)))
    acceleration = 1 / mass * (-1 * spring_constant * (position_i - equilibrium_pos) - b * velocity + forcetrace_temp + noise)
     
    velocity += (acceleration * dt)
    position_i += (velocity * dt)
    j += 1

    if (velocity*initial_velocity < 0 and velocity > 0):
        velocity_turning += 1
        # impacts[velocity_turning] = forcetrace_temp

    if (oversampling_loop == oversampling):
        i += 1
        positions[i] = position_i
        eqb_pos[i] = equilibrium_pos
        force_profile[i] = forcetrace_temp
        oversampling_loop = 0
    oversampling_loop += 1 
#    bar.update(i)

for p in range(1,len(positions)):
    scaled_position[p]  = positions[p] - (equilibrium_distance * p/wave_points + equilibrium_begin_pos)

eqb_pos = np.asarray([p * 1e9 for p in eqb_pos])
scaled_position = np.asarray([sp * 1e9 for sp in scaled_position])

eqb_pos_filename = 'pos_k{k}_T{T}_v{v}_Q{Q}_DFS.csv'.format(k=spring_constant,T=temperature,v=probe_vel,Q=args.quality_factor)
scaled_position_filename = 'defl_k{k}_T{T}_v{v}_Q{Q}_DFS.csv'.format(k=spring_constant,T=temperature,v=probe_vel,Q=args.quality_factor)

pd.DataFrame(eqb_pos).to_csv(os.path.join(args.save_path,eqb_pos_filename), header=None)
pd.DataFrame(scaled_position).to_csv(os.path.join(args.save_path,scaled_position_filename), header=None)