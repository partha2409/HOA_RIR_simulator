Bare minimum implementation of HOA RIR simulator

Currently supports:
 - Shoebox rooms - fs 24k
 - various material types for walls, floors, ceiling
 - HOA microphone (random placement, 0.5m away from any wall)
 - 64 stationary sources - first 32 omni, next 16 cardioid, last 16 randomly sucardioid and hypercardioid. Directional sources - some face mic, some away from mic.
 - 6 moving trajectories - first 2 clcose to floor for footsteps. first 4 are omni, last 2 are cardioid
 - moving trajectories - 2,3,4,5m trajectories depnding on room size. 26 point RIR in each trajectory. Moving cardioid sources orientation is in the direction of  motion.

 Metadata stores all these information like source postions, directivity, orientation, distance to microphone, room dims, wall types and so on.


To run:

python simulate_rirs.py --args
