import matplotlib.pyplot as plt
from robot import Robot  # Imports your Robot from robot.py

# Times 
dt = 0.1  
SIM_TIME = 40
num_steps = int(SIM_TIME / dt)

# 1. SETUP: Create 3 robots in a vertical line (a "Search Line" start)
# Robot(x, y, theta)
robots = [
    Robot(0, -1, 0), # Bottom robot
    Robot(0,  0, 0), # Middle robot (Leader)
    Robot(0,  1, 0)  # Top robot
]

# Create a list of lists to store paths for each robot
paths_x = [[], [], []]
paths_y = [[], [], []]

print("Starting Multi-Robot Simulation...")

# 2. SIMULATION LOOP
for i in range(num_steps):
    plt.cla()  # Clear axes for the next frame
    
    # Update and Plot each robot
    for idx, r in enumerate(robots):
        # Control: For now, give them all the same constant command
        v_command = 1.0  
        w_command = 0.2  # Slight turn to see the trails better
        
        r.update(v_command, w_command, dt)

        # Store positions for trails
        paths_x[idx].append(r.x)
        paths_y[idx].append(r.y)

        # Plot the trail (different colors for each)
        colors = ['r--', 'g--', 'b--']
        plt.plot(paths_x[idx], paths_y[idx], colors[idx], label=f"Robot {idx}")
        
        # Plot the current position
        plt.plot(r.x, r.y, 'ko') # Black dot for robot

    plt.legend()
    plt.axis('equal')
    plt.pause(0.01) # Brief pause to animate

plt.show()