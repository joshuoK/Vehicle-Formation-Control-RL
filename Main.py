import matplotlib.pyplot as plt
from robot import Robot  # Imports your Robot from robot.py

# Times 
dt = 0.1  # Time step in seconds
SIM_TIME = 20  # Total simulation time in seconds

# Setup
# Create a robot starting at (0, 0) and facing 0 radians (to the right)
robot = Robot(0, 0, 0)

# Calculate the number of steps in the simulation
num_steps = int(SIM_TIME / dt)

# Lists to store the robot's path for plotting
path_x = []
path_y = []

print("Starting simulation)")

# Simulation Loop
for i in range(num_steps):
    
    # Control Commands 
    v_command = 1.0  # constant forward speed
    w_command = 0.5  # constant turning speed

    # Update the robot's state
    robot.update(v_command, w_command, dt)

    # Store the robot's new position for plotting its trail
    path_x.append(robot.x)
    path_y.append(robot.y)

    # Visualization
    plt.cla()  # Clear the current axes
    
    # Plot the trail of the robot
    plt.plot(path_x, path_y, 'g--', label="Robot Trail") # 'g--' = green dashed line

    # Plot the robot's current position as a blue circle
    plt.plot(robot.x, robot.y, 'bo', markersize=10, label="Robot") # 'bo' = blue circle
    
    # Set plot limits and labels
    plt.xlim(-5, 15)
    plt.ylim(-5, 15)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Simulator Base")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') 

    # Pause to create a visual simulation
    plt.pause(0.01)

print("Simulation finished.")
plt.show()