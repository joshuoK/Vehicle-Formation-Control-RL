import numpy as np

class Robot:
    def __init__(self, x, y, theta):
        """
        Initializes the robot's state.
        x initial x-position
        y initial y-position
        theta initial orientation (angle in radians)
        """
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, v, w, dt):
        """
        Updates the robot's state based on the unicycle model.
        v linear velocity (forward speed)
        w angular velocity (turning speed)
        dt time step (how much time has passed)
        """""
        # This is the core kinematic Equations
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += w * dt