import numpy as np
from cmath import polar

INITIAL_SPEED = 100
BOUNCE_LOSS = 0.9
VELOCITY_GRADIENT = 5
INERTIA = 0.1


class Agent:
    """Class holds all functionalities of an free-moving agent"""
    def __init__(self, limit):
        self.limit = limit
        self.velocity = VELOCITY_GRADIENT
        self.inertia = INERTIA * (2*np.pi)
        self.bounce_loss = BOUNCE_LOSS
        # Cartesian Coordinates
        self.x, self.y = np.random.uniform(-limit, limit), np.random.uniform(-limit, limit)
        # Trajectory
        initial_speed = INITIAL_SPEED
        initial_direction = np.random.uniform(0, 2*np.pi)
        self.trajectory = complex(initial_speed*np.cos(initial_direction), initial_direction*np.sin(initial_direction))

    def update(self):
        """Function first updates the trajectory of the agent and then maps it to cartesian coordinates."""
        # Update trajectory
        speed, direction = polar(self.trajectory)
        direction += np.random.normal(0, self.inertia)
        speed += np.random.normal(0, self.velocity)
        self.trajectory = complex(speed*np.cos(direction), speed*np.sin(direction))
        # Update cartesian coordinates
        self.x, self.y = self._update_coordinate(self.x, True), self._update_coordinate(self.y, False)

    def _update_coordinate(self, coord, horizontal):
        """
        Function updates an individual cartesian coordinate.
        Importantly, it takes the environment into respect.
        :param coord: Current cartesian coordinate
        :param horizontal: True, if input coord is x axis
                           False, if input coord is y axis
        :return: new cartesian coordinate
        """
        trig_func = np.sin if horizontal else np.cos
        while True:
            # Extract speed and direction from trajectory
            speed, direction = polar(self.trajectory)
            new_coord = coord + trig_func(direction)*speed
            if abs(new_coord) > self.limit:
                # Let agent bounce of wall
                if horizontal:
                    self.trajectory = complex(self.trajectory.real, self.trajectory.imag * -1)
                else:
                    self.trajectory = complex(self.trajectory.real * -1, self.trajectory.imag)
                # Bounce slows down agent
                speed, direction = polar(self.trajectory)
                self.trajectory = complex(self.bounce_loss*speed*np.cos(direction), self.bounce_loss*speed*np.sin(direction))
            else:
                break
        return new_coord

    def get_position(self):
        return [self.x, self.y]


