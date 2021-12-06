import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulator import PandemicSimulator


LIMIT = 1000


class AnimatedScatter(object):
    def __init__(self, simulator, num_agents, epochs=100, **kwargs):
        # Set constants for figure
        self.limit = LIMIT
        self.range = [-self.limit * 1.1, self.limit * 1.1, -self.limit * 1.1, self.limit * 1.1]

        # Initialize the simulator
        self.simulator = simulator(num_agents, self.limit, **kwargs)

        # Create Matplotlib Figure
        self.fig, self.ax = plt.subplots()
        self.ax.axis(self.range)
        # Create Animation
        self.anim = FuncAnimation(self.fig, self.update,
                                  frames=epochs,
                                  interval=1,
                                  init_func=self.init,
                                  blit=True)

    def init(self):
        xy, color = self.simulator.retrieve_state()
        scatter = self.ax.scatter(xy[:, 0], xy[:, 1], c=color)
        return scatter,

    def update(self, i):
        # Reset the figure
        self.ax.clear()
        self.ax.axis(self.range)

        # Update Simulator
        self.simulator.update()

        # Retrieve the positions for each agent
        xy, color = self.simulator.retrieve_state()
        scatter = self.ax.scatter(xy[:, 0], xy[:, 1], c=color)
        return scatter,

    def save(self, path):
        self.anim.save(path, writer='pillow')


if __name__ == '__main__':
    anim_ = AnimatedScatter(simulator=PandemicSimulator, num_agents=50)
    anim_.save('./animations/animation.gif')
