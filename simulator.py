import numpy as np
from abc import abstractmethod, ABC
from agent import Agent


class Simulator(ABC):
    @abstractmethod
    def update(self):
        """Function updates all internal parameters."""
        pass

    @abstractmethod
    def retrieve_state(self):
        pass


class PandemicSimulator(Simulator):
    """Class manages the interactions between all agents"""
    def __init__(self, num_agents, limit):
        # Healthy agent gets infected as soon as it gets within proximity of an infected agent
        self.proximity = limit*0.08
        # Initialize the agents
        self.agents = [Agent(limit=limit) for i in enumerate(range(num_agents))]
        # Note that wlog the first agent is the only infected agent
        self.infected = np.zeros(num_agents)
        self.infected[0] = True

    def update(self):
        """Function first updates all agent's positions. Afterwards, it updates the health status of all agents"""
        # Update positions of all agents first
        [agent.update() for agent in self.agents]
        # Update health status of all agents
        self._udpate_health_status()

    def retrieve_state(self):
        colors = ['r' if i == True else 'b' for i in self.infected]
        return self._get_agent_positions(), colors

    def _udpate_health_status(self):
        infected_agents = np.where(self.infected == True)[0]
        agent_positions = self._get_agent_positions()
        newly_infected = list(map(lambda i: self._check_new_infections(i, agent_positions), infected_agents))
        self.infected[np.unique(np.concatenate(newly_infected))] = True

    def _check_new_infections(self, i, agent_positions):
        """If healthy agent gets within close enough proximity of an infected agent, it gets infected"""
        infected_agent_position = agent_positions[i, :]
        # Get distances to infected agent
        distances = np.sqrt(np.sum(np.power(agent_positions - infected_agent_position, 2), axis=1))
        return np.where(distances <= self.proximity)[0]

    def _get_agent_positions(self):
        return np.array([agent.get_position() for agent in self.agents])

