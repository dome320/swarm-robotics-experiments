import os
import numpy as np

from swarmsim.world.simulate import main as sim

from sim_runner import make_world
from controller_nn import MillingNNController


BASE_DIR = os.path.dirname(__file__)
BEST_PATH = os.path.join(BASE_DIR, "outputs", "best_genome.npy")


def main():
    genome = np.load(BEST_PATH).astype(np.float32)

    world = make_world()

    for agent in world.population:
        agent.controller = MillingNNController(agent, genome)

    sim(world)


if __name__ == "__main__":
    main()
