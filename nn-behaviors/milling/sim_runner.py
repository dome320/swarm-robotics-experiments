from dataclasses import dataclass
import numpy as np

from swarmsim.world.RectangularWorld import RectangularWorld, RectangularWorldConfig
from swarmsim.agent.MazeAgent import MazeAgent, MazeAgentConfig
from swarmsim.world.spawners.AgentSpawner import PointAgentSpawner
from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor
from swarmsim.agent.control.StaticController import StaticController

from controller_nn import MillingNNController

from swarmsim import metrics



# hard coded parameters for now but could be changed later
N_AGENTS = 6
T_STEPS = 400


@dataclass
class EpisodeData:
    positions: np.ndarray


def make_world() -> RectangularWorld:
    #generic RSS world creation 
    world_config = RectangularWorldConfig(
        size=[10, 10],
        time_step=1/40,
    )
    world = RectangularWorld(world_config)

    #template for angents which spawner clones
    agent_cfg = MazeAgentConfig(
        position=(5, 5),
        agent_radius=0.1,
    )
    template_agent = MazeAgent(agent_cfg, world)

    #placeholder
    template_agent.controller = StaticController(output=[0.0, 0.0])

    #attatching binary sensor to template agent
    sensor = BinaryFOVSensor(
        template_agent,
        theta=0.45,
        distance=2.0,
    )
    template_agent.sensors.append(sensor)

    #create spawner 
    spawner = PointAgentSpawner(
        world,
        n=N_AGENTS,
        facing="away",
        avoid_overlap=True,
        agent=template_agent,
        mode="oneshot",
    )
    world.spawners.append(spawner)

    world.step()

    return world


def collect_positions(world) -> np.ndarray:
    pos = []
    for a in world.population:
        x = a.get_x_pos()
        y = a.get_y_pos()
        pos.append([x, y])
    return np.array(pos, dtype=np.float32)


def run_episode(genome) -> float:
    world = make_world()

    #attach evolved controller
    for agent in world.population:
        agent.controller = MillingNNController(agent, genome)

    #attach metrics 
    circ = metrics.Circliness(history=T_STEPS, avg_history_max=450)

    # bind it
    circ.population = world.population

    world.metrics = [circ]

    #run sim 
    for _ in range(T_STEPS):
        world.step()

    return extract_circliness(world)


def extract_circliness(world) -> float:
    if not world.metrics:
        return 0.0
    m = world.metrics[0]
    return float(m.average if getattr(m, "instantaneous", False) else m.value)



