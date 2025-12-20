#World Creation 
from swarmsim.world.RectangularWorld import RectangularWorld, RectangularWorldConfig
world_config = RectangularWorldConfig(size=[10, 10], time_step=1 / 40)
world = RectangularWorld(world_config)

# Agent Defintion 
from swarmsim.agent.MazeAgent import MazeAgent, MazeAgentConfig
agent_config = MazeAgentConfig(position=(5, 5), agent_radius=0.1)
agent = MazeAgent(agent_config, world)
# world.population.append(agent)

# Multiple Agent Generation (Swarm)
from swarmsim.world.spawners.AgentSpawner import PointAgentSpawner
spawner = PointAgentSpawner(world, n=6, facing="away", avoid_overlap=True, agent=agent, mode="oneshot")
world.spawners.append(spawner)


#Add in a controller
from swarmsim.agent.control.StaticController import StaticController
controller = StaticController(output=[0.01, 0.1])  # 10 cm/s forwards, 0.1 rad/s clockwise.
agent.controller = controller

#FOV for Controller
from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor
sensor = BinaryFOVSensor(agent, theta=0.45, distance=2,)
agent.sensors.append(sensor)

#read the sensor data and change how the robot moves
from swarmsim.agent.control.BinaryController import BinaryController
controller = BinaryController(a=(0.02, -0.5), b=(0.02, 0.5), agent=agent)
agent.controller = controller

#Start the simulation
from swarmsim.world.simulate import main as sim

sim(world)
# sim(world, start_paused=True)

