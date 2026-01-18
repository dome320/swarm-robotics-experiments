import random
from swarmsim.agent.control.Controller import Controller

class LevyWalkController(Controller):
    """
    Levy Walk controller for Maze Agent with the following behavior:

    - Output is [foward speed, turning speed] for each step 
    - Chooses a "heavy tail" run length
    - Goes straight for that many steps
    - one turning burst between straight runs 
    - Repeat
    """

    def __init__(
        self,
        agent,
        forward_speed=0.02,     #units per second
        max_turn_rate=1.5,      #rad/s
        alpha=1.5,              #tail exponent
        min_run_steps=15,       #minimum straight-run length (in simulation steps)
        turn_steps_range=(5, 20),
        seed=None,
    ):
        super().__init__(agent)
        self._rng = random.Random(seed)

        self.v = forward_speed
        self.max_w = max_turn_rate
        self.alpha = alpha
        self.min_run_steps = min_run_steps
        self.turn_steps_range = turn_steps_range

        self.run_steps_left = 0
        self.turn_steps_left = 0
        self.current_w = 0.0

        self._start_new_run()

    def _sample_run_steps(self) -> int:
        x = self._rng.paretovariate(self.alpha)
        return int(self.min_run_steps * x)

    def _start_new_run(self):
        self.run_steps_left = self._sample_run_steps()
        self.current_w = 0.0
        self.turn_steps_left = 0

    def _start_turn(self):
        self.current_w = self._rng.uniform(-self.max_w, self.max_w)
        self.turn_steps_left = self._rng.randint(*self.turn_steps_range)

    def get_actions(self, agent, *_args, **_kwargs):
        #turning phase
        if self.turn_steps_left > 0:
            self.turn_steps_left -= 1
            return (self.v, self.current_w)

        #if run ended start a new turn + new run
        if self.run_steps_left <= 0:
            self._start_turn()
            self._start_new_run()
            return (self.v, self.current_w)

        #straight run phase
        self.run_steps_left -= 1
        return (self.v, 0.0)

    

# Same implementatoin/world creation as in "my_first_simulation.py" 


#world Creation
from swarmsim.world.RectangularWorld import RectangularWorld, RectangularWorldConfig
world_config = RectangularWorldConfig(size=[10, 10], time_step=1 / 40)
world = RectangularWorld(world_config)

#agent Template
from swarmsim.agent.MazeAgent import MazeAgent, MazeAgentConfig
agent_config = MazeAgentConfig(position=(5, 5), agent_radius=0.1)
agent = MazeAgent(agent_config, world)

#attach levy controller
agent.controller = LevyWalkController(
    agent,
    forward_speed=0.02,
    max_turn_rate=2.0,
    alpha=1.5,
    min_run_steps=10
)


#use Spanwer to spawn multiple agents
from swarmsim.world.spawners.AgentSpawner import PointAgentSpawner

spawner = PointAgentSpawner(
    world,
    n=8,                 
    facing="away",
    avoid_overlap=True,
    agent=agent,        
    mode="oneshot",
)

world.spawners.append(spawner)

#start simulation
from swarmsim.world.simulate import main as sim
sim(world)
