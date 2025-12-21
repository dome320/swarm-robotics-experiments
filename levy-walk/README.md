# Lévy Walk Controller (RSS)

## Goal
Implement a Lévy-walk-style controller for swarm agents in the
RobotSwarmSimulator (RSS) to study exploratory behavior.

## Description
Each agent alternates between:
- Straight runs whose lengths are sampled from a heavy-tailed (Pareto) distribution
- Short random reorientation phases

This produces a Lévy-walk-like motion pattern commonly observed in
efficient search and exploration strategies.

## Implementation Notes
- No sensors are required
- Controller outputs `[forward_speed, turning_rate]`
- Designed to be compatible with RSS `MazeAgent` controllers

## Controller Implementation Sum Up:
init
The constructor initializes all of the parameters that define how the Lévy walk controller behaves and stores them as instance variables so that each agent has its own independent controller state. These parameters include the constant forward speed, the maximum angular turning rate, the heavy-tail exponent that controls run-length variability, the minimum number of steps in a straight run, and the range of steps used for turning bursts. In addition to storing these configuration values, the constructor initializes internal counters that track how many straight steps and turning steps remain, as well as the current turning rate. Finally, it immediately calls _start_new_run() so that the controller is ready to output valid movement commands as soon as the simulation begins.

_sample_run_steps
This method is responsible for generating the Lévy walk’s defining feature: heavy-tailed straight-run lengths. It samples a value from a Pareto distribution using the parameter alpha, which produces many small values and rare but extremely large values. The sampled value is then scaled by min_run_steps to convert it into an integer number of simulation timesteps. Because each timestep corresponds to a fixed amount of time and distance traveled, this heavy-tailed distribution of steps directly produces heavy-tailed movement distances, which differentiates Lévy walks from standard random or Brownian motion.

_start_new_run
This method resets the controller into a clean straight-running state. It samples a new heavy-tailed run length using _sample_run_steps() and assigns that value to run_steps_left, which determines how long the agent will continue moving straight. It also clears all turning-related state by setting the turning rate to zero and resetting the turning counter. Conceptually, this method represents the controller committing to a new plan of “move straight for N steps,” where N is randomly chosen according to a Lévy-style distribution.

_start_turn
This method initiates a reorientation event that changes the agent’s heading between straight runs. It selects a random angular velocity within the range allowed by max_turn_rate, choosing both direction (left or right) and magnitude. It also chooses how long this turn will last by sampling an integer number of timesteps from turn_steps_range. Together, the turning rate and duration determine how much the agent’s orientation changes before the next straight run begins. This turning burst introduces randomness in direction while still maintaining smooth, physically plausible motion.

get_actions
This method is the core of the controller and is called by the simulator at every timestep to determine the agent’s movement commands. It implements a simple finite-state machine with two main phases: turning and straight running. If the controller is currently in a turning phase, it continues returning a forward velocity and a nonzero angular velocity while decrementing the remaining turn steps. If the straight run has ended, it triggers a new turning burst and samples the next heavy-tailed run length. Otherwise, the controller is in the straight-run phase and returns a constant forward velocity with zero angular velocity while counting down the remaining run steps. This structure produces the characteristic Lévy walk behavior: mostly straight motion punctuated by occasional, randomly oriented reorientation events, with straight-run lengths following a heavy-tailed distribution.