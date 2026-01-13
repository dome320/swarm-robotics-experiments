# run_evolution.py
import numpy as np

from sim_runner import run_episode
from fitness_milling import milling_fitness


#hard coded evolution settings 
NUM_PARAMS = (8*8 + 8) + (8*9 + 9) + (9*2 + 2)

GENERATIONS = 50
CHILDREN_PER_GEN = 20

SIGMA = 0.1
INIT_SCALE = 0.1

SEED = 41


def main():
    np.random.seed(SEED)

    # random initial genome
    best_genome = np.random.normal(
        0.0, INIT_SCALE, size=NUM_PARAMS
    ).astype(np.float32)

    best_score = milling_fitness(
        run_episode(best_genome).positions
    )

    print(f"Initial fitness: {best_score:.4f}")

    for gen in range(GENERATIONS):
        improved = False

        for _ in range(CHILDREN_PER_GEN):
            child = best_genome + np.random.normal(
                0.0, SIGMA, size=NUM_PARAMS
            ).astype(np.float32)

            score = milling_fitness(
                run_episode(child).positions
            )

            if score > best_score:
                best_genome = child
                best_score = score
                improved = True

        print(f"Gen {gen:02d} | best = {best_score:.4f} | improved = {improved}")

    np.save("best_genome.npy", best_genome)
    print("\nSaved best genome to best_genome.npy")


if __name__ == "__main__":
    main()
