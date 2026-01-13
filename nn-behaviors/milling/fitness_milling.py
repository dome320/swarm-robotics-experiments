"""
Some specifications to be considered milling: 
- Agents move around a common center (the swarm centroid)
- agents rotate consistently in one direction (CW or CCW)
- agents distance from the center stays  stable 
- The behavior is stable over time (not just a 1-second fluke)
- So fitness should be high when the swarm forms a stable rotating ring.

How I want to set up the score:
- is continuous
- works with only positions
- isn’t tricked by “spin in place” or “run away fast”
- is comparable across episodes
"""

# fitness_milling.py
import numpy as np


# ----------------------------
# Tunable constants (keep simple)
# ----------------------------
EPS = 1e-8

# Ignore the early "settling" period (fraction of episode)
BURN_IN_FRAC = 0.25

# Weights for combining sub-scores
W_COHERENCE = 0.6
W_RADIUS = 0.4

# Penalties / thresholds
MIN_MEAN_RADIUS = 0.30          # discourage collapsing to centroid (increased from 0.10)
COLLAPSE_EXP = 6                # makes collapse MUCH more expensive
MAX_CENTROID_DRIFT = 3.0        # world-size dependent (10x10 world → 3 is moderate)
DRIFT_PENALTY_WEIGHT = 0.25     # how strong to penalize drifting swarms


# ----------------------------
# Helpers
# ----------------------------
def compute_centroids(positions: np.ndarray) -> np.ndarray:
    """
    positions: (T, N, 2)
    returns centroids: (T, 2)
    """
    return positions.mean(axis=1)


def compute_velocities(positions: np.ndarray) -> np.ndarray:
    """
    Simple finite difference velocity:
      v[t,i] = pos[t+1,i] - pos[t,i]
    positions: (T, N, 2)
    returns velocities: (T-1, N, 2)
    """
    return positions[1:] - positions[:-1]


def angular_momentum_proxy(positions: np.ndarray) -> np.ndarray:
    """
    Computes signed 2D "angular momentum proxy" about centroid:
      L[t,i] = r_x * v_y - r_y * v_x
    where r is position relative to centroid, v is velocity.

    positions: (T, N, 2)
    returns L: (T-1, N)
    """
    centroids = compute_centroids(positions)          # (T, 2)
    r = positions - centroids[:, None, :]             # (T, N, 2)
    v = compute_velocities(positions)                 # (T-1, N, 2)
    r0 = r[:-1]                                       # align with v (T-1, N, 2)

    rx = r0[..., 0]
    ry = r0[..., 1]
    vx = v[..., 0]
    vy = v[..., 1]
    L = rx * vy - ry * vx
    return L


def rotation_coherence(L: np.ndarray) -> np.ndarray:
    """
    Per-timestep coherence in [0,1]:
      coherence[t] = |sum_i L[t,i]| / (sum_i |L[t,i]| + eps)

    L: (T-1, N)
    returns coherence: (T-1,)
    """
    num = np.abs(np.sum(L, axis=1))
    den = np.sum(np.abs(L), axis=1) + EPS
    return num / den


def radius_scores(positions: np.ndarray) -> np.ndarray:
    """
    Scores "ring-ness" per timestep based on radius consistency.

    Steps:
      - compute radii R[t,i] = ||pos[t,i] - centroid[t]||
      - compute coefficient of variation: cv = std/mean
      - convert to score: 1 / (1 + cv)
      - apply collapse penalty if mean radius too small

    returns radius_score: (T,)
    """
    centroids = compute_centroids(positions)              # (T, 2)
    r = positions - centroids[:, None, :]                 # (T, N, 2)
    R = np.linalg.norm(r, axis=2)                         # (T, N)

    mean_R = np.mean(R, axis=1)                           # (T,)
    std_R = np.std(R, axis=1)                             # (T,)
    cv = std_R / (mean_R + EPS)                           # (T,)

    score = 1.0 / (1.0 + cv)                              # higher when radii are consistent

    # Penalize collapsing into centroid (mean radius too small)
    collapse_mask = mean_R < MIN_MEAN_RADIUS
    if np.any(collapse_mask):
        # MUCH harsher penalty: ratio^COLLAPSE_EXP quickly drives score toward ~0
        score = score.copy()
        ratio = mean_R[collapse_mask] / (MIN_MEAN_RADIUS + EPS)
        score[collapse_mask] *= np.power(ratio, COLLAPSE_EXP)

    return score


def centroid_drift_penalty(positions: np.ndarray) -> float:
    """
    Penalize episodes where the centroid travels too far (swarm translating as a blob).
    returns penalty multiplier in (0,1], where 1 means no penalty.
    """
    centroids = compute_centroids(positions)  # (T,2)
    drift = float(np.linalg.norm(centroids[-1] - centroids[0]))

    # soft penalty: no penalty up to MAX_CENTROID_DRIFT, then decay
    if drift <= MAX_CENTROID_DRIFT:
        return 1.0

    excess = drift - MAX_CENTROID_DRIFT
    # simple smooth decay: 1 / (1 + k*excess)
    return 1.0 / (1.0 + DRIFT_PENALTY_WEIGHT * excess)


def burn_in_slice(T: int, frac: float = BURN_IN_FRAC) -> slice:
    """
    Returns a slice that skips the early portion of the episode.
    """
    start = int(T * frac)
    return slice(start, T)


# ----------------------------
# Public API
# ----------------------------
def milling_fitness(positions: np.ndarray) -> float:
    """
    Compute a single milling fitness score from positions (T, N, 2).

    Output: float, higher is better.

    We combine:
      - rotation coherence (does everyone rotate same direction?)
      - radius stability / ring-ness (do they form a stable ring?)
      - centroid drift penalty (optional)

    Notes:
      - We score after burn-in to allow emergence.
      - Coherence is computed on (T-1) frames (because velocity uses differences).
    """
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(f"positions must be (T,N,2). Got {positions.shape}")

    T = positions.shape[0]
    if T < 3:
        return 0.0  # too short to evaluate motion

    # burn-in for position-based signals
    pos_slice = burn_in_slice(T)
    pos_eval = positions[pos_slice]  # (T', N, 2)
    T_eval = pos_eval.shape[0]
    if T_eval < 3:
        return 0.0

    # Rotation coherence uses L which has length (T_eval-1)
    L = angular_momentum_proxy(pos_eval)          # (T_eval-1, N)
    coh = rotation_coherence(L)                   # (T_eval-1,)
    coh_score = float(np.mean(coh))

    # Radius stability uses full positions length (T_eval)
    rad = radius_scores(pos_eval)                 # (T_eval,)
    rad_score = float(np.mean(rad))

    # Combine core scores
    core = W_COHERENCE * coh_score + W_RADIUS * rad_score

    # Optional drift penalty
    drift_mult = centroid_drift_penalty(pos_eval)

    fitness = core * drift_mult

    # Keep in a sane range
    return float(np.clip(fitness, 0.0, 1.0))
