import heapq
import numpy as np
from dataclasses import dataclass
from pygmo import hypervolume


def crowding_distance(points, ranks=None):
    crowding = np.zeros(points.shape)
    # compute crowding distance separately for each non-dominated rank
    if ranks is None:
        ranks = non_dominated_rank(points)
    unique_ranks = np.unique(ranks)
    for rank in unique_ranks:
        current_i = ranks == rank
        current = points[current_i]
        if len(current) == 1:
            crowding[current_i] = 1
            continue
        # first normalize accross dimensions
        current = (current-current.min(axis=0))/(current.ptp(axis=0)+1e-8)
        # sort points per dimension
        dim_sorted = np.argsort(current, axis=0)
        point_sorted = np.take_along_axis(current, dim_sorted, axis=0)
        # compute distances between lower and higher point
        distances = np.abs(point_sorted[:-2] - point_sorted[2:])
        # pad extrema's with 1, for each dimension
        distances = np.pad(distances, ((1,), (0,)), constant_values=1)
        
        current_crowding = np.zeros(current.shape)
        current_crowding[dim_sorted, np.arange(points.shape[-1])] = distances
        crowding[current_i] = current_crowding
    # sum distances of each dimension of the same point
    crowding = np.sum(crowding, axis=-1)
    # normalized by dividing by number of objectives
    crowding = crowding/points.shape[-1]
    return crowding


def non_dominated_rank(points):
    ranks = np.zeros(len(points), dtype=np.float32)
    current_rank = 0
    # get unique points to determine their non-dominated rank
    unique_points, indexes = np.unique(points, return_inverse=True, axis=0)
    # as long as we haven't processed all points
    while not np.all(unique_points==-np.inf):
        _, nd_i = non_dominated(unique_points, return_indexes=True)
        # use indexes to compute inverse of unique_points, but use nd_i instead
        ranks[nd_i[indexes]] = current_rank
        # replace ranked points with -inf, so that they won't be non-dominated again
        unique_points[nd_i] = -np.inf
        current_rank += 1
    return ranks


def epsilon_metric(coverage_set, pareto_front):
    # normalize pareto front and coverage set for each 
    min_, ptp = pareto_front.min(axis=0),pareto_front.ptp(axis=0)
    pareto_front = (pareto_front-min_)/(ptp+1e-8)
    coverage_set = (coverage_set-min_)/(ptp+1e-8)
    # for every point in the pareto front, find the closest point in the coverage set
    # do this for every dimension separately
    # duplicate every point of the PF to compare with every point of the CS
    pf_duplicate = np.tile(np.expand_dims(pareto_front, 1), (1, len(coverage_set), 1))
    # distance for each dimension, for each point
    epsilon = np.abs(pf_duplicate-coverage_set)
    # for each point, take the maximum epsilon with pareto front
    epsilon = epsilon.max(-1)
    # closest point (in terms of epsilon) with PF
    epsilon = epsilon.min(-1)

    return epsilon


@dataclass
class Transition(object):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminal: bool

device = 'cpu'


def non_dominated(solutions, return_indexes=False):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    if return_indexes:
        return solutions[is_efficient], is_efficient
    else:
        return solutions[is_efficient]


def compute_hypervolume(q_set, ref):
    nA = len(q_set)
    q_values = np.zeros(nA)
    for i in range(nA):
        # pygmo uses hv minimization,
        # negate rewards to get costs
        points = np.array(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values


def nlargest(n, experience_replay, objectives, threshold=.2):
    returns = np.array([e[2][0].reward for e in experience_replay])
    # keep only used objectives
    returns = returns[:, objectives]
    # crowding distance of each point, check ones that are too close together
    distances = crowding_distance(returns)
    sma = np.argwhere(distances <= threshold).flatten()

    nd, nd_i = non_dominated(returns, return_indexes=True)
    nd = returns[nd_i]
    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of nd to compute respective distance
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(nd), 1))
    # distance to closest nd point
    l2 = np.min(np.linalg.norm(returns_exp-nd, axis=-1), axis=-1)*-1
    # all points that are too close together (crowding distance < threshold) get a penalty
    nd_i = np.nonzero(nd_i)[0]
    _, unique_i = np.unique(nd, axis=0, return_index=True)
    unique_i = nd_i[unique_i]
    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_i] = False
    l2[duplicates] -= 1e-5
    l2[sma] -= 1

    sorted_i = np.argsort(l2)
    largest = [experience_replay[i] for i in sorted_i[-n:]]
    # before returning largest elements, update all distances in heap
    for i in range(len(l2)):
        experience_replay[i] = (l2[i], experience_replay[i][1], experience_replay[i][2])
    heapq.heapify(experience_replay)
    return largest


def add_episode(transitions, experience_replay, gamma=1., max_size=100, step=0):
    # compute return
    for i in reversed(range(len(transitions)-1)):
        transitions[i].reward += gamma * transitions[i+1].reward
    # pop smallest episode of heap if full, add new episode
    # heap is sorted by negative distance, (updated in nlargest)
    # put positive number to ensure that new item stays in the heap
    if len(experience_replay) == max_size:
        heapq.heappushpop(experience_replay, (1, step, transitions))
    else:
        heapq.heappush(experience_replay, (1, step, transitions))

