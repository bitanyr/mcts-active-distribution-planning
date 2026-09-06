"""Common planning runners and a solver-call budget for fair comparisons."""

import numpy as np

from core.mcts import MCTS
from env.state import NetworkState


class PhysicsBudget:
    """Count only new expensive physics solves; cached evaluations are free."""

    def __init__(self, env, max_solves):
        self.env = env
        self.max_solves = int(max_solves)
        self.start_count = int(env.physics_solve_count)

    @property
    def used(self):
        return int(self.env.physics_solve_count) - self.start_count

    @property
    def remaining(self):
        return max(0, self.max_solves - self.used)

    def terminal_value(self, state):
        key = self.env._cache_key(state, False)
        is_cached = key in self.env._cache
        if self.remaining <= 0 and not is_cached:
            return -1.0
        return self.env.evaluate_terminal_value(state)


def _better(candidate, incumbent):
    if incumbent is None:
        return True
    return float(candidate["terminal_value"]) > float(incumbent["terminal_value"])


def random_search(env, evaluation_budget=500, max_devices=32, seed=0):
    rng = np.random.default_rng(seed)
    budget = PhysicsBudget(env, evaluation_budget)
    best = None
    best_state = NetworkState()
    while budget.remaining > 0:
        state = NetworkState()
        target_length = int(rng.integers(0, int(max_devices) + 1))
        for _ in range(target_length):
            legal = [action for action in state.get_legal_actions() if action[0] != "stop"]
            if not legal:
                break
            action = legal[int(rng.integers(len(legal)))]
            state.add_device(*action)

        record = env.evaluate_terminal_state(state)
        if _better(record, best):
            best, best_state = record, state.clone()
    return best_state, best, budget.used


def greedy_search(env, evaluation_budget=500, max_devices=32, seed=0):
    """Budgeted one-step greedy search with seeded candidate ordering."""

    rng = np.random.default_rng(seed)
    budget = PhysicsBudget(env, evaluation_budget)
    state = NetworkState()
    current = env.evaluate_terminal_state(state)
    best_seen = current
    best_state = state.clone()

    for _ in range(int(max_devices)):
        if budget.remaining <= 0:
            break
        move_best = None
        move_state = None
        actions = [action for action in state.get_legal_actions() if action[0] != "stop"]
        rng.shuffle(actions)
        for action in actions:
            if action[0] == "stop" or budget.remaining <= 0:
                continue
            candidate = state.clone()
            candidate.add_device(*action)
            record = env.evaluate_terminal_state(candidate)
            if _better(record, move_best):
                move_best, move_state = record, candidate
            if _better(record, best_seen):
                best_seen, best_state = record, candidate.clone()
        if move_best is None or move_best["terminal_value"] <= current["terminal_value"]:
            break
        state, current = move_state, move_best
    return best_state, best_seen, budget.used


def mcts_plan(
    env,
    neural_net,
    evaluation_budget=500,
    max_moves=32,
    simulations_per_move=500,
    seed=0,
    add_noise=False,
):
    rng = np.random.default_rng(seed)
    budget = PhysicsBudget(env, evaluation_budget)
    state = NetworkState()
    last_policy = None


    for _ in range(int(max_moves)):
        if budget.remaining <= 0:
            break
        search = MCTS(
            neural_net=neural_net,
            terminal_evaluator=budget.terminal_value,
            num_simulations=simulations_per_move,
            rng=rng,
            exact_terminal_evaluation=False,
            stop_prior_floor=0.05,
        )
        action, last_policy = search.search(
            state,
            temperature=0.0,
            add_noise=add_noise,
        )
        if action[0] == "stop":
            break
        if not state.add_device(*action):
            break

    # This terminal candidate is part of the search budget unless already
    # cached. If the budget is exhausted, it is still hard-verified later by
    # the common final evaluator and is not silently declared feasible here.
    record = None
    key = env._cache_key(state, False)
    if budget.remaining > 0 or key in env._cache:
        record = env.evaluate_terminal_state(state)
    return state, record, budget.used, last_policy


def hard_finalize(env, state):
    """Common, uncached final assessment excluded from the search budget."""

    return env.evaluate_terminal_state(
        state,
        hard_verify=True,
        force_solve=True,
    )