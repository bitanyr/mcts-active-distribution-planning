"""AlphaZero-style MCTS for single-agent ADN planning.

During computationally intensive self-play, terminal STOP nodes can
be evaluated by the value network. The selected episode outcome is
still evaluated exactly by the physical optimization model.

Exact terminal evaluation remains available for verification and
backward compatibility.
"""

import math

import numpy as np

from core.node import MCTSNode
from data.devices import DEVICE_TYPES


class MCTS:
    def __init__(
        self,
        neural_net,
        terminal_evaluator=None,
        num_simulations=500,
        cpuct=5.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        rng=None,
        exact_terminal_evaluation=True,
        stop_prior_floor=0.05,
    ):
        self.neural_net = neural_net
        self.terminal_evaluator = terminal_evaluator
        self.num_simulations = int(num_simulations)
        self.cpuct = float(cpuct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(
            dirichlet_epsilon
        )
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng()
        )

        self.exact_terminal_evaluation = bool(
            exact_terminal_evaluation
        )
        self.stop_prior_floor = float(
            stop_prior_floor
        )

        if self.num_simulations < 1:
            raise ValueError(
                "num_simulations must be at least 1."
            )

        if self.cpuct < 0.0:
            raise ValueError(
                "cpuct cannot be negative."
            )

        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError(
                "dirichlet_epsilon must be in [0, 1]."
            )

        if self.dirichlet_alpha <= 0.0:
            raise ValueError(
                "dirichlet_alpha must be positive."
            )

        if not 0.0 <= self.stop_prior_floor < 1.0:
            raise ValueError(
                "stop_prior_floor must be in [0, 1)."
            )

        if (
            self.exact_terminal_evaluation
            and self.terminal_evaluator is None
        ):
            raise ValueError(
                "Exact terminal evaluation requires "
                "a terminal_evaluator."
            )

        self.device_types = tuple(DEVICE_TYPES)

        self.device_to_index = {
            device: index
            for index, device in enumerate(
                self.device_types
            )
        }

    @property
    def stop_index(self):
        return self.neural_net.policy_dim - 1

    def action_to_index(self, action):
        device, bus = action

        if device == "stop":
            return self.stop_index

        return (
            (int(bus) - 1) * len(self.device_types)
            + self.device_to_index[device]
        )

    def state_to_tensor(self, state):
        tensor = np.zeros(
            self.neural_net.input_dim,
            dtype=np.float32,
        )

        for device, buses in state.placements.items():
            for bus in buses:
                tensor[
                    self.action_to_index(
                        (device, bus)
                    )
                ] = 1.0

        return tensor

    def _predict_state(self, state):
        state_tensor = self.state_to_tensor(state)

        policy, value, performance = (
            self.neural_net.predict(state_tensor)
        )

        return (
            np.asarray(policy, dtype=float),
            float(value),
            performance,
        )

    def _select_child(self, node):
        sqrt_visits = math.sqrt(
            max(1, node.visit_count)
        )

        best_score = -float("inf")
        best = []

        for action, child in node.children.items():
            exploration = (
                self.cpuct
                * child.prior_prob
                * sqrt_visits
                / (1 + child.visit_count)
            )

            score = child.q_value + exploration

            if score > best_score + 1.0e-15:
                best_score = score
                best = [(action, child)]

            elif abs(
                score - best_score
            ) <= 1.0e-15:
                best.append((action, child))

        if not best:
            raise RuntimeError(
                "MCTS could not select a child."
            )

        selected = int(
            self.rng.integers(len(best))
        )

        return best[selected]

    def _apply_stop_prior_floor(
        self,
        actions,
        probabilities,
    ):
        probabilities = np.asarray(
            probabilities,
            dtype=float,
        ).copy()

        if len(actions) == 0:
            return probabilities

        total = probabilities.sum()

        if (
            total <= 0.0
            or not np.all(
                np.isfinite(probabilities)
            )
        ):
            probabilities = np.ones(
                len(actions),
                dtype=float,
            )
            total = probabilities.sum()

        probabilities /= total

        stop_positions = [
            position
            for position, action in enumerate(actions)
            if action[0] == "stop"
        ]

        if not stop_positions:
            return probabilities

        stop_position = stop_positions[0]

        if len(actions) == 1:
            probabilities[0] = 1.0
            return probabilities

        current_stop_prior = probabilities[
            stop_position
        ]

        if (
            current_stop_prior
            >= self.stop_prior_floor
        ):
            return probabilities

        non_stop_positions = [
            position
            for position in range(len(actions))
            if position != stop_position
        ]

        non_stop_total = probabilities[
            non_stop_positions
        ].sum()

        remaining_probability = (
            1.0 - self.stop_prior_floor
        )

        if non_stop_total > 0.0:
            probabilities[
                non_stop_positions
            ] *= (
                remaining_probability
                / non_stop_total
            )
        else:
            probabilities[
                non_stop_positions
            ] = (
                remaining_probability
                / len(non_stop_positions)
            )

        probabilities[
            stop_position
        ] = self.stop_prior_floor

        probabilities /= probabilities.sum()

        return probabilities

    def _evaluate_terminal_node(self, node):
        if self.exact_terminal_evaluation:
            value = float(
                self.terminal_evaluator(
                    node.state
                )
            )
        else:
            _policy, value, _performance = (
                self._predict_state(node.state)
            )

        return float(
            np.clip(value, -1.0, 1.0)
        )

    def _expand_and_evaluate(self, node):
        if node.is_terminal:
            return self._evaluate_terminal_node(
                node
            )

        policy, value, _performance = (
            self._predict_state(node.state)
        )

        legal_actions = (
            node.state.get_legal_actions()
        )

        if not legal_actions:
            return float(
                np.clip(value, -1.0, 1.0)
            )

        raw = np.array(
            [
                max(
                    0.0,
                    float(
                        policy[
                            self.action_to_index(
                                action
                            )
                        ]
                    ),
                )
                for action in legal_actions
            ],
            dtype=float,
        )

        raw = self._apply_stop_prior_floor(
            legal_actions,
            raw,
        )

        for action, prior in zip(
            legal_actions,
            raw,
        ):
            child_state = node.state.clone()

            if (
                action[0] != "stop"
                and not child_state.add_device(
                    *action
                )
            ):
                continue

            node.children[action] = MCTSNode(
                state=child_state,
                parent=node,
                action=action,
                prior_prob=float(prior),
            )

        return float(
            np.clip(value, -1.0, 1.0)
        )

    @staticmethod
    def _backpropagate(path, value):
        # The planning problem has one decision-maker. Unlike a
        # two-player zero-sum game, the value sign is not reversed
        # between consecutive tree levels.
        for node in path:
            node.visit_count += 1
            node.value_sum += value

    def _add_root_noise(self, root):
        actions = list(root.children)

        if not actions:
            return

        noise = self.rng.dirichlet(
            np.full(
                len(actions),
                self.dirichlet_alpha,
                dtype=float,
            )
        )

        mixed_priors = np.array(
            [
                (
                    (1.0 - self.dirichlet_epsilon)
                    * root.children[
                        action
                    ].prior_prob
                    + self.dirichlet_epsilon
                    * float(sample)
                )
                for action, sample in zip(
                    actions,
                    noise,
                )
            ],
            dtype=float,
        )

        mixed_priors = (
            self._apply_stop_prior_floor(
                actions,
                mixed_priors,
            )
        )

        for action, prior in zip(
            actions,
            mixed_priors,
        ):
            root.children[
                action
            ].prior_prob = float(prior)

    def _visit_probabilities(
        self,
        visits,
        temperature,
    ):
        visits = np.asarray(
            visits,
            dtype=float,
        )

        if float(temperature) <= 1.0e-8:
            probabilities = np.zeros_like(
                visits
            )

            max_visit = visits.max()
            candidates = np.flatnonzero(
                visits == max_visit
            )

            selected = int(
                self.rng.choice(candidates)
            )

            probabilities[selected] = 1.0
            return probabilities

        positive = visits > 0.0

        if not np.any(positive):
            return np.ones_like(
                visits
            ) / len(visits)

        inverse_temperature = (
            1.0 / float(temperature)
        )

        log_weights = np.full_like(
            visits,
            -np.inf,
        )

        log_weights[positive] = (
            np.log(visits[positive])
            * inverse_temperature
        )

        finite_weights = np.isfinite(
            log_weights
        )

        maximum = np.max(
            log_weights[finite_weights]
        )

        weights = np.zeros_like(visits)

        weights[finite_weights] = np.exp(
            log_weights[finite_weights]
            - maximum
        )

        total = weights.sum()

        if total <= 0.0:
            return np.ones_like(
                visits
            ) / len(visits)

        return weights / total

    def search(
        self,
        state,
        temperature=1.0,
        add_noise=False,
    ):
        root = MCTSNode(
            state=state.clone()
        )

        self._expand_and_evaluate(root)

        if add_noise:
            self._add_root_noise(root)

        for _ in range(
            self.num_simulations
        ):
            node = root
            path = [node]

            while node.is_expanded():
                _action, node = (
                    self._select_child(node)
                )

                path.append(node)

                if node.is_terminal:
                    break

            value = self._expand_and_evaluate(
                node
            )

            self._backpropagate(
                path,
                value,
            )

        actions = list(root.children)

        if not actions:
            raise RuntimeError(
                "MCTS root has no legal actions."
            )

        visits = np.array(
            [
                root.children[
                    action
                ].visit_count
                for action in actions
            ],
            dtype=float,
        )

        probabilities = (
            self._visit_probabilities(
                visits,
                temperature,
            )
        )

        chosen_position = int(
            self.rng.choice(
                len(actions),
                p=probabilities,
            )
        )

        policy_target = np.zeros(
            self.neural_net.policy_dim,
            dtype=np.float32,
        )

        for action, probability in zip(
            actions,
            probabilities,
        ):
            policy_target[
                self.action_to_index(action)
            ] = float(probability)

        return (
            actions[chosen_position],
            policy_target,
        )