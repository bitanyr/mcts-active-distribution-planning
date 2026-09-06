"""MCTS node statistics."""


class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior_prob=1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_prob = float(prior_prob)
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def is_terminal(self):
        return self.action is not None and self.action[0] == "stop"

    def is_expanded(self):
        return bool(self.children)