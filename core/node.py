# core / node
import math

class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior_prob=1.0):
        self.state = state          
        self.parent = parent        
        self.action = action        
        self.children = {}          
        
        
        self.visit_count = 0        
        self.value_sum = 0.0        
        self.prior_prob = prior_prob 
        
    @property
    def q_value(self):
        
        if self.visit_count == 0:
            
            if self.parent and self.parent.visit_count > 0:
                return self.parent.q_value
            return 0.0 
        
        return self.value_sum / self.visit_count
    def is_expanded(self):
        
        return len(self.children) > 0