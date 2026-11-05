"""Double DQN with online, uniform replay and proportional replay variants."""

from .deep_q_learning import DeepQLearningAgent


class DoubleDeepQLearningAgent(DeepQLearningAgent):
    def __init__(self, input_size, action_space_size, **kwargs):
        super().__init__(input_size, action_space_size, double=True, use_replay=False, **kwargs)
        self.name = "DoubleDQN (online)"


class DDQNWithERAgent(DeepQLearningAgent):
    def __init__(self, input_size, action_space_size, **kwargs):
        super().__init__(input_size, action_space_size, double=True, **kwargs)
        self.name = "DDQN+ER"


class DDQNWithPERAgent(DeepQLearningAgent):
    def __init__(self, input_size, action_space_size, **kwargs):
        super().__init__(input_size, action_space_size, double=True, prioritized=True, **kwargs)
        self.name = "DDQN+PER"
