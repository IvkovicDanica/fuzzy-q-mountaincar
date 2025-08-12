import numpy as np
import random
import copy
from typing import List
from .fuzzy_logic import Build

class FQLModel:
    """
    Fuzzy Q-Learning Model.

    This class implements a fuzzy reinforcement learning agent using a Q-learning
    approach with a fuzzy inference system (FIS) for state representation.

    Attributes:
        gamma (float): Discount factor for future rewards.
        alpha (float): Learning rate.
        epsilon (float): Exploration rate (for ε-greedy policy).
        action_set_length (int): Number of possible actions.
        fis (Build): Fuzzy inference system for computing rule memberships.
        q_table (np.ndarray): Q-values table, shape = (num_rules, num_actions).
        R (List[float]): Truth values (rule activations) for the current state.
        R_ (List[float]): Truth values for the previous state.
        M (List[int]): Selected action index per rule.
        V (List[float]): State value history.
        Q (List[float]): Q-value history.
        Error (float): Temporal Difference (TD) error.
    """

    def __init__(self, gamma: float, alpha: float, epsilon: float, action_set_length: int, fis: "Build"):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_set_length = action_set_length
        self.fis = fis

        # Initialize Q-table: rows = rules, columns = actions
        self.q_table = np.zeros((self.fis.get_number_of_rules(), action_set_length))

        # Internal state variables
        self.R: List[float] = []
        self.R_: List[float] = []
        self.M: List[int] = []
        self.V: List[float] = []
        self.Q: List[float] = []
        self.Error: float = 0.0

    def truth_value(self, state_value: List[float]) -> "FQLModel":
        """
        Compute truth values (rule activations) for a given state.

        Args:
            state_value (List[float]): Crisp state values for each input variable.

        Returns:
            FQLModel: Self (for method chaining).
        """
        self.R = self.fis.get_rule_memberships(state_value)
        return self

    def action_selection(self) -> int:
        """
        Select an action using an ε-greedy strategy across fuzzy rules.

        Returns:
            int: Index of the selected action.
        """
        self.M.clear()

        # Select action for each rule
        for rule_idx in range(len(self.R)):
            if random.random() < self.epsilon:
                # Exploration
                action_index = random.randint(0, self.action_set_length - 1)
            else:
                # Exploitation
                action_index = int(np.argmax(self.q_table[rule_idx]))
            self.M.append(action_index)

        # Aggregate actions weighted by rule activations
        action_weights = np.zeros(self.action_set_length)
        for rule_idx, truth_value in enumerate(self.R):
            if truth_value > 0:
                for action_idx in range(self.action_set_length):
                    action_weights[action_idx] += truth_value * self.q_table[rule_idx, action_idx]

        # Add small noise if actions are too similar
        if np.std(action_weights) < 0.1:
            action_weights += np.random.normal(0, 0.1, self.action_set_length)

        return int(np.argmax(action_weights))

    def calculate_q_value(self):
        """
        Compute the Q-value for the previous state based on selected actions.
        """
        q_curr = sum(
            truth_value * self.q_table[index, self.M[index]]
            for index, truth_value in enumerate(self.R_)
        )
        self.Q.append(q_curr)

    def calculate_state_value(self):
        """
        Compute the state value for the current state (max-Q over all actions for each rule).
        """
        v_curr = sum(
            self.R[index] * np.max(rule_q_values)
            for index, rule_q_values in enumerate(self.q_table)
        )
        self.V.append(v_curr)

    def update_q_value(self, reward: float) -> "FQLModel":
        """
        Update the Q-table using the Temporal Difference (TD) learning rule.

        Args:
            reward (float): Immediate reward received.

        Returns:
            FQLModel: Self (for method chaining).
        """
        if not self.V or not self.Q:
            return self

        # TD Error
        self.Error = reward + self.gamma * self.V[-1] - self.Q[-1]

        # Update Q-values for rules activated in previous state
        for index, truth_value in enumerate(self.R_):
            if truth_value > 0:
                self.q_table[index, self.M[index]] += self.alpha * (self.Error * truth_value)

        return self

    def save_state_history(self):
        """
        Save current truth values (R) to R_ for the next update step.
        """
        self.R_ = copy.copy(self.R)

    def get_initial_action(self, state: List[float]) -> int:
        """
        Get the first action for an episode, clearing history.

        Args:
            state (List[float]): Initial crisp state values.

        Returns:
            int: Selected action index.
        """
        self.V.clear()
        self.Q.clear()
        self.truth_value(state)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action

    def get_action(self, state: List[float]) -> int:
        """
        Select an action for the given state (no Q-update).

        Args:
            state (List[float]): Crisp state values.

        Returns:
            int: Selected action index.
        """
        self.truth_value(state)
        return self.action_selection()

    def run(self, state: List[float], reward: float) -> int:
        """
        Perform one step of the fuzzy Q-learning algorithm.

        Args:
            state (List[float]): Current crisp state values.
            reward (float): Immediate reward received.

        Returns:
            int: Selected action index for the next step.
        """
        self.truth_value(state)
        self.calculate_state_value()
        self.update_q_value(reward)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action