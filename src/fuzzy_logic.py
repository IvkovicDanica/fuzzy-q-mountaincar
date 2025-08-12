import numpy as np
from dataclasses import dataclass
from skfuzzy.membership import trapmf
from typing import List, Tuple


@dataclass
class Trapezium:
    """
    Represents a trapezoidal membership function for fuzzy logic.
    
    Attributes:
        left (float): Start of the trapezoid's base (lower bound).
        left_top (float): Start of the top plateau.
        right_top (float): End of the top plateau.
        right (float): End of the trapezoid's base (upper bound).
    """
    left: float
    left_top: float
    right_top: float
    right: float

    def membership_value(self, input_value: float) -> float:
        """
        Calculate the membership value of an input for this trapezoidal fuzzy set.

        Args:
            input_value (float): The crisp input value.

        Returns:
            float: Membership value in the range [0, 1].
        """
        x = np.array([input_value])
        params = [self.left, self.left_top, self.right_top, self.right]
        return float(trapmf(x, params)[0])


class InputStateVariable:
    """
    Represents a fuzzy input variable containing multiple fuzzy sets.
    """

    def __init__(self, *fuzzy_sets: Trapezium):
        """
        Initialize an input variable with fuzzy sets.

        Args:
            *fuzzy_sets (Trapezium): One or more trapezoidal fuzzy sets.
        """
        self.fuzzy_set_list: Tuple[Trapezium, ...] = fuzzy_sets

    def get_fuzzy_sets(self) -> Tuple[Trapezium, ...]:
        """
        Get all fuzzy sets for this input variable.

        Returns:
            tuple[Trapezium, ...]: The fuzzy sets.
        """
        return self.fuzzy_set_list

    def get_memberships(self, value: float) -> List[float]:
        """
        Get membership values for a crisp value across all fuzzy sets.

        Args:
            value (float): The crisp input value.

        Returns:
            list[float]: Membership values for each fuzzy set.
        """
        return [fs.membership_value(value) for fs in self.fuzzy_set_list]


class Build:
    """
    Represents a fuzzy inference system builder.
    """

    def __init__(self, *input_vars: InputStateVariable):
        """
        Initialize the fuzzy system with input variables.

        Args:
            *input_vars (InputStateVariable): One or more input variables.
        """
        self.input_vars: Tuple[InputStateVariable, ...] = input_vars

    def get_input(self) -> Tuple[InputStateVariable, ...]:
        """
        Get all input variables.

        Returns:
            tuple[InputStateVariable, ...]: The input variables.
        """
        return self.input_vars

    def get_number_of_fuzzy_sets(self, input_variable: InputStateVariable) -> int:
        """
        Get the number of fuzzy sets for a given input variable.

        Args:
            input_variable (InputStateVariable): The input variable.

        Returns:
            int: Number of fuzzy sets.
        """
        return len(input_variable.get_fuzzy_sets())

    def get_number_of_rules(self) -> int:
        """
        Compute the total number of fuzzy rules.

        Returns:
            int: The number of possible rules.
        """
        num_rules = 1
        for var in self.input_vars:
            num_rules *= self.get_number_of_fuzzy_sets(var)
        return num_rules

    def get_rule_memberships(self, state: List[float]) -> List[float]:
        """
        Calculate normalized membership values for all possible rules.

        Args:
            state (list[float]): Crisp values for each input variable.

        Returns:
            list[float]: Normalized rule membership degrees.
        """
        memberships = [
            var.get_memberships(state[i]) for i, var in enumerate(self.input_vars)
        ]

        rule_memberships = []
        for idx in np.ndindex(*[len(m) for m in memberships]):
            mu = 1.0
            for var_idx, set_idx in enumerate(idx):
                mu *= memberships[var_idx][set_idx]
            rule_memberships.append(mu)

        total = sum(rule_memberships)
        if total > 0:
            rule_memberships = [m / total for m in rule_memberships]

        return rule_memberships