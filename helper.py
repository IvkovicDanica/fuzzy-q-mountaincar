# Functions only for pretty printing and explanations

from typing import List, Optional
from itertools import product
import numpy as np
import random
import copy

def explain_rule_strengths(
    system,
    state: List[float],
    labels: Optional[List[List[str]]] = None,
    decimals: int = 4,
) -> None:
    """
    Print a step-by-step explanation of rule strength calculation.
    
    Parameters:
        system: Build - the fuzzy system container with input variables.
        state: List[float] - crisp values for each input variable (same order).
        labels: Optional list of lists of labels for fuzzy sets per variable.
                If None, generic labels will be used.
        decimals: number of decimal places for printed membership values.
    """
    # Basic info
    n_inputs = len(system.get_input())
    if len(state) != n_inputs:
        raise ValueError("Length of state must match number of input variables.")

    fmt = f".{decimals}f"
    print("=" * 80)
    print("FUZZY INFERENCE — Step-by-step evaluation".center(80))
    print("=" * 80)
    print("\n1) Crisp input values:")
    for i, val in enumerate(state, start=1):
        print(f"   Input {i:>2}: value = {val}")

    # Membership values per fuzzy set
    print("\n2) Memberships per input variable (for the given crisp values):")
    memberships = []
    generated_labels = []
    for i, (var, val) in enumerate(zip(system.get_input(), state), start=1):
        sets = var.get_fuzzy_sets()
        lbls = (labels[i-1] if labels and i-1 < len(labels) else
                [f"V{i}_S{j+1}" for j in range(len(sets))])
        generated_labels.append(lbls)
        print(f"\n   Input {i} (value = {val}):")
        for j, fs in enumerate(sets, start=1):
            mu = fs.membership_value(val)
            print(f"     {j:>2}. {lbls[j-1]:<12} | params = ({fs.left}, {fs.left_top}, {fs.right_top}, {fs.right})"
                  f" -> μ = {mu:{fmt}}")
        memberships.append([fs.membership_value(val) for fs in sets])

    # Build all rules (cartesian product of fuzzy-set indices)
    print("\n3) Building rule combinations and computing unnormalized strengths:")
    index_ranges = [range(len(ms)) for ms in memberships]
    rule_indices = list(product(*index_ranges))
    unnorm = []
    print("\n   {:>4} | {:<30} | {:<20} | {:>10}".format("Rule", "Antecedents", "Memberships", "Unnorm"))
    print("   " + "-" * 70)
    for k, idx_tuple in enumerate(rule_indices, start=1):
        antecedents = [generated_labels[var_idx][set_idx] for var_idx, set_idx in enumerate(idx_tuple)]
        per_mus = [memberships[var_idx][set_idx] for var_idx, set_idx in enumerate(idx_tuple)]
        prod = 1.0
        for m in per_mus:
            prod *= m
        unnorm.append(prod)
        mus_str = ", ".join(f"{m:{fmt}}" for m in per_mus)
        print(f"   {k:>4} | {', '.join(antecedents):<30} | {mus_str:<20} | {prod:10.{decimals}f}")

    # Normalization
    total = sum(unnorm)
    print("\n4) Normalization:")
    print(f"   Sum of unnormalized strengths = {total:.{decimals}f}")
    if total > 0:
        norm = [u / total for u in unnorm]
    else:
        norm = [0.0 for _ in unnorm]

    print("\n   {:>4} | {:<30} | {:<20} | {:>10} | {:>10}".format(
        "Rule", "Antecedents", "Memberships", "Unnorm", "Norm"))
    print("   " + "-" * 90)
    for k, (idx_tuple, u, nrm) in enumerate(zip(rule_indices, unnorm, norm), start=1):
        antecedents = [generated_labels[var_idx][set_idx] for var_idx, set_idx in enumerate(idx_tuple)]
        per_mus = [memberships[var_idx][set_idx] for var_idx, set_idx in enumerate(idx_tuple)]
        mus_str = ", ".join(f"{m:{fmt}}" for m in per_mus)
        print(f"   {k:>4} | {', '.join(antecedents):<30} | {mus_str:<20} | {u:10.{decimals}f} | {nrm:10.{decimals}f}")

    # 5) Final mapping
    print("\n5) Final rule mapping (sorted by normalized strength desc):")
    sorted_rules = sorted(enumerate(norm, start=1), key=lambda x: x[1], reverse=True)
    for rank, (rule_idx, nrm) in enumerate(sorted_rules, start=1):
        idx_tuple = rule_indices[rule_idx - 1]
        antecedents = [generated_labels[var_idx][set_idx] for var_idx, set_idx in enumerate(idx_tuple)]
        print(f"   {rank:>2}. Rule {rule_idx}: IF " +
              " AND ".join(f"{lbl}" for lbl in antecedents) +
              f" => normalized strength = {nrm:.{decimals}f}")

    print("\n" + "=" * 80)
    

class FQLModelVerbose:
    """
    Simple Fuzzy Q-Learning model with per-rule action choices and aggregation.
    """
    def __init__(self, gamma: float, alpha: float, epsilon: float, action_set_length: int, fis):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.action_set_length = int(action_set_length)
        self.fis = fis

        # Q-table: rows = rules, cols = actions
        self.q_table = np.zeros((self.fis.get_number_of_rules(), self.action_set_length))

        # History / internal
        self.R_prev: List[float] = []         # rule strengths for previous state
        self.M_prev: List[int] = []           # per-rule chosen actions for previous state
        self.V_history: List[float] = []
        self.Q_history: List[float] = []
        self.td_error: float = 0.0

    # ---- core step method with detailed printing ----
    def step_verbose(self, state: List[float], reward: float, decimals: int = 4) -> int:
        """
        One fuzzy-Q step with verbose printing.

        Args:
            state: current crisp state (list of input values)
            reward: reward received for the transition from previous state -> current state
            decimals: printing precision
        Returns:
            selected action index (int)
        """
        fmt = f".{decimals}f"
        print("=" * 80)
        print(f"STATE (t): {state}")
        # 1) per-variable memberships
        print("\nPer-variable memberships:")
        per_var_memberships = []
        for i, var in enumerate(self.fis.get_input(), start=1):
            membs = var.get_memberships(state[i-1])
            per_var_memberships.append(membs)
            labels = [f"S{j+1}" for j in range(len(membs))]
            memb_str = ", ".join(f"{lab}:{val:{fmt}}" for lab, val in zip(labels, membs))
            print(f"  Input {i}: {memb_str}")

        # 2) normalized rule strengths R_t
        R_t = self.fis.get_rule_memberships(state)
        print("\nNormalized rule strengths (R_t):")
        for j, r in enumerate(R_t, start=1):
            print(f"  Rule {j:>2}: R_t = {r:{fmt}}")
        # 3) compute V_t
        V_t = sum(R_t[j] * np.max(self.q_table[j, :]) for j in range(len(R_t)))
        self.V_history.append(V_t)
        print(f"\nState value V_t = sum_j R_t[j] * max_a Q[j,a] = {V_t:{fmt}}")

        # 4) compute Q_prev (aggregated Q for previous state)
        if self.R_prev and self.M_prev:
            Q_prev = sum(self.R_prev[j] * self.q_table[j, self.M_prev[j]] for j in range(len(self.R_prev)))
        else:
            Q_prev = 0.0
        self.Q_history.append(Q_prev)
        print(f"Aggregated Q_prev (for previous state): {Q_prev:{fmt}}")
        print(f"Received reward r_t = {reward:{fmt}}")

        # 5) TD error and Q-table update (using previous state's R_prev and M_prev)
        td = reward + self.gamma * V_t - Q_prev
        self.td_error = td
        print(f"\nTD error δ = r_t + γ * V_t - Q_prev = {td:{fmt}}")

        if self.R_prev and self.M_prev:
            print("\nUpdating Q-table for rules active in previous state (showing changed entries):")
            for j, r_prev in enumerate(self.R_prev):
                if r_prev > 0.0:
                    a_prev = self.M_prev[j]
                    before = self.q_table[j, a_prev]
                    delta = self.alpha * td * r_prev
                    self.q_table[j, a_prev] = before + delta
                    after = self.q_table[j, a_prev]
                    print(f"  Rule {j+1:>2}, action {a_prev}: Q_before={before:{fmt}} + (α*δ*R_prev)={delta:{fmt}} -> Q_after={after:{fmt}}")
        else:
            print("\nNo previous rule activations recorded — skipping Q update.")

        # 6) per-rule action selection for current state (ε-greedy)
        M_t = []
        print("\nPer-rule action choices for current state (ε-greedy per rule):")
        for j in range(len(R_t)):
            if random.random() < self.epsilon:
                act = random.randint(0, self.action_set_length - 1)
                reason = "explore"
            else:
                act = int(np.argmax(self.q_table[j, :]))
                reason = "exploit"
            M_t.append(act)
            print(f"  Rule {j+1:>2}: chosen action = {act} ({reason})")

        # 7) aggregate action weights and select final action
        action_weights = np.zeros(self.action_set_length)
        for j, rj in enumerate(R_t):
            if rj > 0.0:
                action_weights += rj * self.q_table[j, :]

        # add small jitter if uninformative
        if np.std(action_weights) < 1e-6:
            jitter = np.random.normal(0, 0.01, self.action_set_length)
            action_weights += jitter
            print("\nAction weights had nearly zero variance, added tiny jitter to break ties.")

        print("\nAggregate action weights (W[a] = sum_j R_t[j] * Q[j,a]):")
        for a, w in enumerate(action_weights):
            print(f"  Action {a}: W = {w:{fmt}}")
        final_action = int(np.argmax(action_weights))
        print(f"\nSelected final action: {final_action}")

        # 8) Save current R and M as previous for the next step
        self.R_prev = copy.copy(R_t)
        self.M_prev = copy.copy(M_t)

        print("=" * 80 + "\n")
        return final_action