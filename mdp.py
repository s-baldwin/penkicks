import pandas as pd
import numpy as np

def load_clean_data(path):
    """
    Loads the cleaned dataset. 
    Must contain columns:
      - prev_kick  (0=Right, 1=Left)
      - action_kicker (0=Right, 1=Left)
    """
    return pd.read_excel(path)

def build_transition_matrix(df):
    """
    Builds the simple 2x2 transition matrix:
        P[next kick | previous kick]

    States:
        0 = Right
        1 = Left
    """

    # Count transitions
    counts = np.zeros((2,2))

    for _, row in df.iterrows():
        prev_k = int(row["prev_kick"])
        next_k = int(row["action_kicker"])
        counts[prev_k, next_k] += 1

    # Normalize to get probabilities
    P = counts / counts.sum(axis=1, keepdims=True)

    return P, counts

def compute_steady_state(P):
    """
    For a 2x2 Markov chain, the steady-state vector π solves:
        πP = π  and  π1 + π2 = 1

    Closed-form solution:
        π_L = P[0,1] / (P[0,1] + P[1,0])
        π_R = 1 - π_L
    """
    p01 = P[0,1]
    p10 = P[1,0]

    pi_L = p01 / (p01 + p10)
    pi_R = 1 - pi_L

    return np.array([pi_R, pi_L])  # order: [Right, Left]

def payoff_matrix():
    """
    Returns the simple 2x2 payoff matrix M:
        Rows = kicker direction (R,L)
        Columns = keeper direction (R,L)

    Mismatch => goal (1)
    Match => save (0)
    """
    return np.array([
        [0, 1],  # kicker Right vs keeper Right/Left
        [1, 0]   # kicker Left  vs keeper Right/Left
    ])

def expected_payoff(pi, q, M):
    """
    Computes expected payoff using:
        E = πᵀ M q
    π = steady state kicker distribution
    q = keeper strategy vector
    M = payoff matrix
    """
    return pi @ M @ q

def main():
    # ------------------------------
    # 1. Load cleaned data
    # ------------------------------
    df = load_clean_data("clean_data.xlsx")

    # ------------------------------
    # 2. Build transition matrix
    # ------------------------------
    P, counts = build_transition_matrix(df)
    print("Transition Counts:\n", counts)
    print("\nTransition Matrix P:\n", P)

    # ------------------------------
    # 3. Compute steady-state distribution
    # ------------------------------
    pi = compute_steady_state(P)
    print("\nSteady-State Distribution π (Right, Left):\n", pi)

    # ------------------------------
    # 4. Expected payoff under simple keeper strategies
    # ------------------------------
    M = payoff_matrix()

    # Example keeper strategies:
    uniform_q = np.array([0.5, 0.5])
    left_heavy_q = np.array([0.3, 0.7])
    right_heavy_q = np.array([0.7, 0.3])

    print("\nExpected payoff (uniform keeper):", expected_payoff(pi, uniform_q, M))
    print("Expected payoff (left heavy keeper):", expected_payoff(pi, left_heavy_q, M))
    print("Expected payoff (right heavy keeper):", expected_payoff(pi, right_heavy_q, M))

    # ------------------------------
    # 5. Save results
    # ------------------------------
    out = pd.DataFrame({
        "state": ["Right", "Left"],
        "steady_state_prob": pi
    })
    out.to_csv("simple_mdp_summary.csv", index=False)
    print("\nSaved summary to simple_mdp_summary.csv")

if __name__ == "__main__":
    main()