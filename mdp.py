import pandas as pd
import numpy as np

def load_clean_data(path):
    """
    Loads the cleaned dataset.
    Required columns:
        - prev_dive (0=Right, 1=Left)
        - currdivedir (keeper's current dive direction)
        - action_kicker (0=Right, 1=Left)
    """
    return pd.read_excel(path)


# 1. Keeper Markov Chain 
def build_keeper_transition_matrix(df):
    """
    Builds the 2x2 keeper transition matrix:
        P_G[next dive | previous dive]

    States:
        0 = Keeper dove Right last time
        1 = Keeper dove Left last time
    """

    counts = np.zeros((2,2))

    for _, row in df.iterrows():
        prev_d = int(row["prev_dive"])       # previous dive
        next_d = int(row["currdivedir"])     # current dive
        counts[prev_d, next_d] += 1

    # Normalize rows to probability
    P_G = counts / counts.sum(axis=1, keepdims=True)
    return P_G, counts

def keeper_steady_state(P):
    """
    Steady-state distribution of the keeper's dive behavior.
    π satisfies: πP = π, π_R + π_L = 1.

    Closed form for 2x2 Markov chain:
        π_L = P(R->L) / (P(R->L) + P(L->R))
        π_R = 1 - π_L
    """
    p01 = P[0,1]  # R -> L
    p10 = P[1,0]  # L -> R

    pi_L = p01 / (p01 + p10)
    pi_R = 1 - pi_L

    return np.array([pi_R, pi_L])  # order = [Right, Left]


# 2. Kicker Behavior
def compute_empirical_kicker_strategy(df):
    """
    Computes the stationary distribution of kicker directions.
    This is not modeled as a Markov chain — we simply use
    empirical frequencies.

    Returns:
        p = [P(kick=R), P(kick=L)]
    """
    kR = np.sum(df["action_kicker"] == 0)
    kL = np.sum(df["action_kicker"] == 1)
    total = kR + kL
    return np.array([kR/total, kL/total])

# 3. Payoff Model
def payoff_matrix():
    """
    Payoff matrix from Keeper's perspective:
        1 = Save
        0 = Goal

    If keeper dives correctly -> save (1)
    If keeper guesses wrong -> goal (0)
    """
    return np.array([
        [1, 0],  # kicker R vs keeper R,L
        [0, 1]   # kicker L vs keeper R,L
    ])

def expected_save_probability(pi_keeper, p_kicker, M):
    """
    Expected save probability:
        E = pᵀ M q

    But now:
      - p = kicker distribution
      - q = keeper steady-state distribution
    """

    q = pi_keeper     # keeper dive distribution
    p = p_kicker      # kicker direction distribution

    return p @ M @ q


# 4. Main
def main():

    # 1. Load data
    df = load_clean_data("clean_data.xlsx")

    # 2. Build keeper transition matrix
    P_G, counts_G = build_keeper_transition_matrix(df)
    print("\nKeeper Transition Counts:\n", counts_G)
    print("\nKeeper Transition Matrix P_G:\n", P_G)

    # 3. Keeper steady-state distribution
    pi_G = keeper_steady_state(P_G)
    print("\nKeeper Steady-State π_G (Right, Left):\n", pi_G)

    # 4. Kicker empirical direction distribution
    p_kicker = compute_empirical_kicker_strategy(df)
    print("\nKicker direction distribution p (Right, Left):\n", p_kicker)

    # 5. Payoff matrix
    M = payoff_matrix()

    # 6. Expected save probability
    E_save = expected_save_probability(pi_G, p_kicker, M)
    print("\nExpected save probability (data-driven kicker + keeper):", E_save)

    # 7. Save summary
    out = pd.DataFrame({
        "agent": ["keeper_R", "keeper_L"],
        "steady_state_prob": pi_G
    })
    out.to_csv("keeper_mdp_summary.csv", index=False)
    print("\nSaved summary to keeper_mdp_summary.csv")

if __name__ == "__main__":
    main()