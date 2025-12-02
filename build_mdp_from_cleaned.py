#!/usr/bin/env python3
"""
build_mdp_from_cleaned.py

Input: cleaned penalty data with columns:
    shootout, kicknum,
    kick-1, kick-2, kick-3,
    dive-1, dive-2, dive-3
(Optional: may include currdivedir or currkick; if not, they are inferred)

Output files (in working directory):
 - mdp_events.csv        : one row per observed event with state, actions, reward, next_state
 - mdp_transitions.json  : transition probabilities P(next_state | state, a_k, a_g)
 - mdp_states.csv        : list of states and their ids
 - mdp_setup.json        : small meta file with actions/states

Run:
    python build_mdp_from_cleaned.py --input cleaned_penalty_data.xlsx
"""
import argparse
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# ---------------------------
# Helper utilities
# ---------------------------
def token_from_row(row):
    """Create a compact state token from kick-1..kick-3 and dive-1..dive-3.
       Uses 'L' for 1 and 'R' for 0; uses 'X' for missing.
    """
    def map_val(v):
        if pd.isna(v):
            return 'X'
        if int(v) == 1:
            return 'L'
        if int(v) == 0:
            return 'R'
        return 'X'
    k1 = map_val(row.get('kick-1'))
    k2 = map_val(row.get('kick-2'))
    k3 = map_val(row.get('kick-3'))
    d1 = map_val(row.get('dive-1'))
    d2 = map_val(row.get('dive-2'))
    d3 = map_val(row.get('dive-3'))
    return f"{k1}{k2}{k3}_{d1}{d2}{d3}"

def infer_current_actions(group):
    """
    For a group (sorted by kicknum) infer current kicker and keeper actions for each row.
    Assumes sliding-window structure:
      - In row with kicknum k, kick-1 represents kick k-1.
      - Therefore kick k (the current kick action) appears as kick-1 in the next row (kicknum k+1).
    Returns series arrays: action_kicker, action_keeper (strings 'L'/'R' or None for unavailable).
    """
    n = len(group)
    # prepare arrays
    kicker = [None] * n
    keeper = [None] * n

    # If group already contains 'currkick' or 'currdivedir', prefer them
    if 'currkick' in group.columns:
        kicker = group['currkick'].tolist()
    if 'currdivedir' in group.columns:
        keeper = group['currdivedir'].tolist()

    # If not fully specified, try to infer for each row from the next row's kick-1 / dive-1
    # (i -> next row index i+1)
    for i in range(n):
        if kicker[i] is None or (isinstance(kicker[i], float) and np.isnan(kicker[i])):
            if i + 1 < n:
                val = group.iloc[i+1].get('kick-1', None)
                if not pd.isna(val):
                    kicker[i] = 'L' if int(val) == 1 else 'R'
        else:
            kicker[i] = 'L' if int(kicker[i]) == 1 else 'R'

        if keeper[i] is None or (isinstance(keeper[i], float) and np.isnan(keeper[i])):
            if i + 1 < n:
                val = group.iloc[i+1].get('dive-1', None)
                if not pd.isna(val):
                    keeper[i] = 'L' if int(val) == 1 else 'R'
        else:
            keeper[i] = 'L' if int(keeper[i]) == 1 else 'R'

    # Last row in a shootout cannot be inferred (no next row) -> keep None
    return kicker, keeper

def build_next_state_token(curr_kick_char, curr_dive_char, curr_state_token):
    """
    Given current state token 'K1K2K3_D1D2D3' and current actions (chars 'L'/'R'),
    produce next state token by shifting in the current actions:
      next_k1 = curr_kick_char, next_k2 = K1, next_k3 = K2
      same for dives.
    If token contains 'X' placeholders, they shift too.
    """
    if pd.isna(curr_state_token):
        return None
    try:
        kpart, dpart = curr_state_token.split("_")
        K1, K2, K3 = list(kpart)
        D1, D2, D3 = list(dpart)
    except Exception:
        # malformed token -> return None
        return None
    next_k = f"{curr_kick_char}{K1}{K2}"
    next_d = f"{curr_dive_char}{D1}{D2}"
    return f"{next_k}_{next_d}"

# ---------------------------
# Main processing function
# ---------------------------
def build_mdp_events(df):
    """
    Input df must have columns: shootout,kicknum,kick-1,kick-2,kick-3,dive-1,dive-2,dive-3
    Optionally may include outcome/result/goal column.
    Returns DataFrame of events and dictionary of transition probabilities.
    """
    required = ['shootout','kicknum','kick-1','kick-2','kick-3','dive-1','dive-2','dive-3']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column missing: {c}")

    # normalize column types
    df = df.copy()
    # Ensure ordering
    df = df.sort_values(['shootout','kicknum']).reset_index(drop=True)

    events = []
    # iterate per shootout
    for sid, group in df.groupby('shootout', sort=False):
        group = group.sort_values('kicknum').reset_index(drop=True)
        # infer current actions
        kicker_actions, keeper_actions = infer_current_actions(group)

        for i, row in group.iterrows():
            state_token = token_from_row(row)
            curr_kick = kicker_actions[i]  # 'L' or 'R' or None
            curr_dive = keeper_actions[i]
            # compute next state token deterministically
            next_state = None
            if curr_kick is not None and curr_dive is not None:
                next_state = build_next_state_token(curr_kick, curr_dive, state_token)

            # reward handling: detect common outcome columns
            reward = np.nan
            # prefer columns named 'outcome','result','goal','scored'
            for col in ['outcome','result','goal','scored']:
                if col in df.columns:
                    val = row.get(col)
                    if pd.isna(val):
                        reward = np.nan
                    else:
                        # flexible mapping: 'goal'/'scored'/1 -> 1 ; 'save'/'miss'/0 -> 0
                        sval = str(val).lower()
                        if sval in ['1','1.0','goal','scored','g','yes','y','true','t']:
                            reward = 1
                        else:
                            reward = 0
                    break

            events.append({
                'shootout': sid,
                'kicknum': int(row['kicknum']),
                'state': state_token,
                'action_kicker': curr_kick,
                'action_keeper': curr_dive,
                'reward': reward,
                'next_state': next_state
            })

    events_df = pd.DataFrame(events)

    # Build empirical transition probabilities P(next_state | state, a_k, a_g)
    # Only count rows where next_state is not None
    trans_counts = defaultdict(lambda: defaultdict(int))
    for _, r in events_df.dropna(subset=['next_state']).iterrows():
        key = (r['state'], r['action_kicker'], r['action_keeper'])
        trans_counts[key][r['next_state']] += 1

    trans_probs = {}
    for key, dests in trans_counts.items():
        total = sum(dests.values())
        trans_probs_key = {snext: cnt/total for snext, cnt in dests.items()}
        trans_probs[str(key)] = trans_probs_key

    return events_df, trans_probs

# ---------------------------
# CLI / Run
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='cleaned_penalty_data.xlsx',
                   help='Input cleaned penalty data (xlsx or csv).')
    p.add_argument('--sheet', '-s', default=0, help='If xlsx, sheet name/index.')
    p.add_argument('--output-prefix', '-o', default='mdp_', help='Output prefix for files.')
    args = p.parse_args()

    # Load data (xlsx or csv)
    if args.input.lower().endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input, sheet_name=args.sheet)

    print(f"Loaded {len(df)} rows from {args.input}")

    events_df, trans_probs = build_mdp_events(df)

    # Save events
    events_file = args.output_prefix + 'events.csv'
    events_df.to_csv(events_file, index=False)
    print(f"Saved events to {events_file} ({len(events_df)} rows)")

    # Save list of states (unique)
    states = sorted(events_df['state'].dropna().unique())
    states_df = pd.DataFrame({'state': states})
    states_file = args.output_prefix + 'states.csv'
    states_df.to_csv(states_file, index=False)
    print(f"Saved {len(states)} states to {states_file}")

    # Save transition probs JSON
    trans_file = args.output_prefix + 'transitions.json'
    with open(trans_file, 'w') as f:
        json.dump(trans_probs, f, indent=2)
    print(f"Saved transition probabilities to {trans_file}")

    # Save meta/setup JSON
    actions = {'kicker': ['L','R'], 'keeper': ['L','R']}
    setup = {
        'n_events': len(events_df),
        'n_states': len(states),
        'actions': actions,
        'events_file': events_file,
        'states_file': states_file,
        'transitions_file': trans_file
    }
    setup_file = args.output_prefix + 'setup.json'
    with open(setup_file, 'w') as f:
        json.dump(setup, f, indent=2)
    print(f"Saved setup summary to {setup_file}")

    # Short diagnostics
    n_missing_actions = events_df['action_kicker'].isna().sum() + events_df['action_keeper'].isna().sum()
    if n_missing_actions > 0:
        print("WARNING: Some events had missing inferred current actions (likely last kick of shootouts).")
        print("You may drop those rows before further analysis, or fill with assumed values.")
    if events_df['reward'].isna().all():
        print("WARNING: No reward/outcome column detected. Reward values are all NaN.")
        print("If you have outcome info, add a column named 'outcome' (goal/save) or 'result' etc.")

if __name__ == '__main__':
    main()
