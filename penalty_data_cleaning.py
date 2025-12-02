import pandas as pd

def clean_penalty_data(df):
    """
    Clean a penalty shootout dataset that uses a sliding-window structure
    with columns:
        shootout, kicknum, currdivedir,
        kick-1, kick-2, kick-3,
        dive-1, dive-2, dive-3

    This function simply verifies and preserves those windows,
    and outputs a consistent format grouped by shootout.
    """

    required_cols = [
        "shootout", "kicknum",
        "kick-1", "kick-2", "kick-3",
        "dive-1", "dive-2", "dive-3"
    ]

    # Ensure required columns exist
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}\n"
                f"Columns found: {df.columns.tolist()}"
            )
        
    cleaned_rows = []

    # Group by shootout to keep sequences separate
    grouped = df.groupby("shootout")

    for shootout_id, group in grouped:

        # Sort by kick number
        group = group.sort_values("kicknum")

        for i in range(len(group)):
            row = group.iloc[i]

            cleaned_rows.append({
                "shootout": shootout_id,
                "kicknum": row["kicknum"],
                "kick-1": row["kick-1"],
                "kick-2": row["kick-2"],
                "kick-3": row["kick-3"],
                "dive-1": row["dive-1"],
                "dive-2": row["dive-2"],
                "dive-3": row["dive-3"]
            })

    return pd.DataFrame(cleaned_rows)


# ---------------------------------------------------------------------
#                       MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("Loading local penalty kick XLSX file...")

    # Change filename here as needed
    input_file = "raw_penalty_data.xlsx"
    output_file = "cleaned_penalry_data.xlsx"

    try:
        df = pd.read_excel(input_file)
        print(f"Loaded: {input_file}")
    except Exception as e:
        print(f"ERROR: Could not load {input_file}")
        print(e)
        exit(1)

    print("Cleaning dataset...")
    cleaned = clean_penalty_data(df)

    print("Saving cleaned dataset...")
    cleaned.to_excel(output_file, index=False)

    print("\n--------------------------------------")
    print("Cleaning complete!")
    print(f"Output file saved as: {output_file}")
    print("--------------------------------------\n")
