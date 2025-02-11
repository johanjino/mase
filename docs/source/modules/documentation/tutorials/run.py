import pandas as pd
import re

# Define the mapping of type numbers to type names
type_map = {
    0: "Integer",
    1: "MinifloatDenorm",
    2: "MinifloatIEEE",
    3: "Log",
    4: "BlockFP",
    6: "BlockLog",
    7: "Binary",
    8: "BinaryScaling",
    9: "Mixed",
}

# Read the text file
with open("data.txt", "r") as file:
    lines = file.readlines()

# Process each line
for line in lines:
    match = re.match(r"(\d+) \[(.*)\]", line.strip())  # Match number and list
    if match:
        trial_type = int(match.group(1))
        accuracies = list(map(float, match.group(2).split(", ")))

        # Create a DataFrame
        df = pd.DataFrame({"Trial": range(1, len(accuracies) + 1), "Accuracy": accuracies})

        # Get the type name from the map
        type_name = type_map.get(trial_type, f"unknown_{trial_type}")

        # Save to CSV
        file_name = f"tut6_{type_name}_search.csv"
        df.to_csv(file_name, index=False)
        print(f"Saved: {file_name}")
