import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob


# Find all CSV files matching the pattern
csv_files = glob.glob("tut6_*_search.csv")

# Create a figure
plt.figure(figsize=(12, 6))

# Plot each CSV file
for file in csv_files:
    # Extract the type name from the file name
    type_name = file.split("_")[1]

    # Read CSV
    df = pd.read_csv(file)

    # Compute max accuracy so far
    df["Max Accuracy So Far"] = df["Accuracy"].cummax()

    # Plot using Seaborn
    sns.lineplot(x=df["Trial"], y=df["Max Accuracy So Far"], label=type_name, drawstyle='steps-post')


df = pd.read_csv("tutorial_6_accuracy_per_trial.csv")
baseline_accuracy = max(df["Accuracy"])
plt.axhline(y=baseline_accuracy, color="cornflowerblue", linestyle="dashed", linewidth=2, label="Baseline", alpha=0.75)


# Labels and title
plt.xlabel("Trial")
plt.ylabel("Max Accuracy")
plt.ylim(0.85,0.882)
plt.title("Accuracy Comparison of different precision")
plt.legend(title="Experiment Type", loc=4)
plt.savefig("tut6_full_mixed_precision_search.png")
