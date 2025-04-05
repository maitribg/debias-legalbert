import pandas as pd
import os

# === CONFIG ===
input_files = [
    "results/baseline-bec-pro.tsv",
    "results/gap-debiased-bec-pro.tsv",
    "results/lcd-debiased-bec-pro.tsv"
]
output_csv = "analysis/table7_summary.csv"
attribute_order = ["balanced", "female", "male"]  # enforce row order

# === Helper function to summarize by attribute and gender ===
def summarize_by_profession_gender(input_file):
    df = pd.read_csv(input_file, sep="\t")
    df["Gender"] = df["Gender"].str.lower().str.strip()
    df["Prof_Gender"] = df["Prof_Gender"].str.lower().str.strip()

    summaries = []
    model_name = os.path.basename(input_file).split("-")[0]

    for attr in attribute_order:
        subset = df[df["Prof_Gender"] == attr]
        male_scores = subset[subset["Gender"] == "male"]["AssociationScore"]
        female_scores = subset[subset["Gender"] == "female"]["AssociationScore"]

        summary = {
            "Model": model_name,
            "Attribute": attr,
            "Female_Mean": round(female_scores.mean(), 5),
            "Male_Mean": round(male_scores.mean(), 5),
            "Abs_Gender_Gap": round(abs(female_scores.mean() - male_scores.mean()), 5),
            "Female_STD": round(female_scores.std(), 5),
            "Male_STD": round(male_scores.std(), 5),
            "Count_F": len(female_scores),
            "Count_M": len(male_scores)
        }
        summaries.append(summary)

    return summaries

# === Run and save ===
all_rows = []
for file in input_files:
    all_rows.extend(summarize_by_profession_gender(file))

summary_df = pd.DataFrame(all_rows)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
summary_df.to_csv(output_csv, index=False)

print("✅ Table 7 summary saved to", output_csv)
print(summary_df)
