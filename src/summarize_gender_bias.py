# import pandas as pd

# # Load your BEC-Pro results
# df = pd.read_csv("results/lcd-debiased-bec-pro.tsv", sep="\t")

# # Mark first 1763 rows as male, the rest as female (based on observation)
# df["gender"] = ["male" if i < 1763 else "female" for i in range(len(df))]

# # Same analysis
# female_mean = df[df["gender"] == "female"]["AssociationScore"].mean()
# male_mean = df[df["gender"] == "male"]["AssociationScore"].mean()
# gap = abs(female_mean - male_mean)

# print("\n📊 Gender Bias Summary (GAP – BEC-Cri)")
# print(f"→ Female Mean Score: {female_mean:.5f}")
# print(f"→ Male Mean Score:   {male_mean:.5f}")
# print(f"→ |Female - Male|:   {gap:.5f}")

import pandas as pd

df = pd.read_csv("results/gap-debiased-bec-pro.tsv", sep="\t")

# Define male and female indicator words
male_words = {"he", "man", "father", "husband", "boyfriend", "uncle", "son", "brother", "dad"}
female_words = {"she", "woman", "mother", "wife", "girlfriend", "aunt", "daughter", "sister", "mom"}

# Detect gender from the original sentence
def detect_gender(sent):
    sent_lower = sent.lower()
    for word in male_words:
        if word in sent_lower:
            return "male"
    for word in female_words:
        if word in sent_lower:
            return "female"
    return "unknown"

# Apply gender detection
df["gender"] = df["Sentence"].apply(detect_gender)

# Compute results
female_mean = df[df["gender"] == "female"]["AssociationScore"].mean()
male_mean = df[df["gender"] == "male"]["AssociationScore"].mean()
gap = abs(female_mean - male_mean)

print("\n📊 Gender Bias Summary (GAP – BEC-Pro)")
print(f"→ Female Mean Score: {female_mean:.5f}")
print(f"→ Male Mean Score:   {male_mean:.5f}")
print(f"→ |Female - Male|:   {gap:.5f}")
