from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

data = gene_type_counter

rename = {
    "protein_coding": "Protein coding",
    "lincRNA": "Long intergenic non-coding RNA",
    "pseudogene": "Pseudogene",
    "antisense": "Antisense",
    "processed_transcript": "Processed transcript",
    "processed_pseudogene": "Processed pseudogene",
    "unprocessed_pseudogene": "Unprocessed pseudogene",
    "Other": "Other",
    "misc_RNA": "Miscellaneous RNA",
    "snRNA": "Small nuclear RNA",
    "transcribed_unprocessed_pseudogene": "Transcribed unprocessed pseudogene",
    "transcribed_processed_pseudogene": "Transcribed processed pseudogene",
    "sense_intronic": "Sense intronic",
    "snoRNA": "Small nucleolar RNA",
    "rRNA": "Ribosomal RNA",
    "sense_overlapping": "Sense overlapping transcript",
}


df = pd.DataFrame(data.items(), columns=["Type", "Count"])

# Sort by count
df = df.sort_values(by="Count", ascending=False)

# Aggregate small categories into "Other" for clarity
threshold = 500
df["Type"] = df.apply(
    lambda row: row["Type"] if row["Count"] >= threshold else "Other", axis=1
)
# remove type == "TEC"
df = df[df["Type"] != "TEC"]

df_agg = (
    df.groupby("Type")["Count"]
    .sum()
    .reset_index()
    .sort_values(by="Count", ascending=True)
)

df_agg["Type"] = df_agg["Type"].apply(lambda x: rename.get(x, x))

# Plot
# plt.figure(figsize=(4, 2.25))
plt.figure(figsize=(2, 1))
plt.barh(df_agg["Type"], df_agg["Count"], color="skyblue")
plt.xlabel("Count")
plt.title("Gene type annotation")
plt.tight_layout()
plt.savefig("enhancers_gene_count.png")
plt.clf()
plt.close()
