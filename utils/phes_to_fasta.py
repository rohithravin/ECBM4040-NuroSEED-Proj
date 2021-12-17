"""
Utility function to turn phes.tbl into a FASTA file, for input to preprocessing pipeline.
"""

import pandas as pd

data = pd.read_table("./data/phes.tbl")
print(data.columns)
# for row in data.iterrows():
# 	name = row[""]
	# print(row[""])
with open("./data/phes_aa.fa", "w") as f1, open("./data/phes_na.fa", "w") as f2:
	for idx, row in data.iterrows():
		aa = row['feature.aa_sequence']
		na = row['feature.na_sequence']
		fid = row['feature.patric_id']
		print(f">{fid}\n{aa}", file=f1)
		print(f">{fid}\n{na}", file=f2)