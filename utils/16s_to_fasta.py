"""
Utility function to turn 16s.tbl into a FASTA file, for input to preprocessing pipeline.
"""

import pandas as pd

data = pd.read_table("./data/16s.tbl")
print(data.columns)
# for row in data.iterrows():
# 	name = row[""]
	# print(row[""])
with open("./data/16s_na.fa", "w") as f:
	for idx, row in data.iterrows():
		na = row['feature.na_sequence']
		fid = row['feature.patric_id']
		print(f">{fid}\n{na}", file=f)