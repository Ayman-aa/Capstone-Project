# calculate_stats.py
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load your scored file
df = pd.read_excel("blind_scoring_CLEAN.xlsx")

# Get scores (assuming column H is SCORE_0_3 from one evaluator)
# You'll need two evaluator columns. If you only have one, add a second.

# For now, just get summary stats
summary = df.groupby('model')['SCORE_0_3'].agg(['mean', 'median', 'count'])
print(summary)