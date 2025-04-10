from pathlib import Path
import pandas as pd

root = Path("path to aggregator_metric")
result = list(root.glob("*.parquet"))[0]
df = pd.read_parquet(result)
outcsv_dir = "result.csv"
df.to_csv(outcsv_dir, index=False)
df_scenario = df["scenario"]
df_log_name = df["log_name"]
df_score = df["score"]



