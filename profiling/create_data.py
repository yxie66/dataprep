##
import pandas as pd

df = pd.read_csv("../notebooks/titanic/train.csv")
df.to_parquet("titanic.pq")
