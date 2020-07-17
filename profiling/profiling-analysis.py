##
import pandas as pd
import altair as alt
from json import loads

##
with open("profiling-result-missing.json") as f:
    lines = f.readlines()
rs = [r for line in lines for r in loads(line)]
df = pd.DataFrame(rs)


##
pdf = df.pivot_table(
    index=["Mem", "CPU", "Dataset", "Partition", "Row", "Col", "Mode"],
    columns="Func",
    values="Elapsed",
).reset_index()

##
alt.Chart(
    df[df.Mem == "2G"], title="Plot Missing Comparason: 2G Mem/8 CPU/16 Data Partition"
).mark_bar().encode(
    y="Func:N",
    x=alt.X("Elapsed", title="Elapsed (s)"),
    color="Func",
    tooltip="Elapsed",
    row="Row:Q",
    column=alt.Column("Mode:O", title="Data Loading Mode"),
).resolve_scale(
    x="independent"
)


##
