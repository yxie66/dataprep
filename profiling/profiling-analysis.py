##
import pandas as pd
import altair as alt

##
rawdf = pd.read_json("profiling-result.json", lines=True, orient="records")

##
rawdf["ElapsedPlot"] = rawdf.Elapsed.apply(
    lambda d: d["plot"] if isinstance(d, dict) else 0
)
rawdf["ElapsedPlotCorrelation"] = rawdf.Elapsed.apply(
    lambda d: d["plot"] if isinstance(d, dict) else 0
)
rawdf["ElapsedPlotMissing"] = rawdf.Elapsed.apply(
    lambda d: d["plot"] if isinstance(d, dict) else 0
)
rawdf["ElapsedTot"] = rawdf["Elapsed"]
rawdf["ElapsedTot"] = rawdf.Elapsed.apply(
    lambda d: sum(d.values()) if isinstance(d, dict) else d
)
##
alt.Chart(rawdf).transform_filter(alt.datum.CPU == 4).mark_bar().encode(
    x="Library:N", y="ElapsedTot", color="Library", column="Size:O", row="Mem"
).resolve_scale(x="independent").properties(title="Dataprep vs Pandas-Profiling")

##
