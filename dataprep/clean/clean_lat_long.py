"""
Implement clean_lat_long function
"""
import re
from typing import Union

import dask.dataframe as dd
import numpy as np
import pandas as pd

PATTERN = re.compile(
    r"""
    .*?
      (?P<dir_front>[NSEW])?[ ]*
        (?P<deg>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
        )?(?P<dir_back>[NSEW])?
    (?:\s*[,;/\s]\s*
      (?P<dir_front2>[NSEW])?[ ]*
      (?P<deg2>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
      (?:(?P<min2>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
      (?:(?P<sec2>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back2>[NSEW])?)?
    \s*$
"""
    % {
        "FLOAT": r"\d+(?:\.\d+)?",
        "DEGREE": chr(176),
        "PRIME": chr(8242),
        "DOUBLE_PRIME": chr(8243),
    },
    re.VERBOSE | re.UNICODE,
)


def clean_lat_long(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "dd",
    split: bool = False,
    hor_coord: str = "lat",
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    This function cleans latitdinal and longitudinal coordinates

    Parameters
    ----------
    df
        pandas or Dask DataFrame
    column
        column name
    output_format
        the desired format of the coordinates: decimal degrees ("dd"),
        decimal degrees with hemisphere ("ddh"), degrees minutes ("dm"),
        degrees minutes seconds ("dms")
    split
        if True, split a column containing latitudinal and longitudinal
        coordinates into one column for latitude and one for longitude
    hor_coord
        If a column of decimals, this parameter defines the horizontal axis.
        Can be "lat" for latitude, or "long" for longitude.
    """

    if isinstance(df, dd.DataFrame):
        df = df.apply(
            lambda row: format_lat_long(row, column, output_format, split, hor_coord),
            axis=1,
            meta=df,
        ).compute()
    elif isinstance(df, pd.DataFrame):
        df = df.apply(
            lambda row: format_lat_long(row, column, output_format, split, hor_coord), axis=1
        )

    return df


def format_lat_long(
    row: pd.Series,
    col: str,
    output_format: str = "dd",
    split: bool = False,
    hor_coord: str = "lat",
) -> pd.Series:
    """
    Function to transform a coordinate instance into the
    desired format
    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    mch = re.match(PATTERN, re.sub(r"''", r'"', str(row[col])))
    if mch is None:
        return row
    if mch.group("deg") is None:
        return row
    mins = float(mch.group("min")) if mch.group("min") else 0
    secs = float(mch.group("sec")) if mch.group("sec") else 0
    dds = float(mch.group("deg")) + mins / 60 + secs / 3600
    hem = mch.group("dir_front") or mch.group("dir_back")

    if not mch.group("deg2"):
        if hem is None:
            # infer the hemisphere
            if hor_coord == "lat":
                hem = "N" if dds >= 0 else "S"
            elif hor_coord == "long":
                hem = "E" if dds >= 0 else "W"
        if output_format == "dd":
            fctr = -1 if hem in {"S", "W"} else 1
            row[col] = round(fctr * dds, 4)
        dds = abs(dds)
        if output_format == "ddh":
            row[col] = f"{round(dds, 4)}{chr(176)} {hem}"
        elif output_format == "dm":
            mins = round(60 * (dds - int(dds)), 4)
            mins = int(mins) if mins.is_integer() else mins
            row[col] = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {hem}"
        elif output_format == "dms":
            mins = int(60 * (dds - np.floor(dds)))
            secs = round(3600 * (dds - np.floor(dds)) - 60 * mins, 4)
            secs = int(secs) if secs.is_integer() else secs
            row[col] = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {secs}{chr(8243)} {hem}"
    else:
        mins2 = float(mch.group("min2")) if mch.group("min2") else 0
        secs2 = float(mch.group("sec2")) if mch.group("sec2") else 0
        dds2 = float(mch.group("deg2")) + mins2 / 60 + secs2 / 3600
        hem = mch.group("dir_front") or mch.group("dir_back")
        hem2 = mch.group("dir_front2") or mch.group("dir_back2")

        lat: Union[str, float]
        lon: Union[str, float]
        if hem is None:
            hem = "N" if dds >= 0 else "S"
        if hem2 is None:
            hem2 = "E" if dds2 >= 0 else "W"
        if output_format == "dd":
            fctr = -1 if hem == "S" else 1
            fctr2 = -1 if hem2 == "W" else 1
            lat, lon = round(fctr * dds, 4), round(fctr2 * dds2, 4)
        dds, dds2 = abs(dds), abs(dds2)
        if output_format == "ddh":
            lat = f"{round(dds, 4)}{chr(176)} {hem}"
            lon = f"{round(dds2, 4)}{chr(176)} {hem2}"
        elif output_format == "dm":
            mins = round(60 * (dds - int(dds)), 4)
            mins = int(mins) if mins.is_integer() else mins
            mins2 = round(60 * (dds2 - int(dds2)), 4)
            mins2 = int(mins2) if mins2.is_integer() else mins2
            lat = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {hem}"
            lon = f"{int(dds2)}{chr(176)} {mins2}{chr(8242)} {hem2}"
        elif output_format == "dms":
            mins = int(60 * (dds - np.floor(dds)))
            secs = round(3600 * (dds - np.floor(dds)) - 60 * mins, 4)
            secs = int(secs) if secs.is_integer() else secs
            mins2 = int(60 * (dds2 - np.floor(dds2)))
            secs2 = round(3600 * (dds2 - np.floor(dds2)) - 60 * mins2, 4)
            secs2 = int(secs2) if secs2.is_integer() else secs2
            lat = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {secs}{chr(8243)} {hem}"
            lon = f"{int(dds2)}{chr(176)} {mins2}{chr(8242)} {secs2}{chr(8243)} {hem2}"
        if split:
            row["latitude"], row["longitude"] = lat, lon
        else:
            row[col] = (lat, lon) if output_format == "dd" else f"{lat}, {lon}"
    return row
