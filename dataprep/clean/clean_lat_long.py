"""
Implement clean_lat_long function
"""
# pylint: disable=too-many-boolean-expressions
import re
from typing import Any, Tuple, Union

import dask.dataframe as dd
import dask
import numpy as np
import pandas as pd

from ..eda.utils import to_dask

PATTERN = re.compile(
    r"""
    .*?[(]?
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
    [)]?\s*$
"""
    % {
        "FLOAT": r"\d+(?:\.\d+)?",
        "DEGREE": chr(176),
        "PRIME": chr(8242),
        "DOUBLE_PRIME": chr(8243),
    },
    re.VERBOSE | re.UNICODE,
)

NULL_VALUES = {
    np.nan,
    float("NaN"),
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "",
}
STATS = {"cleaned": 0, "null": 0, "unknown": 0}


def clean_lat_long(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "dd",
    split: bool = False,
    inplace: bool = False,
    hor_coord: str = "lat",
) -> pd.DataFrame:
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
        coordinates into one column for latitude and one column for longitude
    inplace
        If True, delete the given column with dirty data, else, create a new
        column with cleaned data.
    hor_coord
        If a column of decimals, this parameter defines the horizontal axis.
        Can be "lat" for latitude, or "long" for longitude.
    """
    # pylint: disable=too-many-arguments

    reset_stats()

    df = to_dask(df)
    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()
    if split:
        if output_format == "dd":
            meta.update(zip(("latitude", "longitude"), (float, float)))
        else:
            meta.update(zip(("latitude", "longitude"), (str, str)))
    else:
        meta[f"{column}_clean"] = float if output_format == "dd" else str

    df = df.apply(
        format_lat_long,
        args=(column, output_format, split, hor_coord),
        axis=1,
        meta=meta,
    )

    if inplace:
        df = df.drop(columns=[column])

    df, nrows = dask.compute(df, df.shape[0])

    report(nrows)

    return df


def validate_lat_long(
    x: Union[str, float, pd.Series, Tuple[float, float]], hor_coord: str = "lat"
) -> Union[bool, pd.Series]:
    """
    This function validates latitdinal and longitudinal coordinates

    Parameters
    ----------
    x
        pandas Series of coordinates or str/float coordinate
    hor_coord
        If a column of decimals, this parameter defines the horizontal axis.
        Can be "lat" for latitude, or "long" for longitude.
    """

    if isinstance(x, pd.Series):
        return x.apply(check_lat_long, args=(hor_coord,))
    else:
        return check_lat_long(x, hor_coord)


def format_lat_long(
    row: pd.Series, col: str, output_format: str, split: bool, hor_coord: str,
) -> pd.Series:
    """
    Function to transform a coordinate instance into the
    desired format
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-return-statements
    if row[col] in NULL_VALUES:
        return not_lat_long(row, col, split, "null")

    mch = re.match(PATTERN, re.sub(r"''", r'"', str(row[col])))
    if not mch:
        return not_lat_long(row, col, split, "unknown")
    if not mch.group("deg"):
        return not_lat_long(row, col, split, "unknown")
    mins = float(mch.group("min")) if mch.group("min") else 0
    secs = float(mch.group("sec")) if mch.group("sec") else 0
    dds = float(mch.group("deg")) + mins / 60 + secs / 3600
    hem = mch.group("dir_back") or mch.group("dir_front")

    # minutes and seconds need to be in the interval [0, 60)
    # hemishpere must be "N", "S", "E", or "W"
    # if a hemisphere is given, degrees must not be negative
    # hemisphere should not be given before and after the coordinate
    if (
        not 0 <= mins < 60
        or not 0 <= secs < 60
        or hem
        and (hem not in {"N", "S", "E", "W"} or float(mch.group("deg")) < 0)
        or mch.group("dir_back")
        and mch.group("dir_front")
    ):
        return not_lat_long(row, col, split, "unknown")

    if not mch.group("deg2"):
        if not hem:
            # infer the hemisphere
            if hor_coord == "lat":
                hem = "N" if dds >= 0 else "S"
            elif hor_coord == "long":
                hem = "E" if dds >= 0 else "W"
        dds = abs(dds)
        # latitude must be in [-90, 90] and longitude must be in [-180, 180]
        if hor_coord == "lat":
            if dds > 90 or hem not in {"N", "S"}:
                return not_lat_long(row, col, split, "unknown")
        elif hor_coord == "long":
            if dds > 180 or hem not in {"E", "W"}:
                return not_lat_long(row, col, split, "unknown")

        if output_format == "dd":
            fctr = -1 if hem in {"S", "W"} else 1
            row[f"{col}_clean"] = round(fctr * dds, 4)
        if output_format == "ddh":
            row[f"{col}_clean"] = f"{round(dds, 4)}{chr(176)} {hem}"
        elif output_format == "dm":
            mins = round(60 * (dds - int(dds)), 4)
            mins = int(mins) if mins.is_integer() else mins
            row[f"{col}_clean"] = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {hem}"
        elif output_format == "dms":
            mins = int(60 * (dds - int(dds)))
            secs = round(3600 * (dds - int(dds)) - 60 * mins, 4)
            secs = int(secs) if secs.is_integer() else secs
            row[
                f"{col}_clean"
            ] = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {secs}{chr(8243)} {hem}"

        if row[col] != row[f"{col}_clean"]:
            STATS["cleaned"] += 1

    else:
        mins2 = float(mch.group("min2")) if mch.group("min2") else 0
        secs2 = float(mch.group("sec2")) if mch.group("sec2") else 0
        dds2 = float(mch.group("deg2")) + mins2 / 60 + secs2 / 3600
        hem2 = mch.group("dir_back2") or mch.group("dir_front2")

        # minutes and seconds must be in the interval [0, 60)
        # the first coordinate's hemisphere must be "N" or "S"
        # the second cooridnates hemisphere must be either "E" or "W"
        # and if it's given, the degrees must not be negative
        # latitude (dds) must be in the interval [-90, 90]
        # longitude (dds2) must be in the interval [-180, 180]
        # hemisphere should not be given before and after the coordinate
        if (
            not 0 <= mins2 < 60
            or not 0 <= secs2 < 60
            or hem
            and hem not in {"N", "S"}
            or hem2
            and (hem2 not in {"E", "W"} or float(mch.group("deg2")) < 0)
            or not -90 <= dds <= 90
            or not -180 <= dds2 <= 180
            or mch.group("dir_back2")
            and mch.group("dir_front2")
        ):
            return not_lat_long(row, col, split, "unknown")

        lat: Union[str, float]
        lon: Union[str, float]
        if not hem:
            hem = "N" if dds >= 0 else "S"
        if not hem2:
            hem2 = "E" if dds2 >= 0 else "W"
        dds, dds2 = abs(dds), abs(dds2)
        if dds > 90 or dds2 > 180:
            return not_lat_long(row, col, split, "unknown")
        if output_format == "dd":
            fctr = -1 if hem == "S" else 1
            fctr2 = -1 if hem2 == "W" else 1
            lat, lon = round(fctr * dds, 4), round(fctr2 * dds2, 4)
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
            mins = int(60 * (dds - int(dds)))
            secs = round(3600 * (dds - int(dds)) - 60 * mins, 4)
            secs = int(secs) if secs.is_integer() else secs
            mins2 = int(60 * (dds2 - int(dds2)))
            secs2 = round(3600 * (dds2 - int(dds2)) - 60 * mins2, 4)
            secs2 = int(secs2) if secs2.is_integer() else secs2
            lat = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {secs}{chr(8243)} {hem}"
            lon = f"{int(dds2)}{chr(176)} {mins2}{chr(8242)} {secs2}{chr(8243)} {hem2}"
        if split:
            STATS["cleaned"] += 1
            row["latitude"], row["longitude"] = lat, lon
        else:
            row[f"{col}_clean"] = (
                (lat, lon) if output_format == "dd" else f"{lat}, {lon}"
            )
            if row[col] != row[f"{col}_clean"]:
                STATS["cleaned"] += 1
    return row


def check_lat_long(val: Union[str, float, Any], hor_coord: str) -> bool:
    """
    Function to transform a coordinate instance into the
    desired format
    """
    # pylint: disable=too-many-return-statements, too-many-branches
    if val in NULL_VALUES:
        return False
    mch = re.match(PATTERN, re.sub(r"''", r'"', str(val)))
    if not mch:
        return False
    if not mch.group("deg"):
        return False
    mins = float(mch.group("min")) if mch.group("min") else 0
    secs = float(mch.group("sec")) if mch.group("sec") else 0
    dds = float(mch.group("deg")) + mins / 60 + secs / 3600
    hem = mch.group("dir_back") or mch.group("dir_front")

    # minutes and seconds need to be in the interval [0, 60)
    # hemishpere must be "N", "S", "E", or "W"
    # if a hemisphere is given, degrees must not be negative
    # hemisphere should not be given before and after the coordinate
    if (
        not 0 <= mins < 60
        or not 0 <= secs < 60
        or hem
        and (hem not in {"N", "S", "E", "W"} or float(mch.group("deg")) < 0)
        or mch.group("dir_back")
        and mch.group("dir_front")
    ):
        return False

    if not mch.group("deg2"):
        if not hem:
            # infer the hemisphere
            if hor_coord == "lat":
                hem = "N" if dds >= 0 else "S"
            elif hor_coord == "long":
                hem = "E" if dds >= 0 else "W"
        # latitude must be in [-90, 90] and longitude must be in [-180, 180]
        if hor_coord == "lat":
            if not -90 <= dds <= 90 or hem not in {"N", "S"}:
                return False
        elif hor_coord == "long":
            if not -180 <= dds <= 180 or hem not in {"E", "W"}:
                return False

    else:
        mins2 = float(mch.group("min2")) if mch.group("min2") else 0
        secs2 = float(mch.group("sec2")) if mch.group("sec2") else 0
        dds2 = float(mch.group("deg2")) + mins2 / 60 + secs2 / 3600
        hem2 = mch.group("dir_back2") or mch.group("dir_front2")

        # minutes and seconds must be in the interval [0, 60)
        # the first coordinate's hemisphere must be "N" or "S"
        # the second cooridnates hemisphere must be either "E" or "W"
        # and if it's given, the degrees must not be negative
        # latitude (dds) must be in the interval [-90, 90]
        # longitude (dds2) must be in the interval [-180, 180]
        # hemisphere should not be given before and after the coordinate
        if (
            not 0 <= mins2 < 60
            or not 0 <= secs2 < 60
            or hem
            and hem not in {"N", "S"}
            or hem2
            and (hem2 not in {"E", "W"} or float(mch.group("deg2")) < 0)
            or not -90 <= dds <= 90
            or not -180 <= dds2 <= 180
            or mch.group("dir_back2")
            and mch.group("dir_front2")
        ):
            return False

    return True


def not_lat_long(row: pd.Series, col: str, split: bool, errtype: str) -> pd.Series:
    """
    Return result when value unable to be parsed
    """
    STATS[errtype] += 1
    if split:
        row["latitude"], row["longitude"] = np.nan, np.nan
    else:
        row[f"{col}_clean"] = np.nan
    return row


def reset_stats() -> None:
    """
    Reset global statistics dictionary
    """
    STATS["cleaned"] = 0
    STATS["null"] = 0
    STATS["unknown"] = 0


def report(nrows: int) -> None:
    """
    Describe what was done in the cleaning process
    """
    print("Latitude and Longitude Cleaning Report:")
    if STATS["cleaned"] > 0:
        nclnd = STATS["cleaned"]
        pclnd = round(nclnd / nrows * 100, 2)
        print(f"\t{nclnd} values cleaned ({pclnd}%)")
    if STATS["unknown"] > 0:
        nunknown = STATS["unknown"]
        punknown = round(nunknown / nrows * 100, 2)
        print(f"\t{nunknown} values unable to be parsed ({punknown}%), set to NaN")
    nnull = STATS["null"] + STATS["unknown"]
    pnull = round(nnull / nrows * 100, 2)
    ncorrect = nrows - nnull
    pcorrect = round(100 - pnull, 2)
    print(
        f"""Result contains {ncorrect} ({pcorrect}%) values in the correct format and {nnull} null values ({pnull}%)"""
    )
