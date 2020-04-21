"""Profiling

Usage:
  profiling --data=<dataset> --num=<num> --mem=<mem> --cpu=<cpu> (pandas|dataprep)

Options:
  -h --help    Show this screen.
"""
import logging
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from itertools import product

import numpy as np
import pandas as pd
from contexttimer import Timer
from docopt import docopt


logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[{asctime} {levelname}] {name}: {message}", style="{")
ch.setFormatter(formatter)
logger.addHandler(ch)


def main() -> None:
    args = docopt(__doc__)
    print(args)

    dataset = args["--data"]
    N = int(args["--num"])

    df = create_dataset(dataset, N)
    data_in_mem = df.memory_usage(deep=True).sum()

    if args["pandas"]:
        from pandas_profiling import ProfileReport

        with TemporaryDirectory() as tdir:
            with Timer() as timer:
                profile = ProfileReport(
                    df,
                    title="Pandas Profiling Report",
                    config_file=f"{Path(__file__).parent}/pandas-profiling-config.yaml",
                )
                profile.to_file(output_file=f"{Path(__file__).parent}/a_report.html")

        print(f"Pandas Profiling Elapsed: {timer.elapsed}s")
        print(
            {
                "Mem": args["--mem"],
                "CPU": args["--cpu"],
                "Library": "pandas-profiling",
                "Dataset": dataset,
                "Size": N,
                "MSize": data_in_mem,
                "Elapsed": timer.elapsed,
            }
        )
    else:
        from dataprep.eda import plot, plot_correlation, plot_missing

        with TemporaryDirectory() as tdir:
            logger.info(f"Computing plot")
            then = time()
            plot(df).save(f"{tdir}/report1.html")
            plotdf_t = time() - then

            # then = time()
            # for x, y in list(product(df.columns, df.columns)):
            #     if x != y:
            #         logger.info(f"Computing plotxy {x} {y}")
            #         plot(df, x=x, y=y).save(f"{tdir}/report1.html")
            # plotdfxy_t = time() - then

            then = time()
            logger.info(f"Computing plot_correlation")
            plot_correlation(df).save(f"{tdir}/report2.html")
            plotcorr_t = time() - then

            then = time()
            logger.info(f"Computing plot_missing")
            plot_missing(df, bins=100).save(f"{tdir}/report3.html")
            plotmissing_t = time() - then

            elapsed = plotdf_t + plotcorr_t + plotmissing_t

            print(
                f"Dataprep Elapsed: {elapsed}s, breaking down: {plotdf_t}, {plotcorr_t}, {plotmissing_t}"
            )

            print(
                {
                    "Mem": args["--mem"],
                    "CPU": args["--cpu"],
                    "Library": "dataprep",
                    "Dataset": dataset,
                    "Size": N,
                    "MSize": data_in_mem,
                    "Elapsed": {
                        "plot": plotdf_t,
                        "plot_correlation": plotcorr_t,
                        "plot_missing": plotmissing_t,
                    },
                }
            )


def create_dataset(dataset: str, n: int) -> pd.DataFrame:

    folder = Path(__file__).parent
    df = pd.read_parquet(f"{folder/dataset}.pq")

    logger.info(f"Original DataFrame shape: {df.shape}")

    rep = int(np.ceil(n / len(df)))

    df = pd.concat([df] * rep)

    df = df.sample(frac=1).reset_index(drop=True)[:n]

    logger.info(f"new DataFrame shape: {df.shape}")

    return df


if __name__ == "__main__":
    main()
