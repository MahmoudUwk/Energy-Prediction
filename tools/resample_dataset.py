"""Utilities for resampling raw 1Hz power data into aggregated CSV files.

This script converts the raw measurements stored in ``dataset/1Hz`` into
down-sampled datasets (for example ``1T`` or ``5T``) under
``dataset/resampled data``.  It is a cleaned-up version of the historic
resampling code that shipped with the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT_DIR = Path("dataset") / "1Hz"
DEFAULT_OUTPUT_DIR = Path("dataset") / "resampled data"
DEFAULT_INTERVALS = ("1s",)


@dataclass(frozen=True)
class ResampleJob:
    interval: str
    aggregation: str

    def output_stub(self) -> str:
        return f"{self.interval}.csv"


def _discover_csv_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    csv_files = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    return csv_files


def _read_1hz_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"File {path} missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


def _load_concatenated_frame(files: Iterable[Path]) -> pd.DataFrame:
    frames = [_read_1hz_csv(path) for path in files]
    combined = pd.concat(frames, axis=0).sort_index()
    return combined


def _resample_frame(df: pd.DataFrame, job: ResampleJob) -> pd.DataFrame:
    if job.aggregation == "max":
        resampled = df.resample(job.interval).max()
    elif job.aggregation == "mean":
        resampled = df.resample(job.interval).mean()
    else:
        raise ValueError(f"Unsupported aggregation '{job.aggregation}'")

    resampled.dropna(inplace=True)
    return resampled


def _write_frame(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination)


def resample_dataset(
    input_dir: Path,
    output_dir: Path,
    jobs: Iterable[ResampleJob],
    *,
    export_raw: bool = False,
    raw_filename: str = "1s.csv",
    subset_rows: int | None = None,
    subset_filename: str = "1s_more.csv",
) -> None:
    csv_files = _discover_csv_files(input_dir)
    base_frame = _load_concatenated_frame(csv_files)

    output_dir.mkdir(parents=True, exist_ok=True)

    if export_raw:
        raw_path = output_dir / raw_filename
        _write_frame(base_frame.copy(), raw_path)
        print(f"✔ Saved concatenated 1Hz data to {raw_path}")

    if subset_rows is not None and subset_rows > 0:
        subset = base_frame.iloc[:subset_rows].copy()
        subset_path = output_dir / subset_filename
        _write_frame(subset, subset_path)
        print(
            f"✔ Saved first {subset_rows} rows to {subset_path}"
        )

    for job in jobs:
        resampled = _resample_frame(base_frame, job)
        output_file = output_dir / job.output_stub()
        _write_frame(resampled, output_file)
        print(f"✔ Saved {job.interval} ({job.aggregation}) to {output_file}")


def _parse_jobs(intervals: Iterable[str], include_max: bool) -> list[ResampleJob]:
    jobs: list[ResampleJob] = []
    for interval in intervals:
        jobs.append(ResampleJob(interval=interval, aggregation="mean"))
    return jobs


CONFIG = {
    "input_dir": DEFAULT_INPUT_DIR,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "intervals": list(DEFAULT_INTERVALS),
    "include_max": False,
    "export_raw": True,
    "raw_filename": "1s.csv",
    "subset_rows": None,
    "subset_filename": "1s_more.csv",
}


def main() -> None:
    jobs = _parse_jobs(CONFIG["intervals"], include_max=CONFIG["include_max"])
    resample_dataset(
        input_dir=Path(CONFIG["input_dir"]),
        output_dir=Path(CONFIG["output_dir"]),
        jobs=jobs,
        export_raw=CONFIG["export_raw"],
        raw_filename=CONFIG["raw_filename"],
        subset_rows=CONFIG["subset_rows"],
        subset_filename=CONFIG["subset_filename"],
    )


if __name__ == "__main__":
    main()

