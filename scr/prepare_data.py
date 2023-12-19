import shutil
import numpy as np
import polars as pl
from tqdm import tqdm
from pathlib import Path

# Constants and schema definition
SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}

FEATURE_NAMES = [
    "anglez",
    "enmo",
    "step",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
    "anglez_sin",
    "anglez_cos",
]

# Constants for normalization
ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


# Function Definitions
def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    """
    Converts a given column into sine and cosine components.
    Args:
    - x (pl.Expr): Column to be converted.
    - max_ (int): The maximum value for normalization.
    - name (str): The base name for the new columns.

    Returns:
    - List[pl.Expr]: Two columns representing sine and cosine components.
    """
    rad = 2 * np.pi * (x % max_) / max_
    return [rad.sin().alias(f"{name}_sin"), rad.cos().alias(f"{name}_cos")]


def deg_to_rad(x: pl.Expr) -> pl.Expr:
    """
    Converts degrees to radians.
    Args:
    - x (pl.Expr): Column in degrees.

    Returns:
    - pl.Expr: Column in radians.
    """
    return np.pi / 180 * x


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds engineered features to the dataframe.
    Args:
    - series_df (pl.DataFrame): Original DataFrame.

    Returns:
    - pl.DataFrame: DataFrame with added features.
    """
    return (
        series_df
        .with_row_count("step")
        .with_columns(
            *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
            *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
            *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
            pl.col("step") / pl.count("step"),
            pl.col('anglez_rad').sin().alias('anglez_sin'),
            pl.col('anglez_rad').cos().alias('anglez_cos'),
        )
        .select("series_id", *FEATURE_NAMES)
    )


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    """
    Saves each series in the DataFrame to a separate Numpy file.
    Args:
    - this_series_df (pl.DataFrame): DataFrame to be saved.
    - columns (list[str]): List of column names to be saved.
    - output_dir (Path): Directory where files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


# Main processing function
def process_data(phases, data_dir, processed_dir):
    """
    Processes the data by reading, transforming, and saving it in a structured format for multiple phases.

    Args:
    - phases (list of str): The phases of data processing ('train', 'test', 'dev').
    - data_dir (str): The directory containing the raw data files.
    - processed_dir (str): The directory where processed data will be saved.
    """
    for phase in phases:
        current_processed_dir = Path(processed_dir) / phase

        # Remove existing processed data directory for the current phase
        if current_processed_dir.exists():
            shutil.rmtree(current_processed_dir)
            print(f"Removed {phase} dir: {current_processed_dir}")

        # Load data depending on phase
        file_name = f"{phase}_series.parquet" if phase != "dev" else "dev_series.parquet"
        series_lf = pl.scan_parquet(Path(data_dir) / file_name, low_memory=True)

        # Preprocess data
        series_df = series_lf.with_columns(
            pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
            deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
            (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
            (pl.col("enmo") - ENMO_MEAN) / ENMO_STD
        ).select([
            pl.col("series_id"), pl.col("anglez"), pl.col("enmo"),
            pl.col("timestamp"), pl.col("anglez_rad")
        ]).collect(streaming=True).sort(by=["series_id", "timestamp"])

        # Process each series individually
        n_unique = series_df.get_column("series_id").n_unique()
        for series_id, this_series_df in tqdm(series_df.groupby("series_id"), total=n_unique):
            enhanced_series_df = add_feature(this_series_df)
            series_dir = current_processed_dir / series_id
            save_each_series(enhanced_series_df, FEATURE_NAMES, series_dir)


# Configuration
config = {
    'data_dir': '../data/',
    'processed_dir': '../data/processed/',
    'phases': ['train', 'test']  # List of phases to process
}

# Main Execution
if __name__ == "__main__":
    process_data(config['phases'], config['data_dir'], config['processed_dir'])
