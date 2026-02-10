import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw athletes data"""
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    return data


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """Create total_lift target variable"""
    data["total_lift"] = data[["candj", "snatch", "deadlift", "backsq"]].sum(
        axis=1, min_count=4
    )
    print(f"Created target variable 'total_lift'")
    return data


def remove_irrelevant_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed for modeling"""
    # First dropna on required columns
    required_cols = [
        "region", "age", "weight", "height", "howlong", "gender",
        "eat", "train", "background", "experience", "schedule",
        "deadlift", "candj", "snatch", "backsq"
    ]
    
    initial_count = len(data)
    data = data.dropna(subset=required_cols)
    print(f"Removed {initial_count - len(data)} rows with missing required values")
    
    # Drop columns
    cols_to_drop = [
        "affiliate", "team", "name", "athlete_id",
        "fran", "helen", "grace", "filthy50", "fgonebad",
        "run400", "run5k", "pullups", "train"
    ]
    data = data.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns")
    
    return data


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and invalid values"""
    initial_count = len(data)
    
    # Weight outliers
    data = data[data["weight"] < 1500]
    
    # Gender cleanup
    data = data[data["gender"] != "--"]
    
    # Age filter
    data = data[data["age"] >= 18]
    
    # Height filter
    data = data[(data["height"] < 96) & (data["height"] > 48)]
    
    # Lift filters
    data = data[
        ((data["deadlift"] > 0) & (data["deadlift"] <= 1105)) |
        ((data["gender"] == "Female") & (data["deadlift"] <= 636))
    ]
    data = data[(data["candj"] > 0) & (data["candj"] <= 395)]
    data = data[(data["snatch"] > 0) & (data["snatch"] <= 496)]
    data = data[(data["backsq"] > 0) & (data["backsq"] <= 1069)]
    
    print(f"Removed {initial_count - len(data)} outlier rows")
    return data


def clean_survey_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean survey response data"""
    initial_count = len(data)
    
    # Replace decline to answer with NaN
    decline_dict = {"Decline to answer|": np.nan}
    data = data.replace(decline_dict)
    
    # Drop rows with missing survey responses
    data = data.dropna(subset=["background", "experience", "schedule", "howlong", "eat"])
    
    print(f"Removed {initial_count - len(data)} rows with incomplete survey data")
    return data


def preprocess_data(input_path: str, output_path: str) -> None:
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("Starting preprocessing pipeline")
    print("=" * 60)
    
    # Load data
    data = load_raw_data(input_path)
    
    # Create target first (before dropping lift columns)
    data = create_target(data)
    
    # Preprocessing steps
    data = remove_irrelevant_columns(data)
    data = remove_outliers(data)
    data = clean_survey_data(data)

    # Drop individual lifts - they compose the target!
    print("Dropping individual lift columns (prevent data leakage)")
    data = data.drop(columns=["candj", "snatch", "deadlift", "backsq"])
    
    # Save preprocessed data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_file, index=False)
    
    print("=" * 60)
    print(f"Preprocessing complete!")
    print(f"Final dataset: {len(data)} rows, {len(data.columns)} columns")
    print(f"Saved to: {output_path}")
    print("=" * 60)
    
    return data


if __name__ == "__main__":
    # For testing locally
    preprocess_data(
        input_path="/workspace/include/data/raw/athletes.csv",
        output_path="/workspace/include/data/processed/athletes_clean.parquet"
    )