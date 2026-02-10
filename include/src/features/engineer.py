import pandas as pd
import numpy as np
from pathlib import Path


def create_base_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create base features used in both V1 and V2"""
    features = pd.DataFrame()
    
    # Numeric features
    features['age'] = data['age']
    features['height'] = data['height']
    features['weight'] = data['weight']
    
    # Derived: BMI
    features['bmi'] = data['weight'] / ((data['height'] / 39.37) ** 2)  # Convert inches to meters
    
    # Gender binary
    features['gender_is_male'] = (data['gender'] == 'Male').astype(int)
    
    # Keep target
    features['total_lift'] = data['total_lift']
    
    # Add timestamp for Feast (required for point-in-time joins)
    features['event_timestamp'] = pd.Timestamp.now()
    
    # Add entity key (row number as athlete_id)
    features['athlete_id'] = range(len(features))
    
    return features


def parse_multiselect(series: pd.Series, keyword: str) -> pd.Series:
    """Parse pipe-separated multi-select survey responses"""
    return series.str.contains(keyword, case=False, na=False).astype(int)


def encode_howlong(series: pd.Series) -> pd.Series:
    """Convert howlong to ordinal encoding, taking max if multiple"""
    mapping = {
        'Less than 6 months': 0,
        '6-12 months': 1,
        '1-2 years': 2,
        '2-4 years': 3,
        '4+ years': 4
    }
    
    def extract_max(val):
        if pd.isna(val):
            return np.nan
        # Find all matches and take max
        matches = [mapping[k] for k in mapping.keys() if k in val]
        return max(matches) if matches else np.nan
    
    return series.apply(extract_max)


def target_encode_region(data: pd.DataFrame) -> pd.Series:
    """Target encoding for region (mean total_lift per region)"""
    region_means = data.groupby('region')['total_lift'].mean()
    return data['region'].map(region_means)


def create_v2_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create categorical features for V2"""
    features = pd.DataFrame()
    
    # Region: target encoding
    features['region_encoded'] = target_encode_region(data)
    
    # Eat: parse multi-select
    features['eat_weighs_measures'] = parse_multiselect(data['eat'], 'weigh and measure')
    features['eat_paleo'] = parse_multiselect(data['eat'], 'Paleo')
    features['eat_quality'] = parse_multiselect(data['eat'], 'quality foods')
    features['eat_cheat_meals'] = parse_multiselect(data['eat'], 'cheat meals')
    features['eat_convenient'] = parse_multiselect(data['eat'], 'convenient')
    
    # Background: parse multi-select
    features['bg_no_sports'] = parse_multiselect(data['background'], 'no athletic background')
    features['bg_youth_sports'] = parse_multiselect(data['background'], 'youth or high school')
    features['bg_college_sports'] = parse_multiselect(data['background'], 'college sports')
    features['bg_pro_sports'] = parse_multiselect(data['background'], 'professional sports')
    features['bg_recreational'] = parse_multiselect(data['background'], 'recreational sports')
    
    # Experience: parse multi-select
    features['exp_coach_start'] = parse_multiselect(data['experience'], 'began CrossFit with a coach')
    features['exp_alone_start'] = parse_multiselect(data['experience'], 'trying it alone')
    features['exp_level1_cert'] = parse_multiselect(data['experience'], 'Level 1 certificate')
    features['exp_specialty_course'] = parse_multiselect(data['experience'], 'specialty courses')
    features['exp_life_changing'] = parse_multiselect(data['experience'], 'life changing')
    features['exp_trains_others'] = parse_multiselect(data['experience'], 'train other people')
    
    # Schedule: parse multi-select
    features['sched_one_per_day'] = parse_multiselect(data['schedule'], 'only do 1 workout')
    features['sched_multi_1x'] = parse_multiselect(data['schedule'], 'multiple workouts in a day 1x')
    features['sched_multi_2x'] = parse_multiselect(data['schedule'], 'multiple workouts in a day 2x')
    features['sched_multi_3x'] = parse_multiselect(data['schedule'], 'multiple workouts in a day 3\\+ times')
    features['sched_rest_4plus'] = parse_multiselect(data['schedule'], 'rest 4 or more days')
    features['sched_rest_under4'] = parse_multiselect(data['schedule'], 'rest fewer than 4 days')
    features['sched_strict_rest'] = parse_multiselect(data['schedule'], 'strictly schedule my rest')
    
    # Howlong: ordinal encoding
    features['howlong_encoded'] = encode_howlong(data['howlong'])
    
    return features


def create_features_v1(input_path: str, output_path: str) -> None:
    """Create Feature Version 1: Base features only"""
    print("=" * 60)
    print("Creating Feature Version 1 (Base features)")
    print("=" * 60)
    
    # Load preprocessed data
    data = pd.read_parquet(input_path)
    print(f"Loaded {len(data)} rows")
    
    # Create base features
    features = create_base_features(data)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_file, index=False)
    
    print(f"Created {len(features.columns)} features: {list(features.columns)}")
    print(f"Saved to: {output_path}")
    print("=" * 60)


def create_features_v2(input_path: str, output_path: str) -> None:
    """Create Feature Version 2: Base + Categorical features"""
    print("=" * 60)
    print("Creating Feature Version 2 (Base + Categorical features)")
    print("=" * 60)
    
    # Load preprocessed data
    data = pd.read_parquet(input_path)
    print(f"Loaded {len(data)} rows")
    
    # Create base features
    base_features = create_base_features(data)
    
    # Create categorical features
    categorical_features = create_v2_categorical_features(data)
    
    # Combine
    features = pd.concat([base_features, categorical_features], axis=1)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_file, index=False)
    
    print(f"Created {len(features.columns)} features: {list(features.columns)}")
    print(f"Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    # For testing locally
    create_features_v1(
        input_path="/workspace/include/data/processed/athletes_clean.parquet",
        output_path="/workspace/include/data/processed/features_v1.parquet"
    )
    
    create_features_v2(
        input_path="/workspace/include/data/processed/athletes_clean.parquet",
        output_path="/workspace/include/data/processed/features_v2.parquet"
    )