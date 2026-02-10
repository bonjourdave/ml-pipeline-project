from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64


# Entity: individual athlete
athlete = Entity(
    name="athlete",
    join_keys=["athlete_id"],
    description="CrossFit athlete"
)


# Data sources
features_v1_source = FileSource(
    path="/usr/local/airflow/include/data/processed/features_v1.parquet",
    timestamp_field="event_timestamp",
)

features_v2_source = FileSource(
    path="/usr/local/airflow/include/data/processed/features_v2.parquet",
    timestamp_field="event_timestamp",
)


# Feature View V1: Base features only
athlete_features_v1 = FeatureView(
    name="athlete_features_v1",
    entities=[athlete],
    schema=[
        Field(name="age", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="weight", dtype=Float32),
        Field(name="bmi", dtype=Float32),
        Field(name="gender_is_male", dtype=Int64),
        Field(name="total_lift", dtype=Float32),  # Target (for training data retrieval)
    ],
    source=features_v1_source,
    ttl=timedelta(days=365),
    online=True,
    description="Feature Version 1: Demographics and physical attributes only"
)


# Feature View V2: Base + Categorical features
athlete_features_v2 = FeatureView(
    name="athlete_features_v2",
    entities=[athlete],
    schema=[
        # Base features
        Field(name="age", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="weight", dtype=Float32),
        Field(name="bmi", dtype=Float32),
        Field(name="gender_is_male", dtype=Int64),
        Field(name="total_lift", dtype=Float32),
        
        # Categorical features
        Field(name="region_encoded", dtype=Float32),
        
        # Eat
        Field(name="eat_weighs_measures", dtype=Int64),
        Field(name="eat_paleo", dtype=Int64),
        Field(name="eat_quality", dtype=Int64),
        Field(name="eat_cheat_meals", dtype=Int64),
        Field(name="eat_convenient", dtype=Int64),
        
        # Background
        Field(name="bg_no_sports", dtype=Int64),
        Field(name="bg_youth_sports", dtype=Int64),
        Field(name="bg_college_sports", dtype=Int64),
        Field(name="bg_pro_sports", dtype=Int64),
        Field(name="bg_recreational", dtype=Int64),
        
        # Experience
        Field(name="exp_coach_start", dtype=Int64),
        Field(name="exp_alone_start", dtype=Int64),
        Field(name="exp_level1_cert", dtype=Int64),
        Field(name="exp_specialty_course", dtype=Int64),
        Field(name="exp_life_changing", dtype=Int64),
        Field(name="exp_trains_others", dtype=Int64),
        
        # Schedule
        Field(name="sched_one_per_day", dtype=Int64),
        Field(name="sched_multi_1x", dtype=Int64),
        Field(name="sched_multi_2x", dtype=Int64),
        Field(name="sched_multi_3x", dtype=Int64),
        Field(name="sched_rest_4plus", dtype=Int64),
        Field(name="sched_rest_under4", dtype=Int64),
        Field(name="sched_strict_rest", dtype=Int64),
        
        # Howlong
        Field(name="howlong_encoded", dtype=Float32),
    ],
    source=features_v2_source,
    ttl=timedelta(days=365),
    online=True,
    description="Feature Version 2: Full feature set with encoded categoricals"
)