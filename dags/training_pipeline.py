from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import sys

# Add include/src to path so we can import our modules
sys.path.insert(0, '/usr/local/airflow/include/src')

from data.preprocess import preprocess_data
from features.engineer import create_features_v1, create_features_v2
from models.train import run_training_pipeline


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def preprocess_task():
    """Preprocess raw data"""
    print("Starting data preprocessing...")
    preprocess_data(
        input_path="/usr/local/airflow/include/data/raw/athletes.csv",
        output_path="/usr/local/airflow/include/data/processed/athletes_clean.parquet"
    )
    print("Preprocessing complete!")


def engineer_features_v1_task():
    """Create Feature Version 1"""
    print("Engineering Feature Version 1...")
    create_features_v1(
        input_path="/usr/local/airflow/include/data/processed/athletes_clean.parquet",
        output_path="/usr/local/airflow/include/data/processed/features_v1.parquet"
    )
    print("Feature Version 1 complete!")


def engineer_features_v2_task():
    """Create Feature Version 2"""
    print("Engineering Feature Version 2...")
    create_features_v2(
        input_path="/usr/local/airflow/include/data/processed/athletes_clean.parquet",
        output_path="/usr/local/airflow/include/data/processed/features_v2.parquet"
    )
    print("Feature Version 2 complete!")


def train_model_task(feature_version: str, n_estimators: int, max_depth: int):
    """Train model with specified configuration"""
    print(f"Training model: {feature_version}, trees={n_estimators}, depth={max_depth}")
    
    # Get Feast repo path
    feast_repo = Variable.get('FEAST_REPO_PATH')
    
    # Hyperparameters
    hyperparameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Run training pipeline
    run_training_pipeline(
        feast_repo_path=feast_repo,
        feature_version=feature_version,
        hyperparameters=hyperparameters,
        mlflow_tracking_uri=Variable.get('MLFLOW_TRACKING_URI'),
        experiment_name="crossfit_athlete_airflow_run"
    )
    
    print(f"Training complete: {feature_version}, trees={n_estimators}, depth={max_depth}")


with DAG(
    'crossfit_training_pipeline',
    default_args=default_args,
    description='Train CrossFit athlete performance models with multiple configurations',
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=['ml', 'training', 'crossfit'],
) as dag:
    
    # Step 1: Preprocess data
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_task,
    )
    
    # Step 2: Engineer both feature versions (in parallel)
    engineer_v1 = PythonOperator(
        task_id='engineer_features_v1',
        python_callable=engineer_features_v1_task,
    )
    
    engineer_v2 = PythonOperator(
        task_id='engineer_features_v2',
        python_callable=engineer_features_v2_task,
    )
    
    # Step 3: Train 4 model configurations (in parallel after feature engineering)
    
    # V1 experiments
    train_v1_100_10 = PythonOperator(
        task_id='train_v1_trees100_depth10',
        python_callable=train_model_task,
        op_kwargs={
            'feature_version': 'v1',
            'n_estimators': 100,
            'max_depth': 10
        }
    )
    
    train_v1_200_20 = PythonOperator(
        task_id='train_v1_trees200_depth20',
        python_callable=train_model_task,
        op_kwargs={
            'feature_version': 'v1',
            'n_estimators': 200,
            'max_depth': 20
        }
    )
    
    # V2 experiments
    train_v2_100_10 = PythonOperator(
        task_id='train_v2_trees100_depth10',
        python_callable=train_model_task,
        op_kwargs={
            'feature_version': 'v2',
            'n_estimators': 100,
            'max_depth': 10
        }
    )
    
    train_v2_200_20 = PythonOperator(
        task_id='train_v2_trees200_depth20',
        python_callable=train_model_task,
        op_kwargs={
            'feature_version': 'v2',
            'n_estimators': 200,
            'max_depth': 20
        }
    )
    
    # Define dependencies
    preprocess >> [engineer_v1, engineer_v2]
    engineer_v1 >> [train_v1_100_10, train_v1_200_20]
    engineer_v2 >> [train_v2_100_10, train_v2_200_20]


### Visualize the Pipeline
'''
The DAG structure:

preprocess_data
    ├─> engineer_features_v1
    │       ├─> train_v1_trees100_depth10
    │       └─> train_v1_trees200_depth20
    │
    └─> engineer_features_v2
            ├─> train_v2_trees100_depth10
            └─> train_v2_trees200_depth20
'''