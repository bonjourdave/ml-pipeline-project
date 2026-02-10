from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import mlflow
from feast import FeatureStore

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def test_mlflow_connection():
    """Test MLFlow connectivity"""
    print("Testing MLFlow connection...")
    
    # Get tracking URI from Airflow variable
    tracking_uri = Variable.get('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(tracking_uri)

    # Try to create a test run
    with mlflow.start_run(run_name="airflow_connection_test"):
        mlflow.log_param("test_param", "connection_successful")
        mlflow.log_metric("test_metric", 1.0)
        print(f"✓ MLFlow connection successful! Tracking URI: {tracking_uri}")
    
    return "MLFlow connection verified"

def test_feast_connection():
    """Test Feast connectivity"""
    print("Testing Feast connection...")
    
    # Get Feast repo path from Airflow variable
    feast_repo = Variable.get('FEAST_REPO_PATH')
    print(f"Feast repo path: {feast_repo}")
    
    # Initialize feature store
    store = FeatureStore(repo_path=feast_repo)
    
    # List feature views
    feature_views = store.list_feature_views()
    print(f"✓ Feast connection successful! Found {len(feature_views)} feature views:")
    for fv in feature_views:
        print(f"  - {fv.name}")
    
    return "Feast connection verified"

with DAG(
    'test_connections',
    default_args=default_args,
    description='Test MLFlow and Feast connectivity',
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=['test', 'connectivity'],
) as dag:
    
    test_mlflow = PythonOperator(
        task_id='test_mlflow',
        python_callable=test_mlflow_connection,
    )
    
    test_feast = PythonOperator(
        task_id='test_feast',
        python_callable=test_feast_connection,
    )
    
    # Run tests in parallel
    [test_mlflow, test_feast]