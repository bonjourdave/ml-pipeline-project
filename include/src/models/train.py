import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import mlflow
import mlflow.sklearn
from codecarbon import EmissionsTracker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from feast import FeatureStore


def get_features_from_feast(
    feast_repo_path: str,
    feature_view_name: str
) -> pd.DataFrame:
    """Retrieve historical features from Feast"""
    print(f"Loading features from Feast: {feature_view_name}")
    
    store = FeatureStore(repo_path=feast_repo_path)
    
    # Get all data from the feature view
    # In real scenario, you'd provide entity_df with specific athlete_ids and timestamps
    # For this tutorial, we'll load the parquet directly since we want all historical data
    
    feature_view = store.get_feature_view(feature_view_name)
    data_path = feature_view.batch_source.path
    
    print(f"Reading from: {data_path}")
    features = pd.read_parquet(data_path)
    
    print(f"Loaded {len(features)} rows, {len(features.columns)} columns")
    return features


def prepare_train_test_split(
    features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """Split features into train/test, separate X and y"""
    
    # Separate features from target
    X = features.drop(columns=['total_lift', 'event_timestamp', 'athlete_id'])
    y = features['total_lift']
    
    print(f"Feature columns: {list(X.columns)}")
    print(f"Target: total_lift")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: Dict[str, Any]
) -> RandomForestRegressor:
    """Train Random Forest model"""
    print("Training Random Forest Regressor...")
    print(f"Hyperparameters: {hyperparameters}")
    
    model = RandomForestRegressor(**hyperparameters)
    model.fit(X_train, y_train)
    
    print("Training complete!")
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model and return metrics"""
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }
    
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2_score']:.4f}")
    
    return metrics, y_pred


def create_plots(
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_importances: pd.DataFrame,
    output_dir: str
) -> Dict[str, str]:
    """Create evaluation plots"""
    print("Creating plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    # 1. Prediction vs Actual scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Total Lift (lbs)')
    plt.ylabel('Predicted Total Lift (lbs)')
    plt.title('Predictions vs Actual Values')
    plt.tight_layout()
    pred_plot = str(output_path / 'predictions.png')
    plt.savefig(pred_plot)
    plt.close()
    plots['predictions'] = pred_plot
    
    # 2. Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Total Lift (lbs)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    residuals_plot = str(output_path / 'residuals.png')
    plt.savefig(residuals_plot)
    plt.close()
    plots['residuals'] = residuals_plot
    
    # 3. Feature importance (top 15)
    plt.figure(figsize=(10, 8))
    top_features = feature_importances.head(15)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    importance_plot = str(output_path / 'feature_importance.png')
    plt.savefig(importance_plot)
    plt.close()
    plots['feature_importance'] = importance_plot
    
    print(f"Plots saved to: {output_dir}")
    return plots


def run_training_pipeline(
    feast_repo_path: str,
    feature_version: str,
    hyperparameters: Dict[str, Any],
    mlflow_tracking_uri: str,
    experiment_name: str = "crossfit_athlete_default"
) -> None:
    """Main training pipeline with MLFlow and CodeCarbon tracking"""
    
    print("=" * 80)
    print(f"TRAINING PIPELINE: Feature Version {feature_version}")
    print("=" * 80)
    
    # Set MLFlow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # DEBUG: Verify what MLFlow is actually using
    actual_uri = mlflow.get_tracking_uri()
    print(f"ACTUAL MLFlow Tracking URI: {actual_uri}")
    
    if "sqlite" in actual_uri.lower():
        raise ValueError(f"ERROR: MLFlow is using SQLite instead of Postgres! URI: {actual_uri}")

    # Enable autologging for sklearn
    mlflow.sklearn.autolog(log_models=True)
    
    # Map feature version to Feast feature view name
    feature_view_map = {
        "v1": "athlete_features_v1",
        "v2": "athlete_features_v2"
    }
    feature_view_name = feature_view_map[feature_version]

    # Create emissions directory if it doesn't exist
    emissions_dir = "/tmp/emissions"
    Path(emissions_dir).mkdir(parents=True, exist_ok=True)

    # Start CodeCarbon tracking
    tracker = EmissionsTracker(
        project_name=f"crossfit_model_{feature_version}",
        output_dir=emissions_dir,
        save_to_file=True,
        log_level="warning"
    )
    tracker.start()
    
    # Start MLFlow run
    with mlflow.start_run(run_name=f"{feature_version}_{hyperparameters['n_estimators']}trees"):
        
        # Log parameters
        mlflow.log_param("feature_version", feature_version)
        mlflow.log_params(hyperparameters)
        
        # 1. Get features from Feast
        features = get_features_from_feast(feast_repo_path, feature_view_name)
        mlflow.log_param("n_samples", len(features))
        mlflow.log_param("n_features", len(features.columns) - 3)  # Exclude target, timestamp, id
        
        # 2. Prepare train/test split
        X_train, X_test, y_train, y_test = prepare_train_test_split(features)
        
        # 3. Train model
        model = train_model(X_train, y_train, hyperparameters)
        
        # 4. Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # 5. Feature importance
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 6. Create plots
        plots = create_plots(y_test, y_pred, feature_importances, "/tmp/plots")
        
        # Log plots
        for plot_name, plot_path in plots.items():
            mlflow.log_artifact(plot_path, artifact_path="plots")
        
        # Log feature importance as CSV
        importance_csv = "/tmp/feature_importance.csv"
        feature_importances.to_csv(importance_csv, index=False)
        mlflow.log_artifact(importance_csv, artifact_path="data")
       
        # # 7. Log model  
        # model_info = mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path="model"
        # )
        
        # 8. Stop carbon tracking and log emissions
        emissions = tracker.stop()
        mlflow.log_metric("carbon_emissions_kg", emissions)
        
        # Log emissions CSV
        emissions_files = list(Path(emissions_dir).glob("emissions*.csv"))
        if emissions_files:
            latest_emissions = max(emissions_files, key=lambda p: p.stat().st_mtime)
            mlflow.log_artifact(str(latest_emissions), artifact_path="emissions")
        
        print("=" * 80)
        print(f"Training complete! Carbon emissions: {emissions:.6f} kg CO2")
        print("=" * 80)


if __name__ == "__main__":
    # For local testing
    run_training_pipeline(
        feast_repo_path="/workspace/include/feast/repo/feature_repo",
        feature_version="v1",
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        },
        mlflow_tracking_uri="http://mlflow-server:5000"
    )