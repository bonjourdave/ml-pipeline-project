FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python packages (updated versions to match Airflow requirements)
RUN pip install --no-cache-dir \
    # Core ML
    pandas>=2.1.4 \
    numpy>=1.26.2 \
    scikit-learn>=1.3.2 \
    # MLOps tools
    mlflow>=2.17.2 \
    feast>=0.40.0 \
    codecarbon>=2.3.4 \
    # Jupyter
    jupyterlab>=4.0.9 \
    notebook>=7.0.6 \
    ipykernel>=6.27.1 \
    # Utilities
    python-dotenv>=1.0.0 \
    pyyaml>=6.0.1 \
    # Visualization
    matplotlib>=3.7.0 \
    seaborn>=0.12.0

# Create notebooks directory structure
RUN mkdir -p /workspace/notebooks/research \
    /workspace/notebooks/demos \
    /workspace/notebooks/scratch

# Set Python to unbuffered mode (better for logs)
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["/bin/bash"]