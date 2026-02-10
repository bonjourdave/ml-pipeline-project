#!/bin/bash

echo "Starting MLFlow services..."
cd services
docker compose up -d
cd ..

echo "Waiting for MLFlow to be ready..."
sleep 5

echo "Starting Airflow..."
astro dev start

echo ""
echo "========================================="
echo "Services started successfully!"
echo "========================================="
echo "Airflow UI:  http://localhost:8080"
echo "  Username:  admin"
echo "  Password:  admin"
echo ""
echo "MLFlow UI:   http://localhost:5000"
echo "========================================="