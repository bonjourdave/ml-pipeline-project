#!/bin/bash

echo "Stopping Airflow..."
astro dev stop

echo "Stopping MLFlow services..."
cd services
docker compose down
cd ..

echo "All services stopped."