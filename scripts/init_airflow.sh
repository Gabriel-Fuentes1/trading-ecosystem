#!/bin/bash

# Initialize Airflow Database
echo "Initializing Airflow database..."
sudo docker-compose exec airflow-webserver airflow db init

# Create admin user
echo "Creating admin user..."
sudo docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@trading-system.com \
    --password admin123

echo "Airflow initialization complete!"
echo "Access Airflow at http://localhost:8080"
echo "Username: admin"
echo "Password: admin123"
