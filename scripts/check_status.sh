#!/bin/bash

# Check status of all services

echo "=========================================="
echo "Trading Ecosystem Status"
echo "=========================================="
echo ""

# Check if running with sudo or not
if docker ps > /dev/null 2>&1; then
    DOCKER_CMD="docker"
    COMPOSE_CMD="docker-compose"
else
    DOCKER_CMD="sudo docker"
    COMPOSE_CMD="sudo docker-compose"
fi

echo "Running containers:"
echo ""
$COMPOSE_CMD ps

echo ""
echo "=========================================="
echo "Service Health Checks"
echo "=========================================="
echo ""

# Check each service
services=("timescaledb:5432" "redis:6379" "vault:8200" "api-gateway:8000" "web:3000")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if $DOCKER_CMD ps | grep -q "$name"; then
        echo "✅ $name is running"
    else
        echo "❌ $name is not running"
    fi
done

echo ""
echo "=========================================="
echo "Quick Access URLs"
echo "=========================================="
echo ""
echo "Web Interface: http://localhost:3000"
echo "API Gateway: http://localhost:8000/docs"
echo "Airflow: http://localhost:8080"
echo "Grafana: http://localhost:3001"
echo "MLflow: http://localhost:5000"
echo ""
