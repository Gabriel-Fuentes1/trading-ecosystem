#!/bin/bash

# Start full trading ecosystem with all services
# This includes monitoring, MLOps, and workflow orchestration

set -e

echo "🚀 Starting full trading ecosystem..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with required variables."
    exit 1
fi

# Load environment variables
source .env

echo "✅ Environment variables loaded"
echo ""

# Stop any running containers
echo "🛑 Stopping any existing containers..."
sudo docker-compose down

echo ""
echo "🔨 Building and starting all services..."
sudo docker-compose up -d --build

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 15

echo ""
echo "📊 Service Status:"
sudo docker-compose ps

echo ""
echo "✅ Full trading ecosystem started!"
echo ""
echo "📍 Access points:"
echo "   - Web Interface: http://localhost:3000"
echo "   - API Gateway: http://localhost:8000"
echo "   - Airflow: http://localhost:8080"
echo "   - MLflow: http://localhost:5000"
echo "   - Grafana: http://localhost:3001 (password: ${GRAFANA_PASSWORD})"
echo "   - Prometheus: http://localhost:9090"
echo "   - Jaeger: http://localhost:16686"
echo "   - RabbitMQ: http://localhost:15672"
echo ""
echo "📝 View logs with: sudo docker-compose logs -f [service-name]"
echo "🛑 Stop with: sudo docker-compose down"
