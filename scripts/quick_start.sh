#!/bin/bash

# Quick Start Script
# This script helps you get started quickly with the trading ecosystem

set -e

echo "=========================================="
echo "QuantTrade Pro - Quick Start"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   - BINANCE_API_KEY"
    echo "   - BINANCE_API_SECRET"
    echo "   - POLYGON_API_KEY"
    echo ""
    read -p "Press Enter after you've updated the .env file..."
fi

# Check Docker permissions
echo "Checking Docker permissions..."
if docker ps > /dev/null 2>&1; then
    echo "✅ Docker permissions OK"
else
    echo "❌ Docker permission denied"
    echo ""
    echo "Running setup script..."
    bash scripts/setup_docker_permissions.sh
    echo ""
    echo "Please log out and log back in, then run this script again."
    exit 1
fi

echo ""
echo "Starting services..."
echo ""

# Create necessary directories
mkdir -p logs data/postgres data/redis data/mlflow

# Start core infrastructure first
echo "Starting core infrastructure (database, redis, vault)..."
docker-compose up -d timescaledb redis vault

echo "Waiting for services to be ready..."
sleep 10

# Initialize Vault
echo "Initializing Vault..."
bash scripts/init_vault.sh || echo "⚠️  Vault initialization skipped (may already be initialized)"

# Start remaining services
echo "Starting remaining services..."
docker-compose up -d

echo ""
echo "=========================================="
echo "✅ Services Started!"
echo "=========================================="
echo ""
echo "Access the services at:"
echo "  - Web Interface: http://localhost:3000"
echo "  - API Gateway: http://localhost:8000"
echo "  - Airflow: http://localhost:8080 (admin/admin)"
echo "  - MLflow: http://localhost:5000"
echo "  - Grafana: http://localhost:3001 (admin/[GRAFANA_PASSWORD])"
echo ""
echo "Check service status:"
echo "  docker-compose ps"
echo ""
echo "View logs:"
echo "  docker-compose logs -f [service-name]"
echo ""
