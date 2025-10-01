#!/bin/bash

# Start Trading Ecosystem with sudo (for Docker permission issues)
# Use this if you haven't added your user to the docker group yet

set -e

echo "=========================================="
echo "Starting Trading Ecosystem with sudo"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please run: bash scripts/generate_secrets.sh"
    exit 1
fi

echo "✅ .env file found"
echo ""

# Check if Docker is running
if ! sudo docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Stop any existing containers
echo "Stopping existing containers..."
sudo docker-compose down 2>/dev/null || true

echo ""
echo "Starting services..."
echo "This may take a few minutes on first run..."
echo ""

# Start services
sudo docker-compose up -d

echo ""
echo "=========================================="
echo "✅ Services Started!"
echo "=========================================="
echo ""
echo "Check status with:"
echo "  sudo docker-compose ps"
echo ""
echo "View logs with:"
echo "  sudo docker-compose logs -f [service-name]"
echo ""
echo "Access points:"
echo "  - Web Interface: http://localhost:3000"
echo "  - API Gateway: http://localhost:8000"
echo "  - Airflow: http://localhost:8080 (admin/admin)"
echo "  - Grafana: http://localhost:3001 (admin/grafana_admin_password_2024)"
echo "  - MLflow: http://localhost:5000"
echo ""
echo "To fix Docker permissions permanently, run:"
echo "  bash scripts/setup_docker_permissions.sh"
echo "  Then log out and log back in"
echo ""
