#!/bin/bash

# Setup Monitoring Stack
echo "Setting up Trading System Monitoring Stack..."

# Create directories
mkdir -p monitoring/{prometheus/rules,grafana/{provisioning,dashboards},loki,promtail,alertmanager}

# Set permissions
chmod -R 755 monitoring/

# Create Docker network
docker network create monitoring 2>/dev/null || true

# Start monitoring stack
echo "Starting monitoring services..."
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 30

# Check service health
echo "Checking service health..."
services=("prometheus:9090" "grafana:3001" "loki:3100" "jaeger:16686" "alertmanager:9093")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -s "http://localhost:$port" > /dev/null; then
        echo "✓ $name is running on port $port"
    else
        echo "✗ $name failed to start on port $port"
    fi
done

echo "Monitoring stack setup complete!"
echo ""
echo "Access URLs:"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3001 (admin/admin123)"
echo "- Jaeger: http://localhost:16686"
echo "- AlertManager: http://localhost:9093"
echo ""
echo "Import the trading dashboard from monitoring/grafana/dashboards/trading_dashboard.json"
