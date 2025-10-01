#!/bin/bash

# Start minimal trading ecosystem for testing
# This starts only the core services needed for basic functionality

set -e

echo "🚀 Starting minimal trading ecosystem..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with required variables."
    exit 1
fi

# Load environment variables
source .env

# Check required variables
required_vars=("DB_PASSWORD" "REDIS_PASSWORD" "RABBITMQ_USER" "RABBITMQ_PASSWORD" "JWT_SECRET_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "❌ Error: Missing required environment variables:"
    printf '   - %s\n' "${missing_vars[@]}"
    exit 1
fi

echo "✅ Environment variables validated"
echo ""

# Stop any running containers
echo "🛑 Stopping any existing containers..."
sudo docker-compose -f docker-compose.minimal.yml down

echo ""
echo "🔨 Building and starting services..."
sudo docker-compose -f docker-compose.minimal.yml up -d --build

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

echo ""
echo "📊 Service Status:"
sudo docker-compose -f docker-compose.minimal.yml ps

echo ""
echo "✅ Minimal trading ecosystem started!"
echo ""
echo "📍 Access points:"
echo "   - Web Interface: http://localhost:3000"
echo "   - API Gateway: http://localhost:8000"
echo "   - RabbitMQ Management: http://localhost:15672 (user: ${RABBITMQ_USER})"
echo ""
echo "📝 View logs with: sudo docker-compose -f docker-compose.minimal.yml logs -f [service-name]"
echo "🛑 Stop with: sudo docker-compose -f docker-compose.minimal.yml down"
