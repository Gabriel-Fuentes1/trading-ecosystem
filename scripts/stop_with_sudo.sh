#!/bin/bash

# Stop Trading Ecosystem with sudo

set -e

echo "=========================================="
echo "Stopping Trading Ecosystem"
echo "=========================================="
echo ""

sudo docker-compose down

echo ""
echo "✅ All services stopped"
echo ""
