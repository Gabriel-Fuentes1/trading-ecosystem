#!/bin/bash

# Setup Docker Permissions Script
# This script helps fix Docker permission issues

set -e

echo "=========================================="
echo "Docker Permissions Setup Script"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✅ Docker is installed"
echo ""

# Check if docker group exists
if ! getent group docker > /dev/null 2>&1; then
    echo "Creating docker group..."
    sudo groupadd docker
fi

# Add current user to docker group
echo "Adding user '$USER' to docker group..."
sudo usermod -aG docker $USER

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You need to log out and log back in for the changes to take effect."
echo ""
echo "Alternatively, you can run:"
echo "  newgrp docker"
echo ""
echo "Or restart your system."
echo ""
echo "After logging back in, verify with:"
echo "  docker ps"
echo ""
echo "If you still have issues, you can run docker-compose with sudo:"
echo "  sudo docker-compose up -d"
echo ""
