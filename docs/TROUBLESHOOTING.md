# Troubleshooting Guide

## Docker Permission Issues

### Problem: Permission Denied Error

\`\`\`
PermissionError: [Errno 13] Permission denied
\`\`\`

### Solutions:

#### Option 1: Add User to Docker Group (Recommended)

\`\`\`bash
# Run the setup script
bash scripts/setup_docker_permissions.sh

# Log out and log back in
# OR run this command
newgrp docker

# Verify it works
docker ps
\`\`\`

#### Option 2: Use sudo (Quick Fix)

\`\`\`bash
sudo docker-compose up -d
\`\`\`

#### Option 3: Manual Setup

\`\`\`bash
# Create docker group if it doesn't exist
sudo groupadd docker

# Add your user to the docker group
sudo usermod -aG docker $USER

# Activate the changes
newgrp docker

# Verify
docker ps
\`\`\`

## Missing Environment Variables

### Problem: Warning Messages About Missing Variables

\`\`\`
WARNING: The REDIS_PASSWORD variable is not set. Defaulting to a blank string.
\`\`\`

### Solution:

1. **Copy the example environment file:**
   \`\`\`bash
   cp .env.example .env
   \`\`\`

2. **Edit .env and add your API keys:**
   \`\`\`bash
   nano .env
   # or
   vim .env
   \`\`\`

3. **Generate secure secrets for production:**
   \`\`\`bash
   bash scripts/generate_secrets.sh
   \`\`\`

4. **Required API keys to add:**
   - `BINANCE_API_KEY` - Get from Binance
   - `BINANCE_API_SECRET` - Get from Binance
   - `POLYGON_API_KEY` - Get from Polygon.io

## Service Won't Start

### Check Service Status

\`\`\`bash
docker-compose ps
\`\`\`

### View Service Logs

\`\`\`bash
# All services
docker-compose logs

# Specific service
docker-compose logs -f timescaledb
docker-compose logs -f airflow-webserver
\`\`\`

### Restart a Service

\`\`\`bash
docker-compose restart [service-name]
\`\`\`

### Rebuild a Service

\`\`\`bash
docker-compose up -d --build [service-name]
\`\`\`

## Database Connection Issues

### Problem: Can't Connect to Database

\`\`\`bash
# Check if TimescaleDB is running
docker-compose ps timescaledb

# Check database logs
docker-compose logs timescaledb

# Test connection
docker-compose exec timescaledb psql -U trading_user -d trading
\`\`\`

### Solution: Reset Database

\`\`\`bash
# Stop services
docker-compose down

# Remove database volume
docker volume rm trading-ecosystem_postgres_data

# Start fresh
docker-compose up -d timescaledb
\`\`\`

## Port Already in Use

### Problem: Port Conflict Error

\`\`\`
Error: bind: address already in use
\`\`\`

### Solution: Find and Stop Conflicting Process

\`\`\`bash
# Find process using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 [PID]

# Or change the port in docker-compose.yml
\`\`\`

## Out of Memory

### Problem: Services Crashing Due to Memory

\`\`\`bash
# Check memory usage
docker stats

# Check system memory
free -h
\`\`\`

### Solution: Increase Memory Limits

Edit `docker-compose.yml` and adjust memory limits:

\`\`\`yaml
services:
  service-name:
    mem_limit: 2g
    memswap_limit: 2g
\`\`\`

## Airflow Issues

### Problem: Airflow Webserver Won't Start

\`\`\`bash
# Check logs
docker-compose logs airflow-webserver

# Initialize database
docker-compose exec airflow-webserver airflow db init

# Create admin user
docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
\`\`\`

## Quick Start Issues

### Use the Quick Start Script

\`\`\`bash
bash scripts/quick_start.sh
\`\`\`

This script will:
- Check Docker permissions
- Create .env if missing
- Start services in correct order
- Initialize Vault
- Provide access URLs

## Getting Help

If you're still experiencing issues:

1. Check all logs: `docker-compose logs`
2. Verify .env file has all required variables
3. Ensure Docker has enough resources (CPU, Memory, Disk)
4. Check firewall settings
5. Verify network connectivity

For persistent issues, create a GitHub issue with:
- Error messages
- Output of `docker-compose ps`
- Relevant logs
- Your environment (OS, Docker version)
