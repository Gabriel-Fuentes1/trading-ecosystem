# Docker Troubleshooting Guide

## Quick Start

### Option 1: Minimal Setup (Recommended for Testing)
Start with just the core services:
\`\`\`bash
chmod +x scripts/start_minimal.sh
./scripts/start_minimal.sh
\`\`\`

### Option 2: Full Setup
Start all services including monitoring and MLOps:
\`\`\`bash
chmod +x scripts/start_full.sh
./scripts/start_full.sh
\`\`\`

## Common Issues and Solutions

### 1. Permission Denied Error
**Error:** `permission denied while trying to connect to the Docker daemon socket`

**Solution:**
\`\`\`bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, then verify
docker ps

# Or use sudo for immediate access
sudo docker-compose up -d
\`\`\`

### 2. npm ci Failed (Web Service)
**Error:** `npm ci` command fails during build

**Cause:** Missing package-lock.json file

**Solution:** The Dockerfile now uses `npm install` instead of `npm ci` to handle this automatically.

### 3. Vault Image Not Found
**Error:** `manifest for vault:1.15 not found`

**Solution:** Updated to use `hashicorp/vault:1.17` (already fixed in docker-compose.yml)

### 4. Airflow Build Failed
**Error:** pip install fails with package conflicts

**Solution:** Simplified requirements.txt to remove conflicting packages (already fixed)

### 5. Port Already in Use
**Error:** `port is already allocated`

**Solution:**
\`\`\`bash
# Find what's using the port (example for port 3000)
sudo lsof -i :3000

# Kill the process
sudo kill -9 <PID>

# Or change the port in docker-compose.yml
\`\`\`

### 6. Service Won't Start
**Check logs:**
\`\`\`bash
# For minimal setup
sudo docker-compose -f docker-compose.minimal.yml logs -f <service-name>

# For full setup
sudo docker-compose logs -f <service-name>
\`\`\`

**Common service names:**
- web
- api-gateway
- timescaledb
- redis
- rabbitmq

### 7. Database Connection Failed
**Check database is healthy:**
\`\`\`bash
sudo docker-compose ps timescaledb
\`\`\`

**Connect to database manually:**
\`\`\`bash
sudo docker exec -it trading_timescaledb psql -U trading_user -d trading_db
\`\`\`

### 8. Environment Variables Not Set
**Error:** `variable is not set`

**Solution:**
\`\`\`bash
# Check .env file exists
cat .env

# Verify required variables
grep -E "DB_PASSWORD|REDIS_PASSWORD|RABBITMQ" .env
\`\`\`

## Service Dependencies

### Minimal Setup Dependencies
\`\`\`
timescaledb (database)
  ↓
redis (cache)
  ↓
rabbitmq (message broker)
  ↓
api-gateway (backend API)
  ↓
web (frontend)
\`\`\`

### Full Setup Additional Services
- **airflow**: Workflow orchestration (depends on timescaledb)
- **mlflow**: ML model tracking (depends on timescaledb)
- **vault**: Secrets management
- **prometheus/grafana**: Monitoring
- **jaeger**: Distributed tracing

## Useful Commands

### View all running containers
\`\`\`bash
sudo docker ps
\`\`\`

### View all containers (including stopped)
\`\`\`bash
sudo docker ps -a
\`\`\`

### Stop all services
\`\`\`bash
# Minimal
sudo docker-compose -f docker-compose.minimal.yml down

# Full
sudo docker-compose down
\`\`\`

### Remove all volumes (⚠️ deletes all data)
\`\`\`bash
sudo docker-compose down -v
\`\`\`

### Rebuild specific service
\`\`\`bash
sudo docker-compose build <service-name>
sudo docker-compose up -d <service-name>
\`\`\`

### Access container shell
\`\`\`bash
sudo docker exec -it <container-name> /bin/bash
# or for alpine-based images
sudo docker exec -it <container-name> /bin/sh
\`\`\`

### View resource usage
\`\`\`bash
sudo docker stats
\`\`\`

## Performance Tips

1. **Start with minimal setup** - Test core functionality first
2. **Monitor resource usage** - Use `docker stats` to check CPU/memory
3. **Clean up regularly** - Remove unused images and containers
   \`\`\`bash
   sudo docker system prune -a
   \`\`\`
4. **Use build cache** - Don't use `--no-cache` unless necessary
5. **Limit service resources** - Add resource limits in docker-compose.yml if needed

## Getting Help

If you're still experiencing issues:

1. Check the logs for the specific service
2. Verify all environment variables are set correctly
3. Ensure Docker and Docker Compose are up to date
4. Check system resources (disk space, memory)
5. Try the minimal setup first before running all services
