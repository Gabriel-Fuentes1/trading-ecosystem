# Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available
- 20GB free disk space

## Option 1: Fix Docker Permissions (Recommended)

This is the permanent solution that allows you to run Docker without sudo.

\`\`\`bash
# Run the setup script
bash scripts/setup_docker_permissions.sh

# Log out and log back in (or restart your computer)
# Then verify Docker works without sudo
docker ps

# Start the system
bash scripts/quick_start.sh
\`\`\`

## Option 2: Use Sudo (Quick Fix)

If you can't log out right now, use this temporary solution:

\`\`\`bash
# Generate secrets (only needed once)
bash scripts/generate_secrets.sh

# Start with sudo
bash scripts/start_with_sudo.sh

# Check status
bash scripts/check_status.sh

# Stop services
bash scripts/stop_with_sudo.sh
\`\`\`

## Verify Installation

After starting the services, check that everything is running:

\`\`\`bash
bash scripts/check_status.sh
\`\`\`

You should see all services running. Wait 2-3 minutes for all services to fully initialize.

## Access the System

Once all services are running:

- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Airflow**: http://localhost:8080 (username: `admin`, password: `admin`)
- **Grafana**: http://localhost:3001 (username: `admin`, password: `grafana_admin_password_2024`)
- **MLflow**: http://localhost:5000

## Troubleshooting

### Docker Permission Denied

If you see "Permission denied" errors:

\`\`\`bash
# Use the sudo scripts
bash scripts/start_with_sudo.sh
\`\`\`

Or fix permissions permanently:

\`\`\`bash
bash scripts/setup_docker_permissions.sh
# Then log out and back in
\`\`\`

### Services Not Starting

Check the logs:

\`\`\`bash
# With sudo if needed
sudo docker-compose logs -f [service-name]

# Example
sudo docker-compose logs -f api-gateway
\`\`\`

### Port Already in Use

If ports are already in use, stop other services or modify the ports in `docker-compose.yml`.

### Out of Memory

Reduce the number of services by commenting them out in `docker-compose.yml`, or increase Docker's memory limit.

## Next Steps

1. Configure your API keys in `.env` file
2. Run the database initialization: `bash scripts/init_database.sh`
3. Start the Airflow DAGs from the web interface
4. Monitor the system in Grafana

For detailed documentation, see `docs/DEPLOYMENT_GUIDE.md`.
