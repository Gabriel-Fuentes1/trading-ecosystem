#!/bin/bash

# Generate Secure Secrets Script
# This script generates secure random secrets for production use

set -e

echo "=========================================="
echo "Generating Secure Secrets"
echo "=========================================="
echo ""

# Function to generate random string
generate_secret() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Generate Fernet key for Airflow
generate_fernet_key() {
    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
}

echo "Copy these values to your .env file:"
echo ""
echo "# Generated Secrets - $(date)"
echo "POSTGRES_PASSWORD=$(generate_secret)"
echo "REDIS_PASSWORD=$(generate_secret)"
echo "VAULT_ROOT_TOKEN=hvs.$(generate_secret)"
echo "AIRFLOW_FERNET_KEY=$(generate_fernet_key)"
echo "AIRFLOW_SECRET_KEY=$(generate_secret)"
echo "RABBITMQ_PASSWORD=$(generate_secret)"
echo "JWT_SECRET_KEY=$(generate_secret)"
echo "GRAFANA_PASSWORD=$(generate_secret)"
echo ""
echo "=========================================="
echo "⚠️  IMPORTANT: Save these secrets securely!"
echo "=========================================="
