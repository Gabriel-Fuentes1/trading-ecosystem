#!/bin/bash

# HashiCorp Vault Initialization Script
# This script initializes Vault in production mode with proper security

set -e

VAULT_ADDR="http://localhost:8200"
VAULT_INIT_FILE="vault_init.json"

echo "🔐 Initializing HashiCorp Vault..."

# Wait for Vault to be ready
echo "⏳ Waiting for Vault to be ready..."
until curl -s $VAULT_ADDR/v1/sys/health > /dev/null 2>&1; do
    echo "Waiting for Vault..."
    sleep 2
done

# Check if Vault is already initialized
if curl -s $VAULT_ADDR/v1/sys/init | jq -r '.initialized' | grep -q true; then
    echo "✅ Vault is already initialized"
    exit 0
fi

# Initialize Vault with 5 key shares and threshold of 3
echo "🚀 Initializing Vault with 5 key shares (threshold: 3)..."
curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "secret_shares": 5,
        "secret_threshold": 3
    }' \
    $VAULT_ADDR/v1/sys/init > $VAULT_INIT_FILE

echo "✅ Vault initialized successfully!"

# Extract unseal keys and root token
UNSEAL_KEY_1=$(jq -r '.keys[0]' $VAULT_INIT_FILE)
UNSEAL_KEY_2=$(jq -r '.keys[1]' $VAULT_INIT_FILE)
UNSEAL_KEY_3=$(jq -r '.keys[2]' $VAULT_INIT_FILE)
ROOT_TOKEN=$(jq -r '.root_token' $VAULT_INIT_FILE)

echo "🔓 Unsealing Vault..."

# Unseal Vault (need 3 keys)
curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"key\": \"$UNSEAL_KEY_1\"}" \
    $VAULT_ADDR/v1/sys/unseal > /dev/null

curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"key\": \"$UNSEAL_KEY_2\"}" \
    $VAULT_ADDR/v1/sys/unseal > /dev/null

curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"key\": \"$UNSEAL_KEY_3\"}" \
    $VAULT_ADDR/v1/sys/unseal > /dev/null

echo "✅ Vault unsealed successfully!"

# Configure Vault with trading-specific secrets engines and policies
echo "⚙️  Configuring Vault for trading system..."

# Enable KV secrets engine for trading secrets
curl -s -X POST \
    -H "X-Vault-Token: $ROOT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "type": "kv-v2",
        "description": "Trading system secrets"
    }' \
    $VAULT_ADDR/v1/sys/mounts/trading

# Create policy for trading services
curl -s -X POST \
    -H "X-Vault-Token: $ROOT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "policy": "path \"trading/*\" {\n  capabilities = [\"read\", \"list\"]\n}\npath \"trading/data/*\" {\n  capabilities = [\"read\", \"list\"]\n}"
    }' \
    $VAULT_ADDR/v1/sys/policies/acl/trading-policy

# Store initial trading secrets
echo "🔑 Storing initial trading secrets..."

# Binance API credentials (placeholder - replace with real values)
curl -s -X POST \
    -H "X-Vault-Token: $ROOT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "data": {
            "api_key": "your_binance_api_key_here",
            "api_secret": "your_binance_api_secret_here",
            "testnet": true
        }
    }' \
    $VAULT_ADDR/v1/trading/data/binance

# Database credentials
curl -s -X POST \
    -H "X-Vault-Token: $ROOT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "data": {
            "username": "trading_user",
            "password": "'${DB_PASSWORD:-trading_password_123}'",
            "host": "timescaledb",
            "port": "5432",
            "database": "trading_db"
        }
    }' \
    $VAULT_ADDR/v1/trading/data/database

# JWT secrets
curl -s -X POST \
    -H "X-Vault-Token: $ROOT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "data": {
            "secret_key": "'$(openssl rand -base64 32)'",
            "algorithm": "HS256",
            "expiration_hours": 24
        }
    }' \
    $VAULT_ADDR/v1/trading/data/jwt

# External API keys (placeholders)
curl -s -X POST \
    -H "X-Vault-Token: $ROOT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "data": {
            "polygon_api_key": "your_polygon_api_key_here",
            "alpha_vantage_api_key": "your_alpha_vantage_api_key_here",
            "news_api_key": "your_news_api_key_here"
        }
    }' \
    $VAULT_ADDR/v1/trading/data/external_apis

echo "✅ Vault configuration completed!"

# Display important information
echo ""
echo "🔐 IMPORTANT: Save these credentials securely!"
echo "=============================================="
echo "Root Token: $ROOT_TOKEN"
echo ""
echo "Unseal Keys (save all 5, need 3 to unseal):"
jq -r '.keys[]' $VAULT_INIT_FILE | nl -w2 -s': '
echo ""
echo "⚠️  Store these credentials in a secure location!"
echo "⚠️  The root token and unseal keys are required for Vault operations!"
echo ""
echo "🌐 Vault UI: http://localhost:8200"
echo "📁 Initialization data saved to: $VAULT_INIT_FILE"

# Set permissions on the init file
chmod 600 $VAULT_INIT_FILE

echo ""
echo "✅ Vault initialization complete!"
