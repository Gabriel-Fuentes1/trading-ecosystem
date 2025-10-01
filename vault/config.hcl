# HashiCorp Vault Configuration for Production
# This configuration sets up Vault with file storage backend

ui = true
disable_mlock = false

storage "file" {
  path = "/vault/data"
}

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_disable = 1
  # In production, enable TLS:
  # tls_cert_file = "/vault/tls/vault.crt"
  # tls_key_file = "/vault/tls/vault.key"
}

# API address for clustering
api_addr = "http://0.0.0.0:8200"
cluster_addr = "http://0.0.0.0:8201"

# Logging
log_level = "INFO"
log_format = "json"

# Performance and security settings
default_lease_ttl = "168h"
max_lease_ttl = "720h"

# Enable audit logging
# audit {
#   file {
#     file_path = "/vault/logs/audit.log"
#   }
# }

# Telemetry for monitoring
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
}
