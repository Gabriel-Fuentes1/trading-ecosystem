# Deployment Guide

This guide covers deploying the QuantTrade Pro trading ecosystem in production environments.

## 🏗️ Infrastructure Requirements

### Minimum System Requirements
- **CPU**: 8 cores (16 recommended)
- **RAM**: 32GB (64GB recommended)
- **Storage**: 500GB SSD (1TB recommended)
- **Network**: 1Gbps connection with low latency to exchanges

### Recommended Architecture
- **Load Balancer**: 2 instances (HA setup)
- **Application Servers**: 3+ instances (horizontal scaling)
- **Database**: Primary + Read replica
- **Cache**: Redis cluster (3 nodes)
- **Monitoring**: Dedicated monitoring server

## ☁️ Cloud Deployment Options

### Oracle Cloud Infrastructure (OCI)

#### Prerequisites
1. OCI account with appropriate permissions
2. Terraform installed locally
3. OCI CLI configured

#### Deployment Steps

1. **Configure Terraform Variables**
   \`\`\`bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   \`\`\`

   Edit `terraform.tfvars`:
   ```hcl
   tenancy_ocid     = "ocid1.tenancy.oc1..xxx"
   user_ocid        = "ocid1.user.oc1..xxx"
   fingerprint      = "xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx"
   private_key_path = "~/.oci/oci_api_key.pem"
   region           = "us-ashburn-1"
   compartment_ocid = "ocid1.compartment.oc1..xxx"
   ssh_public_key   = "ssh-rsa AAAAB3NzaC1yc2E..."
   \`\`\`

2. **Deploy Infrastructure**
   \`\`\`bash
   terraform init
   terraform plan
   terraform apply
   \`\`\`

3. **Configure DNS**
   \`\`\`bash
   # Point your domain to the load balancer IP
   # Example: trading.yourdomain.com -> <load_balancer_ip>
   \`\`\`

4. **SSL Certificate Setup**
   \`\`\`bash
   # Use Let's Encrypt or upload your SSL certificate
   certbot --nginx -d trading.yourdomain.com
   \`\`\`

### AWS Deployment

#### Using ECS Fargate

1. **Create ECS Cluster**
   \`\`\`bash
   aws ecs create-cluster --cluster-name trading-cluster
   \`\`\`

2. **Deploy Services**
   \`\`\`bash
   # Build and push Docker images
   ./scripts/build_and_push_aws.sh

   # Deploy ECS services
   aws ecs create-service --cli-input-json file://aws/ecs-service.json
   \`\`\`

#### Using EKS

1. **Create EKS Cluster**
   \`\`\`bash
   eksctl create cluster --name trading-cluster --region us-west-2
   \`\`\`

2. **Deploy Application**
   \`\`\`bash
   kubectl apply -f kubernetes/
   \`\`\`

### Google Cloud Platform (GCP)

#### Using GKE

1. **Create GKE Cluster**
   \`\`\`bash
   gcloud container clusters create trading-cluster \
     --num-nodes=3 \
     --machine-type=n1-standard-4 \
     --zone=us-central1-a
   \`\`\`

2. **Deploy Application**
   \`\`\`bash
   kubectl apply -f kubernetes/
   \`\`\`

## 🐳 Docker Deployment

### Single Server Deployment

1. **Prepare the Server**
   \`\`\`bash
   # Install Docker and Docker Compose
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   sudo usermod -aG docker $USER

   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   \`\`\`

2. **Deploy the Application**
   \`\`\`bash
   # Clone repository
   git clone <repository-url>
   cd trading-ecosystem

   # Configure environment
   cp .env.example .env
   # Edit .env with production values

   # Start services
   docker-compose -f docker-compose.prod.yml up -d
   \`\`\`

3. **Setup SSL with Nginx**
   \`\`\`bash
   # Install Certbot
   sudo apt install certbot python3-certbot-nginx

   # Get SSL certificate
   sudo certbot --nginx -d trading.yourdomain.com
   \`\`\`

### Docker Swarm Deployment

1. **Initialize Swarm**
   \`\`\`bash
   docker swarm init
   \`\`\`

2. **Deploy Stack**
   \`\`\`bash
   docker stack deploy -c docker-compose.swarm.yml trading
   \`\`\`

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.x installed

### Deployment Steps

1. **Create Namespace**
   \`\`\`bash
   kubectl create namespace trading-system
   \`\`\`

2. **Configure Secrets**
   \`\`\`bash
   # Create secrets
   kubectl create secret generic trading-secrets \
     --from-literal=jwt-secret-key=$(openssl rand -hex 32) \
     --from-literal=db-password=your-db-password \
     --namespace=trading-system
   \`\`\`

3. **Deploy Database**
   \`\`\`bash
   # Using Helm chart for PostgreSQL
   helm repo add bitnami https://charts.bitnami.com/bitnami
   helm install postgres bitnami/postgresql \
     --namespace trading-system \
     --set auth.postgresPassword=your-password \
     --set primary.persistence.size=100Gi
   \`\`\`

4. **Deploy Redis**
   \`\`\`bash
   helm install redis bitnami/redis \
     --namespace trading-system \
     --set auth.enabled=false \
     --set master.persistence.size=20Gi
   \`\`\`

5. **Deploy Application**
   \`\`\`bash
   kubectl apply -f kubernetes/
   \`\`\`

6. **Verify Deployment**
   \`\`\`bash
   kubectl get pods -n trading-system
   kubectl get services -n trading-system
   \`\`\`

## 🔧 Configuration Management

### Environment Variables

Create production environment file:

\`\`\`bash
# .env.production
DATABASE_URL=postgresql://user:password@db-host:5432/trading
REDIS_HOST=redis-cluster.internal
REDIS_PORT=6379
JWT_SECRET_KEY=your-super-secret-jwt-key
EXCHANGE_API_KEY=your-exchange-api-key
EXCHANGE_SECRET_KEY=your-exchange-secret-key
LOG_LEVEL=INFO
ENVIRONMENT=production
SENTRY_DSN=https://your-sentry-dsn
\`\`\`

### Application Configuration

\`\`\`yaml
# config/production.yaml
database:
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30

redis:
  max_connections: 100
  retry_on_timeout: true

trading:
  max_position_size: 0.15
  var_limit: 0.03
  order_timeout: 30

monitoring:
  metrics_interval: 5
  health_check_interval: 10
\`\`\`

## 🔒 Security Configuration

### SSL/TLS Setup

1. **Generate SSL Certificate**
   \`\`\`bash
   # Using Let's Encrypt
   certbot certonly --standalone -d trading.yourdomain.com
   \`\`\`

2. **Configure Nginx**
   \`\`\`nginx
   server {
       listen 443 ssl http2;
       server_name trading.yourdomain.com;
       
       ssl_certificate /etc/letsencrypt/live/trading.yourdomain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/trading.yourdomain.com/privkey.pem;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   \`\`\`

### Firewall Configuration

\`\`\`bash
# UFW configuration
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
\`\`\`

### Database Security

\`\`\`sql
-- Create application user with limited privileges
CREATE USER trading_app WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE trading TO trading_app;
GRANT USAGE ON SCHEMA public TO trading_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_app;
\`\`\`

## 📊 Monitoring Setup

### Prometheus Configuration

\`\`\`yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'trading-services'
    static_configs:
      - targets: ['api-gateway:8000', 'decision-service:8001']
    scrape_interval: 5s
\`\`\`

### Grafana Setup

1. **Install Grafana**
   \`\`\`bash
   docker run -d \
     --name=grafana \
     -p 3000:3000 \
     -v grafana-storage:/var/lib/grafana \
     grafana/grafana
   \`\`\`

2. **Import Dashboards**
   \`\`\`bash
   # Import pre-built dashboards
   curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
     -H "Content-Type: application/json" \
     -d @monitoring/grafana/dashboards/trading_dashboard.json
   \`\`\`

## 🚀 Performance Optimization

### Database Optimization

\`\`\`sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_trades_timestamp ON trades(timestamp);
CREATE INDEX CONCURRENTLY idx_orders_status ON orders(status);
CREATE INDEX CONCURRENTLY idx_positions_symbol ON positions(symbol);

-- Configure PostgreSQL
-- postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
\`\`\`

### Application Optimization

\`\`\`python
# gunicorn configuration
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
\`\`\`

### Redis Optimization

```redis
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
