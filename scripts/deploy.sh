#!/bin/bash

# QuantTrade Pro Deployment Script
# Automates the deployment process for production environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
DEPLOY_TYPE=${2:-docker}
VERSION=${3:-latest}

echo -e "${GREEN}🚀 Starting QuantTrade Pro Deployment${NC}"
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "Deploy Type: ${YELLOW}$DEPLOY_TYPE${NC}"
echo -e "Version: ${YELLOW}$VERSION${NC}"
echo ""

# Pre-deployment checks
echo -e "${YELLOW}📋 Running pre-deployment checks...${NC}"

# Check if required tools are installed
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ $1 is available${NC}"
    fi
}

check_tool docker
check_tool docker-compose

if [ "$DEPLOY_TYPE" = "kubernetes" ]; then
    check_tool kubectl
    check_tool helm
fi

if [ "$DEPLOY_TYPE" = "terraform" ]; then
    check_tool terraform
fi

# Check environment file
if [ ! -f ".env.$ENVIRONMENT" ]; then
    echo -e "${RED}❌ Environment file .env.$ENVIRONMENT not found${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Environment file found${NC}"
fi

# Backup current deployment
echo -e "${YELLOW}💾 Creating backup...${NC}"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

if [ "$DEPLOY_TYPE" = "docker" ]; then
    # Backup current containers
    docker-compose ps > $BACKUP_DIR/containers.txt
    docker images > $BACKUP_DIR/images.txt
fi

echo -e "${GREEN}✅ Backup created in $BACKUP_DIR${NC}"

# Deploy based on type
case $DEPLOY_TYPE in
    "docker")
        echo -e "${YELLOW}🐳 Deploying with Docker Compose...${NC}"
        
        # Copy environment file
        cp .env.$ENVIRONMENT .env
        
        # Pull latest images
        docker-compose pull
        
        # Start services
        docker-compose -f docker-compose.prod.yml up -d
        
        # Wait for services to be ready
        echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
        sleep 30
        
        # Health check
        ./scripts/health_check.sh
        ;;
        
    "kubernetes")
        echo -e "${YELLOW}☸️ Deploying to Kubernetes...${NC}"
        
        # Apply configurations
        kubectl apply -f kubernetes/namespace.yaml
        kubectl apply -f kubernetes/configmap.yaml
        kubectl apply -f kubernetes/secrets.yaml
        kubectl apply -f kubernetes/deployments.yaml
        kubectl apply -f kubernetes/services.yaml
        
        # Wait for rollout
        kubectl rollout status deployment/api-gateway -n trading-system
        kubectl rollout status deployment/decision-service -n trading-system
        kubectl rollout status deployment/risk-service -n trading-system
        kubectl rollout status deployment/execution-service -n trading-system
        
        # Health check
        kubectl get pods -n trading-system
        ;;
        
    "terraform")
        echo -e "${YELLOW}🏗️ Deploying with Terraform...${NC}"
        
        cd terraform
        terraform init
        terraform plan -var-file="$ENVIRONMENT.tfvars"
        terraform apply -var-file="$ENVIRONMENT.tfvars" -auto-approve
        cd ..
        ;;
        
    *)
        echo -e "${RED}❌ Unknown deployment type: $DEPLOY_TYPE${NC}"
        exit 1
        ;;
esac

# Post-deployment tasks
echo -e "${YELLOW}🔧 Running post-deployment tasks...${NC}"

# Database migrations
if [ "$DEPLOY_TYPE" = "docker" ]; then
    docker-compose exec -T postgres psql -U postgres -d trading -f /docker-entrypoint-initdb.d/schema.sql
elif [ "$DEPLOY_TYPE" = "kubernetes" ]; then
    kubectl exec -n trading-system deployment/api-gateway -- python manage.py migrate
fi

# Initialize monitoring
if [ -f "monitoring/setup_monitoring.sh" ]; then
    echo -e "${YELLOW}📊 Setting up monitoring...${NC}"
    ./monitoring/setup_monitoring.sh
fi

# Verify deployment
echo -e "${YELLOW}🔍 Verifying deployment...${NC}"

case $DEPLOY_TYPE in
    "docker")
        # Check container status
        if docker-compose ps | grep -q "Up"; then
            echo -e "${GREEN}✅ Containers are running${NC}"
        else
            echo -e "${RED}❌ Some containers are not running${NC}"
            docker-compose ps
            exit 1
        fi
        ;;
        
    "kubernetes")
        # Check pod status
        if kubectl get pods -n trading-system | grep -q "Running"; then
            echo -e "${GREEN}✅ Pods are running${NC}"
        else
            echo -e "${RED}❌ Some pods are not running${NC}"
            kubectl get pods -n trading-system
            exit 1
        fi
        ;;
esac

# Final health check
echo -e "${YELLOW}🏥 Running final health check...${NC}"
sleep 10

# Test API endpoints
API_URL="http://localhost:8000"
if [ "$DEPLOY_TYPE" = "kubernetes" ]; then
    API_URL=$(kubectl get service api-gateway-service -n trading-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8000
fi

if curl -f "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API health check passed${NC}"
else
    echo -e "${RED}❌ API health check failed${NC}"
    exit 1
fi

# Success message
echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📋 Deployment Summary:${NC}"
echo -e "Environment: $ENVIRONMENT"
echo -e "Type: $DEPLOY_TYPE"
echo -e "Version: $VERSION"
echo -e "Backup: $BACKUP_DIR"
echo ""
echo -e "${YELLOW}🔗 Access URLs:${NC}"
echo -e "API Gateway: $API_URL"
echo -e "Dashboard: $API_URL"
echo -e "API Docs: $API_URL/docs"

if [ "$DEPLOY_TYPE" = "docker" ]; then
    echo -e "Grafana: http://localhost:3001"
    echo -e "Prometheus: http://localhost:9090"
fi

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
