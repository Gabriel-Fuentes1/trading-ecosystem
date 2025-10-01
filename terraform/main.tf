# Terraform configuration for Oracle Cloud Infrastructure deployment
terraform {
  required_version = ">= 1.0"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 4.0"
    }
  }
}

# Configure the Oracle Cloud Infrastructure Provider
provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# Variables
variable "tenancy_ocid" {
  description = "OCID of the tenancy"
  type        = string
}

variable "user_ocid" {
  description = "OCID of the user"
  type        = string
}

variable "fingerprint" {
  description = "Fingerprint of the public key"
  type        = string
}

variable "private_key_path" {
  description = "Path to the private key file"
  type        = string
}

variable "region" {
  description = "Oracle Cloud region"
  type        = string
  default     = "us-ashburn-1"
}

variable "compartment_ocid" {
  description = "OCID of the compartment"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key for instance access"
  type        = string
}

# Data sources
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

data "oci_core_images" "ubuntu_images" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "20.04"
  shape                    = "VM.Standard.E4.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

# VCN and Networking
resource "oci_core_vcn" "trading_vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "trading-vcn"
  dns_label      = "tradingvcn"
}

resource "oci_core_internet_gateway" "trading_igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.trading_vcn.id
  display_name   = "trading-igw"
}

resource "oci_core_route_table" "trading_rt" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.trading_vcn.id
  display_name   = "trading-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.trading_igw.id
  }
}

# Security Groups
resource "oci_core_security_list" "trading_security_list" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.trading_vcn.id
  display_name   = "trading-security-list"

  # Ingress rules
  ingress_security_rules {
    protocol = "6" # TCP
    source   = "0.0.0.0/0"
    tcp_options {
      min = 22
      max = 22
    }
  }

  ingress_security_rules {
    protocol = "6" # TCP
    source   = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }

  ingress_security_rules {
    protocol = "6" # TCP
    source   = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }

  # Trading services ports
  ingress_security_rules {
    protocol = "6" # TCP
    source   = "10.0.0.0/16"
    tcp_options {
      min = 8000
      max = 8003
    }
  }

  # Monitoring ports
  ingress_security_rules {
    protocol = "6" # TCP
    source   = "10.0.0.0/16"
    tcp_options {
      min = 3000
      max = 3100
    }
  }

  ingress_security_rules {
    protocol = "6" # TCP
    source   = "10.0.0.0/16"
    tcp_options {
      min = 9090
      max = 9093
    }
  }

  # Egress rules
  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

# Subnets
resource "oci_core_subnet" "trading_public_subnet" {
  compartment_id      = var.compartment_ocid
  vcn_id              = oci_core_vcn.trading_vcn.id
  cidr_block          = "10.0.1.0/24"
  display_name        = "trading-public-subnet"
  dns_label           = "tradingpublic"
  route_table_id      = oci_core_route_table.trading_rt.id
  security_list_ids   = [oci_core_security_list.trading_security_list.id]
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
}

resource "oci_core_subnet" "trading_private_subnet" {
  compartment_id                 = var.compartment_ocid
  vcn_id                         = oci_core_vcn.trading_vcn.id
  cidr_block                     = "10.0.2.0/24"
  display_name                   = "trading-private-subnet"
  dns_label                      = "tradingprivate"
  prohibit_public_ip_on_vnic     = true
  availability_domain            = data.oci_identity_availability_domains.ads.availability_domains[0].name
}

# Load Balancer
resource "oci_load_balancer_load_balancer" "trading_lb" {
  compartment_id = var.compartment_ocid
  display_name   = "trading-lb"
  shape          = "flexible"
  subnet_ids     = [oci_core_subnet.trading_public_subnet.id]

  shape_details {
    maximum_bandwidth_in_mbps = 100
    minimum_bandwidth_in_mbps = 10
  }
}

# Database (Autonomous Database)
resource "oci_database_autonomous_database" "trading_db" {
  compartment_id           = var.compartment_ocid
  db_name                  = "tradingdb"
  display_name             = "Trading Database"
  admin_password           = var.db_admin_password
  cpu_core_count           = 2
  data_storage_size_in_tbs = 1
  db_version               = "19c"
  is_auto_scaling_enabled  = true
  license_model            = "LICENSE_INCLUDED"
}

# Compute Instances
resource "oci_core_instance" "trading_app_server" {
  count               = 2
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  display_name        = "trading-app-${count.index + 1}"
  shape               = "VM.Standard.E4.Flex"

  shape_config {
    ocpus         = 4
    memory_in_gbs = 16
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.trading_private_subnet.id
    display_name     = "trading-app-vnic-${count.index + 1}"
    assign_public_ip = false
  }

  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu_images.images[0].id
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(templatefile("${path.module}/cloud-init.yaml", {
      docker_compose_content = base64encode(file("${path.module}/../docker-compose.yml"))
    }))
  }
}

resource "oci_core_instance" "trading_monitoring_server" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  display_name        = "trading-monitoring"
  shape               = "VM.Standard.E4.Flex"

  shape_config {
    ocpus         = 2
    memory_in_gbs = 8
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.trading_private_subnet.id
    display_name     = "trading-monitoring-vnic"
    assign_public_ip = false
  }

  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu_images.images[0].id
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(templatefile("${path.module}/monitoring-cloud-init.yaml", {
      monitoring_compose_content = base64encode(file("${path.module}/../monitoring/docker-compose.monitoring.yml"))
    }))
  }
}

# Object Storage for backups
resource "oci_objectstorage_bucket" "trading_backups" {
  compartment_id = var.compartment_ocid
  name           = "trading-backups"
  namespace      = data.oci_objectstorage_namespace.ns.namespace
}

data "oci_objectstorage_namespace" "ns" {
  compartment_id = var.compartment_ocid
}

# Outputs
output "load_balancer_ip" {
  value = oci_load_balancer_load_balancer.trading_lb.ip_address_details[0].ip_address
}

output "database_connection_string" {
  value = oci_database_autonomous_database.trading_db.connection_strings[0].high
}

output "app_server_private_ips" {
  value = oci_core_instance.trading_app_server[*].private_ip
}

output "monitoring_server_private_ip" {
  value = oci_core_instance.trading_monitoring_server.private_ip
}
