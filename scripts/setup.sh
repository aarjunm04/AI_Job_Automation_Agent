#!/bin/bash
# =============================================================================
# AI JOB AUTOMATION AGENT - COMPLETE SETUP SCRIPT
# =============================================================================
# One-command setup for the entire AI Job Automation system
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="AI Job Automation Agent"
REQUIRED_TOOLS=("docker" "docker-compose" "curl" "git")
PYTHON_VERSION="3.11"
NODE_VERSION="18"

echo -e "${CYAN}${BOLD}"
echo "============================================================================="
echo "🤖 AI JOB AUTOMATION AGENT - COMPLETE SETUP"
echo "============================================================================="
echo -e "${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ This script should not be run as root${NC}"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    local status=$1
    local message=$2
    
    if [ "$status" = "ok" ]; then
        echo -e "${GREEN}✅ ${message}${NC}"
    elif [ "$status" = "warn" ]; then
        echo -e "${YELLOW}⚠️  ${message}${NC}"
    elif [ "$status" = "error" ]; then
        echo -e "${RED}❌ ${message}${NC}"
    else
        echo -e "${BLUE}ℹ️  ${message}${NC}"
    fi
}

# Check system requirements
echo -e "${YELLOW}🔍 Checking system requirements...${NC}"

# Check OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    print_status "ok" "Operating System: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    print_status "ok" "Operating System: macOS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
    print_status "warn" "Operating System: Windows (some features may need adjustment)"
else
    print_status "warn" "Unknown operating system: $OSTYPE"
fi

# Check required tools
missing_tools=()
for tool in "${REQUIRED_TOOLS[@]}"; do
    if command_exists "$tool"; then
        version=$($tool --version 2>/dev/null | head -n 1 | cut -d' ' -f3 2>/dev/null || echo "unknown")
        print_status "ok" "$tool is installed (version: $version)"
    else
        missing_tools+=("$tool")
        print_status "error" "$tool is not installed"
    fi
done

# Install missing tools (Linux/macOS only)
if [ ${#missing_tools[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}📦 Installing missing tools...${NC}"
    
    if [[ "$OS" == "linux" ]]; then
        # Update package list
        if command_exists "apt-get"; then
            sudo apt-get update
            
            for tool in "${missing_tools[@]}"; do
                case $tool in
                    "docker")
                        print_status "info" "Installing Docker..."
                        curl -fsSL https://get.docker.com -o get-docker.sh
                        sudo sh get-docker.sh
                        sudo usermod -aG docker $USER
                        rm get-docker.sh
                        ;;
                    "docker-compose")
                        print_status "info" "Installing Docker Compose..."
                        sudo apt-get install -y docker-compose-plugin
                        ;;
                    "curl")
                        sudo apt-get install -y curl
                        ;;
                    "git")
                        sudo apt-get install -y git
                        ;;
                esac
            done
            
        elif command_exists "yum"; then
            sudo yum update -y
            # Similar installations for CentOS/RHEL
        fi
        
    elif [[ "$OS" == "macos" ]]; then
        if command_exists "brew"; then
            for tool in "${missing_tools[@]}"; do
                case $tool in
                    "docker")
                        print_status "info" "Installing Docker Desktop..."
                        brew install --cask docker
                        ;;
                    *)
                        brew install "$tool"
                        ;;
                esac
            done
        else
            print_status "error" "Homebrew not found. Please install required tools manually."
            exit 1
        fi
    fi
    
    print_status "warn" "Some tools were installed. You may need to restart your terminal or log out/in."
fi

# Check Docker daemon
if command_exists "docker"; then
    if docker info >/dev/null 2>&1; then
        print_status "ok" "Docker daemon is running"
    else
        print_status "error" "Docker daemon is not running. Please start Docker and run this script again."
        exit 1
    fi
fi

# Create directory structure
echo -e "\n${YELLOW}📁 Creating project directory structure...${NC}"

directories=(
    "chrome_extension/icons"
    "chrome_extension/utils" 
    "docker/nginx"
    "docker/postgres"
    "docker/prometheus"
    "docker/grafana/dashboards"
    "docker/grafana/datasources"
    "n8n_workflows"
    "n8n_data"
    "logs"
    "screenshots"
    "downloads"
    "artifacts"
    "config"
    "scripts"
    "core"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "ok" "Created directory: $dir"
    else
        print_status "info" "Directory already exists: $dir"
    fi
done

# Set up environment file
echo -e "\n${YELLOW}📝 Setting up environment configuration...${NC}"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "ok" "Created .env from .env.example"
    else
        # Create basic .env file
        cat > .env << EOF
# =============================================================================
# AI JOB AUTOMATION AGENT - ENVIRONMENT CONFIGURATION
# =============================================================================

# N8N Configuration
N8N_PORT=5678
N8N_HOST=localhost
WEBHOOK_URL=http://localhost:5678
GENERIC_TIMEZONE=Asia/Kolkata
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=ai_job_2025

# Database Configuration
POSTGRES_DB=n8n_ai_job
POSTGRES_USER=n8n_user
POSTGRES_PASSWORD=n8n_password_2025
DB_TYPE=postgresdb

# AI Service API Keys (REQUIRED - ADD YOUR KEYS)
OPENAI_API_KEY=your-openai-key-here
PERPLEXITY_API_KEY=your-perplexity-key-here

# Notion Integration (REQUIRED - ADD YOUR KEYS)
NOTION_API_KEY=your-notion-key-here
APPLICATIONS_DB_ID=your-applications-db-id
JOB_TRACKER_DB_ID=your-job-tracker-db-id

# Resume Generation (OPTIONAL)
OVERLEAF_API_KEY=your-overleaf-key-here
OVERLEAF_PROJECT_ID=your-overleaf-project-id

# Notifications (OPTIONAL)
SLACK_WEBHOOK_URL=your-slack-webhook-url
DISCORD_WEBHOOK_URL=your-discord-webhook-url

# System Configuration
DEBUG_MODE=false
NGINX_HOST=localhost

# Monitoring (OPTIONAL)
GRAFANA_USER=admin
GRAFANA_PASSWORD=ai_job_grafana_2025

# File Storage (OPTIONAL)
MINIO_ROOT_USER=ai_job_admin
MINIO_ROOT_PASSWORD=ai_job_minio_2025
EOF
        print_status "ok" "Created basic .env file"
    fi
else
    print_status "info" ".env file already exists"
fi

# Generate missing configuration files
echo -e "\n${YELLOW}⚙️  Generating configuration files...${NC}"

# Create basic package.json if it doesn't exist
if [ ! -f "package.json" ]; then
    cat > package.json << EOF
{
  "name": "ai-job-automation",
  "version": "1.0.0",
  "description": "AI-powered job automation system",
  "main": "playwright_server.js",
  "scripts": {
    "start": "node playwright_server.js",
    "dev": "nodemon playwright_server.js",
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "playwright": "^1.55.1",
    "ws": "^8.14.2",
    "uuid": "^9.0.1",
    "axios": "^1.6.0",
    "moment": "^2.29.4",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
EOF
    print_status "ok" "Created package.json"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << EOF
# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Dependencies
node_modules/
.pnp
.pnp.js

# Production builds
build/
dist/

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Docker
.dockerignore
docker-compose.override.yml

# Database
postgres_data/
n8n_data/

# Screenshots and artifacts
screenshots/
downloads/
artifacts/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# API keys and secrets
sa-key.json
*-key.pem
*.key

# Temporary files
temp/
tmp/

# Chrome extension builds
chrome_extension/*.crx
chrome_extension/*.zip
EOF
    print_status "ok" "Created .gitignore"
fi

# Install Node.js dependencies
echo -e "\n${YELLOW}📦 Installing Node.js dependencies...${NC}"

if command_exists "npm"; then
    if [ -f "package.json" ]; then
        npm install --no-audit --no-fund
        print_status "ok" "Node.js dependencies installed"
    fi
else
    print_status "warn" "npm not found. Please install Node.js and run 'npm install' manually."
fi

# Create Chrome extension icons (if ImageMagick is available)
echo -e "\n${YELLOW}🎨 Creating Chrome extension icons...${NC}"

if command_exists "convert"; then
    print_status "info" "Generating Chrome extension icons..."
    
    for size in 16 32 48 128; do
        if [ ! -f "chrome_extension/icons/icon-${size}.png" ]; then
            convert -size ${size}x${size} \
                gradient:'#667eea-#764ba2' \
                -font Arial -pointsize $((size/3)) \
                -fill white -gravity center \
                -annotate +0+0 '🤖' \
                chrome_extension/icons/icon-${size}.png 2>/dev/null || {
                
                # Fallback: create simple colored squares
                convert -size ${size}x${size} \
                    xc:'#667eea' \
                    chrome_extension/icons/icon-${size}.png 2>/dev/null || {
                    
                    # Ultimate fallback: create tiny PNG files
                    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77wgAAAABJRU5ErkJggg==" | base64 -d > chrome_extension/icons/icon-${size}.png
                }
            }
            print_status "ok" "Generated icon-${size}.png"
        fi
    done
else
    print_status "warn" "ImageMagick not found. Creating placeholder icons..."
    
    for size in 16 32 48 128; do
        if [ ! -f "chrome_extension/icons/icon-${size}.png" ]; then
            # Create minimal PNG placeholder
            echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77wgAAAABJRU5ErkJggg==" | base64 -d > chrome_extension/icons/icon-${size}.png
        fi
    done
fi

# Set up database initialization script
echo -e "\n${YELLOW}🗄️  Setting up database initialization...${NC}"

cat > docker/postgres/init-multiple-databases.sh << 'EOF'
#!/bin/bash
# Create multiple databases for the AI Job Automation system

set -e
set -u

function create_user_and_database() {
    local database=$1
    echo "Creating user and database '$database'"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE $database;
        GRANT ALL PRIVILEGES ON DATABASE $database TO $POSTGRES_USER;
EOSQL
}

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Multiple database creation requested: $POSTGRES_MULTIPLE_DATABASES"
    for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        create_user_and_database $db
    done
    echo "Multiple databases created"
fi
EOF

chmod +x docker/postgres/init-multiple-databases.sh
print_status "ok" "Database initialization script created"

# Check Docker Compose file
echo -e "\n${YELLOW}🐳 Validating Docker Compose configuration...${NC}"

if [ -f "docker-compose.yml" ]; then
    if docker-compose config >/dev/null 2>&1; then
        print_status "ok" "Docker Compose configuration is valid"
    else
        print_status "error" "Docker Compose configuration has errors"
        print_status "info" "Run 'docker-compose config' to see details"
    fi
else
    print_status "warn" "docker-compose.yml not found"
fi

# Make scripts executable
echo -e "\n${YELLOW}🔧 Setting up executable scripts...${NC}"

scripts=("scripts/setup.sh" "scripts/build_extension.sh" "scripts/daily_cron.sh")

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        print_status "ok" "Made $script executable"
    fi
done

# Test basic functionality
echo -e "\n${YELLOW}🧪 Running basic tests...${NC}"

# Test Python availability
if command_exists "python3"; then
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "ok" "Python 3 available (version: $python_version)"
    
    # Test if we can import basic modules
    if python3 -c "import json, os, sys" 2>/dev/null; then
        print_status "ok" "Python basic modules available"
    else
        print_status "warn" "Python basic modules test failed"
    fi
else
    print_status "warn" "Python 3 not found. Some features may not work."
fi

# Test Node.js availability
if command_exists "node"; then
    node_version=$(node --version 2>/dev/null)
    print_status "ok" "Node.js available (version: $node_version)"
else
    print_status "warn" "Node.js not found. Playwright server may not work."
fi

# Final setup summary
echo -e "\n${CYAN}${BOLD}🎉 Setup Summary${NC}"
echo "============================================================================="

print_status "ok" "Project structure created"
print_status "ok" "Configuration files generated"

if [ -f ".env" ]; then
    print_status "warn" "IMPORTANT: Edit .env file with your API keys before starting the system"
fi

echo -e "\n${YELLOW}📋 Next Steps:${NC}"
echo "1. 📝 Edit the .env file with your API keys:"
echo "   - OpenAI API key (required)"
echo "   - Perplexity API key (optional, for company research)" 
echo "   - Notion API key and database IDs (required for job tracking)"
echo ""
echo "2. 🚀 Start the system:"
echo "   docker-compose up -d"
echo ""
echo "3. 🌐 Access the interfaces:"
echo "   - N8N Workflows: http://localhost:5678"
echo "   - Grafana Monitoring: http://localhost:3000"
echo "   - Backend API: http://localhost:8080"
echo ""
echo "4. 📱 Install Chrome Extension:"
echo "   - Open Chrome -> Extensions -> Load unpacked"
echo "   - Select the chrome_extension/ folder"
echo ""
echo "5. ⚙️  Configure the extension with your API keys and preferences"

echo -e "\n${GREEN}${BOLD}✅ Setup completed successfully!${NC}"
echo -e "${BLUE}For help and documentation: https://github.com/your-repo/ai-job-automation${NC}"

# Optional: Start the system if user confirms
echo -e "\n${YELLOW}Would you like to start the Docker services now? (y/N)${NC}"
read -r start_services

if [[ $start_services =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}🐳 Starting Docker services...${NC}"
    
    if docker-compose up -d; then
        print_status "ok" "Docker services started successfully!"
        echo -e "\n${CYAN}🌐 Access URLs:${NC}"
        echo "   - N8N: http://localhost:5678"
        echo "   - Grafana: http://localhost:3000"
        echo "   - Backend: http://localhost:8080"
        echo ""
        echo "   Login credentials are in your .env file"
    else
        print_status "error" "Failed to start Docker services"
        print_status "info" "Check your .env file and run 'docker-compose up -d' manually"
    fi
fi

echo -e "\n${GREEN}🎯 Happy job hunting with AI automation!${NC}"
