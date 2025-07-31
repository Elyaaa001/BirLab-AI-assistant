#!/bin/bash

# 🚀 BirLab AI - Full-Stack Deployment Script
# This script helps you deploy both frontend and backend easily

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis
ROCKET="🚀"
CHECK="✅"
WARNING="⚠️"
ERROR="❌"
INFO="ℹ️"

echo -e "${PURPLE}${ROCKET} BirLab AI - Full-Stack Deployment${NC}"
echo -e "${CYAN}=============================================${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

print_info() {
    echo -e "${BLUE}${INFO} $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

print_error() {
    echo -e "${RED}${ERROR} $1${NC}"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_warning ".env file not found!"
        echo -e "${YELLOW}Creating a template .env file...${NC}"
        
        cat > .env << EOF
# 🌟 BirLab AI - Environment Configuration
# Add your API keys below:

# Google AI (Gemini) - REQUIRED for basic functionality
GOOGLE_AI_API_KEY=your-google-ai-key-here

# Optional AI Providers
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
COHERE_API_KEY=your-cohere-key-here
HUGGINGFACE_API_KEY=your-huggingface-key-here
MISTRAL_API_KEY=your-mistral-key-here
TOGETHER_API_KEY=your-together-key-here
REPLICATE_API_TOKEN=your-replicate-token-here
GROK_API_KEY=your-grok-key-here
PERPLEXITY_API_KEY=your-perplexity-key-here
AI21_API_KEY=your-ai21-key-here
GROQ_API_KEY=your-groq-key-here
FIREWORKS_API_KEY=your-fireworks-key-here

# App Configuration
LOG_LEVEL=INFO
NODE_ENV=production
EOF
        
        print_warning "Please edit .env file and add your API keys!"
        print_info "At minimum, you need GOOGLE_AI_API_KEY for Gemini functionality"
        echo -e "${CYAN}Get your Google AI key from: https://aistudio.google.com/app/apikey${NC}"
        
        read -p "Press Enter after you've updated the .env file..."
    else
        print_status ".env file found"
    fi
}

# Check requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is required but not installed!"
        exit 1
    fi
    
    # Check Node.js (for frontend)
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js $NODE_VERSION found"
    else
        print_warning "Node.js not found - needed for frontend development"
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        print_status "Docker found - container deployment available"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker not found - only local deployment available"
        DOCKER_AVAILABLE=false
    fi
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    # Install main dependencies
    pip3 install -r requirements.txt
    
    # Install backend-specific dependencies
    pip3 install fastapi uvicorn python-multipart websockets python-dotenv
    
    print_status "Python dependencies installed"
}

# Install frontend dependencies
install_frontend_deps() {
    if [ -d "frontend" ]; then
        print_info "Installing frontend dependencies..."
        cd frontend
        
        if command -v npm &> /dev/null; then
            npm install
            # Fix npm vulnerabilities
            npm audit fix --legacy-peer-deps 2>/dev/null || true
            print_status "Frontend dependencies installed"
        else
            print_warning "npm not found - skipping frontend setup"
        fi
        
        cd ..
    fi
}

# Start backend
start_backend() {
    print_info "Starting backend server..."
    
    export PYTHONPATH=$(pwd)
    export PYTHONUNBUFFERED=1
    
    # Start backend in background
    python3 backend/fastapi_server.py &
    BACKEND_PID=$!
    
    # Wait a moment for startup
    sleep 3
    
    # Wait for backend to start
    print_info "Waiting for backend to initialize..."
    sleep 5
    
    # Check if backend is running
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_status "Backend started successfully at http://localhost:8000"
        echo $BACKEND_PID > .backend.pid
    else
        print_error "Backend failed to start!"
        print_warning "Common issues:"
        print_warning "- Missing GOOGLE_AI_API_KEY environment variable"
        print_warning "- Port 8000 already in use"
        print_warning "- Missing Python dependencies"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
}

# Start frontend
start_frontend() {
    if [ -d "frontend" ] && command -v npm &> /dev/null; then
        print_info "Starting frontend server..."
        
        cd frontend
        npm start &
        FRONTEND_PID=$!
        cd ..
        
        # Wait a moment for startup
        sleep 5
        
        print_status "Frontend started at http://localhost:3000"
        echo $FRONTEND_PID > .frontend.pid
    else
        print_warning "Frontend not available - using backend only"
    fi
}

# Docker deployment
deploy_docker() {
    print_info "Deploying with Docker Compose..."
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        # Build and start services
        docker-compose up --build -d
        
        # Wait for services to be ready
        print_info "Waiting for services to start..."
        sleep 10
        
        # Check backend health
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_status "Backend container running at http://localhost:8000"
        else
            print_warning "Backend container may still be starting..."
        fi
        
        # Check frontend
        if curl -f http://localhost:3000 &> /dev/null; then
            print_status "Frontend container running at http://localhost:3000"
        else
            print_warning "Frontend container may still be starting..."
        fi
        
        print_status "Docker deployment complete!"
        print_info "View logs with: docker-compose logs -f"
        print_info "Stop services with: docker-compose down"
    else
        print_error "Docker not available for container deployment"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    
    # Kill background processes
    if [ -f .backend.pid ]; then
        BACKEND_PID=$(cat .backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        rm .backend.pid
    fi
    
    if [ -f .frontend.pid ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        kill $FRONTEND_PID 2>/dev/null || true
        rm .frontend.pid
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Main deployment logic
main() {
    echo -e "${CYAN}Choose deployment method:${NC}"
    echo "1) 🐳 Docker (Recommended - Full-stack containers)"
    echo "2) 💻 Local Development (Backend + Frontend)"
    echo "3) 🌐 Backend Only (API server)"
    echo "4) 📋 Setup Only (Install dependencies and create .env)"
    echo ""
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            print_info "Selected: Docker Deployment"
            check_env_file
            check_requirements
            deploy_docker
            ;;
        2)
            print_info "Selected: Local Development"
            check_env_file
            check_requirements
            install_python_deps
            install_frontend_deps
            start_backend
            start_frontend
            
            echo -e "${GREEN}${ROCKET} Full-stack deployment complete!${NC}"
            echo -e "${CYAN}📱 Frontend: http://localhost:3000${NC}"
            echo -e "${CYAN}🔗 Backend API: http://localhost:8000${NC}"
            echo -e "${CYAN}📚 API Docs: http://localhost:8000/docs${NC}"
            echo ""
            print_info "Press Ctrl+C to stop all services"
            
            # Keep script running
            wait
            ;;
        3)
            print_info "Selected: Backend Only"
            check_env_file
            check_requirements
            install_python_deps
            start_backend
            
            echo -e "${GREEN}${ROCKET} Backend deployment complete!${NC}"
            echo -e "${CYAN}🔗 Backend API: http://localhost:8000${NC}"
            echo -e "${CYAN}📚 API Docs: http://localhost:8000/docs${NC}"
            
            # Keep script running
            wait
            ;;
        4)
            print_info "Selected: Setup Only"
            check_env_file
            check_requirements
            install_python_deps
            install_frontend_deps
            
            echo -e "${GREEN}${CHECK} Setup complete!${NC}"
                    print_info "You can now run:"
        print_info "  Backend: python3 backend/fastapi_server.py"
        print_info "  Frontend: cd frontend && npm start"
        print_info "  Docker: docker-compose up --build"
            ;;
        *)
            print_error "Invalid choice!"
            exit 1
            ;;
    esac
}

# Run main function
main

print_status "Deployment script finished!"
echo -e "${PURPLE}${ROCKET} Happy BirLab AI building! ${ROCKET}${NC}" 