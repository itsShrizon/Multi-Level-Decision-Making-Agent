#!/bin/bash

# Multi-Level Chatbot API Startup Script
# This script helps set up and run the FastAPI application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.11 or higher."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_status "Found Python version: $python_version"
}

# Check if required environment variables are set
check_env() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Please edit .env file with your OpenAI API key and other settings."
        else
            print_error ".env.example file not found!"
            exit 1
        fi
    fi
    
    # Source the .env file to check variables
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key-here" ]; then
        print_error "OPENAI_API_KEY is not set in .env file!"
        print_error "Please edit .env file and add your OpenAI API key."
        exit 1
    fi
    
    print_success "Environment configuration looks good!"
}

# Install dependencies
install_deps() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed successfully!"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v
        print_success "Tests completed!"
    else
        print_warning "pytest not found, skipping tests."
    fi
}

# Start the application
start_app() {
    print_status "Starting the Multi-Level Chatbot API..."
    print_status "API will be available at: http://localhost:8000"
    print_status "API Documentation: http://localhost:8000/api/docs"
    print_status "Press Ctrl+C to stop the server"
    
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}

# Docker setup
setup_docker() {
    print_status "Setting up with Docker..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed!"
        exit 1
    fi
    
    print_status "Building and starting services..."
    docker-compose up --build -d
    
    print_success "Services started!"
    print_status "API: http://localhost:8000"
    print_status "API Docs: http://localhost:8000/api/docs"
    print_status "Flower (Celery): http://localhost:5555"
    print_status ""
    print_status "To view logs: docker-compose logs -f"
    print_status "To stop services: docker-compose down"
}

# Show help
show_help() {
    echo "Multi-Level Chatbot API Startup Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  start       Start the application (default)"
    echo "  docker      Start with Docker Compose"
    echo "  test        Run tests only"
    echo "  install     Install dependencies only"
    echo "  check       Check environment setup"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0          # Start the application"
    echo "  $0 docker   # Start with Docker"
    echo "  $0 test     # Run tests"
    echo ""
}

# Main script logic
main() {
    case "${1:-start}" in
        "start")
            print_status "Starting Multi-Level Chatbot API setup..."
            check_python
            check_env
            install_deps
            start_app
            ;;
        "docker")
            check_env
            setup_docker
            ;;
        "test")
            check_python
            check_env
            install_deps
            run_tests
            ;;
        "install")
            check_python
            install_deps
            ;;
        "check")
            check_python
            check_env
            print_success "Environment check completed!"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run the main function
main "$@"
