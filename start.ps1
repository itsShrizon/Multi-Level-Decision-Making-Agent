"""
Development configuration script for Windows.
PowerShell version of the startup script.
"""

# Check if Python is installed
function Check-Python {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[INFO] Found Python: $pythonVersion" -ForegroundColor Blue
            return $true
        }
    }
    catch {
        Write-Host "[ERROR] Python is not installed or not in PATH!" -ForegroundColor Red
        Write-Host "Please install Python 3.11+ from https://python.org" -ForegroundColor Red
        return $false
    }
}

# Check environment configuration
function Check-Environment {
    if (!(Test-Path ".env")) {
        Write-Host "[WARNING] .env file not found. Creating from .env.example..." -ForegroundColor Yellow
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Host "[WARNING] Please edit .env file with your OpenAI API key!" -ForegroundColor Yellow
        }
        else {
            Write-Host "[ERROR] .env.example file not found!" -ForegroundColor Red
            return $false
        }
    }
    
    # Read .env file and check for API key
    $envContent = Get-Content ".env" -ErrorAction SilentlyContinue
    $apiKeyLine = $envContent | Where-Object { $_ -match "^OPENAI_API_KEY=" }
    
    if (!$apiKeyLine -or $apiKeyLine -match "your-openai-api-key-here") {
        Write-Host "[ERROR] OPENAI_API_KEY is not set in .env file!" -ForegroundColor Red
        Write-Host "Please edit .env file and add your OpenAI API key." -ForegroundColor Red
        return $false
    }
    
    Write-Host "[SUCCESS] Environment configuration looks good!" -ForegroundColor Green
    return $true
}

# Install dependencies
function Install-Dependencies {
    Write-Host "[INFO] Installing Python dependencies..." -ForegroundColor Blue
    
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Dependencies installed successfully!" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "[ERROR] Failed to install dependencies!" -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "[ERROR] requirements.txt not found!" -ForegroundColor Red
        return $false
    }
}

# Run tests
function Run-Tests {
    Write-Host "[INFO] Running tests..." -ForegroundColor Blue
    
    try {
        pytest tests/ -v
        Write-Host "[SUCCESS] Tests completed!" -ForegroundColor Green
    }
    catch {
        Write-Host "[WARNING] pytest not found or tests failed." -ForegroundColor Yellow
    }
}

# Start the application
function Start-Application {
    Write-Host "[INFO] Starting the Multi-Level Chatbot API..." -ForegroundColor Blue
    Write-Host "[INFO] API will be available at: http://localhost:8000" -ForegroundColor Blue
    Write-Host "[INFO] API Documentation: http://localhost:8000/api/docs" -ForegroundColor Blue
    Write-Host "[INFO] Press Ctrl+C to stop the server" -ForegroundColor Blue
    Write-Host ""
    
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}

# Docker setup
function Start-Docker {
    Write-Host "[INFO] Setting up with Docker..." -ForegroundColor Blue
    
    # Check if Docker is available
    try {
        docker --version | Out-Null
        docker-compose --version | Out-Null
    }
    catch {
        Write-Host "[ERROR] Docker or Docker Compose is not installed!" -ForegroundColor Red
        Write-Host "Please install Docker Desktop from https://docker.com" -ForegroundColor Red
        return
    }
    
    Write-Host "[INFO] Building and starting services..." -ForegroundColor Blue
    docker-compose up --build -d
    
    Write-Host "[SUCCESS] Services started!" -ForegroundColor Green
    Write-Host "[INFO] API: http://localhost:8000" -ForegroundColor Blue
    Write-Host "[INFO] API Docs: http://localhost:8000/api/docs" -ForegroundColor Blue
    Write-Host "[INFO] Flower (Celery): http://localhost:5555" -ForegroundColor Blue
    Write-Host ""
    Write-Host "To view logs: docker-compose logs -f" -ForegroundColor Yellow
    Write-Host "To stop services: docker-compose down" -ForegroundColor Yellow
}

# Show help
function Show-Help {
    Write-Host "Multi-Level Chatbot API Startup Script (PowerShell)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\start.ps1 [OPTION]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  start       Start the application (default)" -ForegroundColor Gray
    Write-Host "  docker      Start with Docker Compose" -ForegroundColor Gray
    Write-Host "  test        Run tests only" -ForegroundColor Gray
    Write-Host "  install     Install dependencies only" -ForegroundColor Gray
    Write-Host "  check       Check environment setup" -ForegroundColor Gray
    Write-Host "  help        Show this help message" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor White
    Write-Host "  .\start.ps1          # Start the application" -ForegroundColor Gray
    Write-Host "  .\start.ps1 docker   # Start with Docker" -ForegroundColor Gray
    Write-Host "  .\start.ps1 test     # Run tests" -ForegroundColor Gray
    Write-Host ""
}

# Main script logic
param(
    [string]$Action = "start"
)

switch ($Action.ToLower()) {
    "start" {
        Write-Host "Starting Multi-Level Chatbot API setup..." -ForegroundColor Cyan
        if ((Check-Python) -and (Check-Environment) -and (Install-Dependencies)) {
            Start-Application
        }
    }
    "docker" {
        if (Check-Environment) {
            Start-Docker
        }
    }
    "test" {
        if ((Check-Python) -and (Check-Environment) -and (Install-Dependencies)) {
            Run-Tests
        }
    }
    "install" {
        if (Check-Python) {
            Install-Dependencies
        }
    }
    "check" {
        if ((Check-Python) -and (Check-Environment)) {
            Write-Host "[SUCCESS] Environment check completed!" -ForegroundColor Green
        }
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "[ERROR] Unknown option: $Action" -ForegroundColor Red
        Show-Help
    }
}
