#!/bin/bash

# Brain Tumor Detection - Deployment Script
echo "🧠 Brain Tumor Detection AI - Deployment Script"
echo "================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads logs trained_model

# Set proper permissions
chmod 755 uploads logs
chmod 644 trained_model 2>/dev/null || true

# Function to build and start the application
start_app() {
    echo "🔨 Building Docker image..."
    docker-compose build

    if [ $? -ne 0 ]; then
        echo "❌ Docker build failed!"
        exit 1
    fi

    echo "🚀 Starting Brain Tumor Detection application..."
    docker-compose up -d

    if [ $? -eq 0 ]; then
        echo "✅ Application started successfully!"
        echo ""
        echo "🌐 Access your application at:"
        echo "   - Web Interface: http://localhost:8000"
        echo "   - Health Check: http://localhost:8000/health"
        echo "   - API Documentation: http://localhost:8000/api/info"
        echo ""
        echo "📋 Useful commands:"
        echo "   - View logs: docker-compose logs -f brain-tumor-detection"
        echo "   - Stop app: docker-compose down"
        echo "   - Restart: docker-compose restart"
        echo ""
    else
        echo "❌ Failed to start the application!"
        exit 1
    fi
}

# Function to start with production settings (nginx)
start_production() {
    echo "🔨 Building and starting with production settings..."
    docker-compose --profile production up -d --build

    if [ $? -eq 0 ]; then
        echo "✅ Production deployment started successfully!"
        echo ""
        echo "🌐 Access your application at:"
        echo "   - Web Interface: http://localhost"
        echo "   - Direct App: http://localhost:8000"
        echo ""
    else
        echo "❌ Failed to start production deployment!"
        exit 1
    fi
}

# Function to stop the application
stop_app() {
    echo "🛑 Stopping Brain Tumor Detection application..."
    docker-compose down
    echo "✅ Application stopped!"
}

# Function to view logs
view_logs() {
    echo "📋 Viewing application logs (Press Ctrl+C to exit)..."
    docker-compose logs -f brain-tumor-detection
}

# Function to check status
check_status() {
    echo "📊 Application Status:"
    docker-compose ps
    
    echo ""
    echo "🏥 Health Check:"
    curl -s http://localhost:8000/health || echo "❌ Health check failed - app may not be running"
}

# Main menu
case "$1" in
    "start")
        start_app
        ;;
    "production")
        start_production
        ;;
    "stop")
        stop_app
        ;;
    "restart")
        stop_app
        sleep 2
        start_app
        ;;
    "logs")
        view_logs
        ;;
    "status")
        check_status
        ;;
    "clean")
        echo "🧹 Cleaning up Docker resources..."
        docker-compose down -v
        docker system prune -f
        echo "✅ Cleanup completed!"
        ;;
    *)
        echo "Usage: $0 {start|production|stop|restart|logs|status|clean}"
        echo ""
        echo "Commands:"
        echo "  start      - Start the application in development mode"
        echo "  production - Start with nginx reverse proxy"
        echo "  stop       - Stop the application"
        echo "  restart    - Restart the application"
        echo "  logs       - View application logs"
        echo "  status     - Check application status"
        echo "  clean      - Clean up Docker resources"
        echo ""
        exit 1
        ;;
esac
