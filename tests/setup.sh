#!/bin/bash
# OpenVINO AI Testing Environment Setup Script
# ============================================

set -e

echo "🚀 Setting up OpenVINO AI Testing Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

print_success "Python 3 found: $(python3 --version)"

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Create sample test images and videos
print_status "Creating sample test data..."

# Create sample test images (placeholder files)
mkdir -p test_images
mkdir -p test_videos

# Create placeholder files for testing
touch test_images/sample_cars.jpg
touch test_images/sample_people.jpg
touch test_images/sample_license_plate.jpg
touch test_images/sample_vehicles.jpg
touch test_images/sample_highway.jpg

touch test_videos/sample_traffic.mp4
touch test_videos/sample_parking.mp4

print_success "Created sample test files"

# Make test scripts executable
print_status "Making test scripts executable..."
chmod +x test_runner.py
chmod +x test_yolov8n.py
chmod +x test_lprnet.py
chmod +x test_vehicle_tracking.py

# Create output directory
mkdir -p output

print_success "Setup completed successfully!"

echo ""
echo "📋 Next Steps:"
echo "1. Download models: python test_runner.py --download-models"
echo "2. List available models: python test_runner.py --list-models"
echo "3. Test a model: python test_runner.py --model yolov8n --input test_images/sample_cars.jpg"
echo "4. Test all models: python test_runner.py --model all --input test_images/sample_cars.jpg"
echo ""
echo "🔧 Individual Model Tests:"
echo "• YOLOv8n: python test_yolov8n.py --input test_images/sample_cars.jpg"
echo "• LPRNet: python test_lprnet.py --input test_images/sample_license_plate.jpg"
echo "• Vehicle Tracking: python test_vehicle_tracking.py --input test_images/sample_vehicles.jpg"
echo ""
echo "📁 Directory Structure:"
echo "• models/ - Downloaded OpenVINO models"
echo "• test_images/ - Sample test images"
echo "• test_videos/ - Sample test videos"
echo "• output/ - Generated annotated outputs"
echo ""
echo "🎉 Ready to test OpenVINO AI models!"
