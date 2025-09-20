#!/bin/bash

# AI-NetGuard-Core Setup Script

echo "Setting up AI-NetGuard-Core development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python version $python_version is compatible"
else
    echo "✗ Python version $python_version is not compatible. Requires Python >= 3.10.0"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Initialize git submodules
echo "Initializing git submodules..."
git submodule init
git submodule update

# Verify submodules
if [ -d "prompts" ] && [ -f "prompts/project-spec.json" ]; then
    echo "✓ Submodules initialized successfully"
else
    echo "✗ Submodule initialization failed"
    exit 1
fi

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To start development, see README.md"