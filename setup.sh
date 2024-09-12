#!/bin/bash

# Define the virtual environment directory name
VENV_DIR="chatbot_venv"

# Check if the virtual environment directory already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists."
else
    # Create the virtual environment
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR

    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Exiting."
        exit 1
    fi
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/Scripts/activate

# Install the required packages
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo "Failed to install packages. Exiting."
        exit 1
    fi
else
    echo "requirements.txt not found. Skipping package installation."
fi

# Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Setup complete."
