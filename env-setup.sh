#!/bin/bash

# Prompt the user for their name
echo "Please enter VENV Name:"

# Read the user's input and store it in the 'name' variable
read VENV_NAME

# Display a greeting using the input
echo "Creating: $VENV_NAME"

# Create a data directory
mkdir -p data

# Create data subdirectories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/interim

# Create the virtual environment
echo "Creating virtual environment '$VENV_NAME'..."
python -m venv "$VENV_NAME"
# Activate the virtual environment
echo "Activating virtual environment '$VENV_NAME'..."
source "$VENV_NAME/Scripts/activate"

# Setup environment
echo "Setting up environment..."
python -m pip install --upgrade pip
pip install ipykernel
ipython kernel install --user --name=$VENV_NAME
pip install -r requirements.txt

sed -i "s/CUSTOM_ENV_NAME/$VENV_NAME/g" .gitignore

echo "Virtual environment '$VENV_NAME' created."

# update requirements.txt
echo "Updating requirements.txt..."
pip freeze > requirements.txt

echo "Process Complete. Press any key to exit..."
read -n 1 -s
exit 0