#!/bin/bash

# Exit on error
set -e

echo "Installing dependencies from requirements.txt"
pip install -r requirements.txt

echo "Running Flask application"
gunicorn --bind=0.0.0.0 --timeout 600 app1:app