#!/bin/bash

# This script sets up a GitHub repository for the VFX Editor project

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
else
    echo "Git repository already initialized."
fi

# Add all files to git
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: VFX Editor with MatAnyone integration"

# Instructions for connecting to GitHub
echo ""
echo "To connect this repository to GitHub:"
echo "1. Create a new repository on GitHub (https://github.com/new)"
echo "2. Run the following commands to push to GitHub:"
echo ""
echo "   git remote add origin https://github.com/yourusername/vfx-editor.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Replace 'yourusername' with your GitHub username."
echo ""
echo "Done! Your local repository is ready to be pushed to GitHub." 