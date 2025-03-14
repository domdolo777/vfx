# VFX Editor - Video Effects Application

A powerful video effects editing application that allows users to upload videos, segment objects, apply effects, and export the results.

## Features

- **Video Upload**: Upload videos for editing
- **Object Segmentation**: Segment objects in videos using interactive points
- **Effects Application**: Apply various effects to segmented objects
- **Video Export**: Export the edited videos

## Architecture

The application consists of two main components:

1. **Backend (FastAPI)**: Handles video processing, segmentation, and effects application
2. **Frontend (React)**: Provides a user-friendly interface for interacting with the application

## Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- FFmpeg
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/vfx-editor.git
cd vfx-editor
```

### Backend Setup

1. Set up a Python virtual environment:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Download the MatAnyone model:

```bash
mkdir -p MatAnyone/pretrained_models
# Download the model from the MatAnyone GitHub releases
curl -L https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth -o MatAnyone/pretrained_models/matanyone.pth
```

4. Clone the MatAnyone repository:

```bash
git clone https://github.com/pq-yang/MatAnyone.git
```

### Frontend Setup

1. Install the required Node.js packages:

```bash
cd vfx-editor
npm install
```

## Running the Application

### Start the Backend Server

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

The backend server will start at http://0.0.0.0:8000.

### Start the Frontend Development Server

```bash
cd vfx-editor
npm start
```

The frontend development server will start at http://localhost:3000.

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Upload a video using the upload button
3. Switch to Segmentation Mode
4. Click on an object to add points
5. Press "SEGMENT OBJECT" to segment the object
6. Track the object across frames using "TRACK ALL OBJECTS"
7. Switch to FX Mode to apply effects
8. Export the video with applied effects

## Technical Details

### Object Segmentation

The application uses a combination of techniques for object segmentation:

1. **Interactive Segmentation**: Users can click on objects to add points
2. **GrabCut Algorithm**: Uses OpenCV's GrabCut algorithm to create a refined segmentation mask
3. **MatAnyone Integration**: Integrates with the MatAnyone model for advanced segmentation (when available)

### Video Processing

- FFmpeg is used for video processing
- OpenCV is used for frame extraction and manipulation
- FastAPI is used for the backend API

## Troubleshooting

### Common Issues

1. **Black Screen in Segmentation Mode**:
   - Make sure the video was uploaded correctly
   - Check if the frames were extracted properly

2. **Segmentation Not Working**:
   - Try adding more points to the object
   - Make sure to include both foreground and background points

3. **Backend Server Not Starting**:
   - Check if all required packages are installed
   - Make sure the MatAnyone model is downloaded correctly

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [MatAnyone](https://github.com/pq-yang/MatAnyone) for the segmentation model
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework
- [OpenCV](https://opencv.org/) for computer vision algorithms

## Git Workflow

### Setting Up the Repository

The repository has been set up with MatAnyone as a Git submodule. Here's how it was done:

```bash
# Initialize the repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: VFX Editor with MatAnyone integration"

# Add MatAnyone as a submodule
git submodule add https://github.com/pq-yang/MatAnyone.git MatAnyone

# Commit the submodule
git commit -m "Add MatAnyone as a submodule"

# Connect to GitHub
git remote add origin https://github.com/domdolo777/vfx.git
git branch -M main
git push -u origin main
```

### Pushing Updates

To push updates to the repository:

```bash
# Add your changes
git add .

# Commit your changes
git commit -m "Description of your changes"

# Push to GitHub
git push origin main
```

### Cloning the Repository in a New Workspace

To clone the repository with the MatAnyone submodule:

```bash
# Clone the repository
git clone https://github.com/domdolo777/vfx.git

# Navigate to the project directory
cd vfx

# Initialize and update the submodule
git submodule init
git submodule update
```

### Pulling Updates

To pull updates from the repository:

```bash
# Pull the latest changes
git pull origin main

# Update the submodule
git submodule update --remote
```

### Working with the MatAnyone Submodule

The MatAnyone code is included as a Git submodule, which means it's a reference to a specific commit in the original MatAnyone repository. If you need to make changes to the MatAnyone code:

1. Navigate to the MatAnyone directory: `cd MatAnyone`
2. Make your changes
3. Commit your changes: `git add . && git commit -m "Your changes"`
4. Go back to the main repository: `cd ..`
5. Update the reference to the submodule: `git add MatAnyone`
6. Commit the update: `git commit -m "Update MatAnyone submodule"`
7. Push your changes: `git push origin main` 