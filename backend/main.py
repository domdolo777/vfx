import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
from pydantic import BaseModel

# Import MatAnyone integration
from matanyone_integration import matanyone_wrapper

# Create FastAPI app
app = FastAPI(title="VFX Editor API", description="API for video segmentation and effects")

# Add CORS middleware with more explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and results
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

class SegmentationRequest(BaseModel):
    video_id: str
    frame_index: int
    points: List[List[float]]
    labels: List[int]

class TrackingRequest(BaseModel):
    video_id: str
    object_ids: List[str]

class EffectRequest(BaseModel):
    video_id: str
    object_id: str
    effect_type: str
    effect_params: dict

@app.get("/")
async def root():
    return {"message": "VFX Editor API is running"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file"""
    try:
        # Generate a unique ID for the video
        video_id = f"video_{tempfile.NamedTemporaryFile().name.split('/')[-1]}"
        
        # Create directory for this video
        video_dir = os.path.join("uploads", video_id)
        os.makedirs(video_dir, exist_ok=True)
        
        # Save the uploaded video
        video_path = os.path.join(video_dir, file.filename)
        try:
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        # Check if the file is a valid video
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Extract frames from the video
        frames_dir = os.path.join(video_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file. Please ensure it's a valid video format.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(frames_dir, f"frame_{frame_index:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_index += 1
        
        cap.release()
        
        # Check if we extracted any frames
        if frame_index == 0:
            raise HTTPException(status_code=400, detail="Could not extract any frames from the video. Please check the video file.")
        
        print(f"Uploaded video {file.filename} with ID {video_id}, extracted {frame_index} frames")
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error and return a 500 error
        print(f"Error processing video upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/video-frames/{video_id}")
async def get_video_frames(video_id: str, start: int = 0, count: int = 10):
    """Get frames from a video"""
    frames_dir = os.path.join("uploads", video_id, "frames")
    
    if not os.path.exists(frames_dir):
        raise HTTPException(status_code=404, detail="Video not found")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    
    # Limit to requested range
    end = min(start + count, len(frame_files))
    frame_files = frame_files[start:end]
    
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frames.append({
            "frame_index": int(frame_file.split("_")[1].split(".")[0]),
            "url": f"/uploads/{video_id}/frames/{frame_file}"
        })
    
    return {
        "video_id": video_id,
        "frames": frames,
        "total_frames": len(os.listdir(frames_dir))
    }

@app.post("/segment")
async def segment_object(request: SegmentationRequest):
    """Segment an object in a video frame using MatAnyone"""
    video_dir = os.path.join("uploads", request.video_id)
    if not os.path.exists(video_dir):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Create a directory for segmentation masks
    masks_dir = os.path.join(video_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # Generate a unique ID for this object
    object_id = f"obj_{tempfile.NamedTemporaryFile().name.split('/')[-1]}"
    
    # Load the frame
    frame_path = os.path.join(video_dir, "frames", f"frame_{request.frame_index:05d}.jpg")
    frame = cv2.imread(frame_path)
    
    try:
        # Use MatAnyone to generate a mask based on the points
        mask = matanyone_wrapper.segment_frame(frame, request.points, request.labels)
        
        # Ensure the mask is visible (set to white)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        
        # Save the mask
        mask_path = os.path.join(masks_dir, f"{object_id}_{request.frame_index:05d}.png")
        cv2.imwrite(mask_path, mask)
        
        # Construct the mask URL
        mask_url = f"/uploads/{request.video_id}/masks/{object_id}_{request.frame_index:05d}.png"
        
        print(f"Mask saved to {mask_path}")
        print(f"Mask URL: {mask_url}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Mask min: {mask.min()}, max: {mask.max()}")
        
        return {
            "video_id": request.video_id,
            "object_id": object_id,
            "frame_index": request.frame_index,
            "mask_url": mask_url
        }
    except Exception as e:
        print(f"Error during segmentation: {e}")
        # If MatAnyone fails, fall back to a simple circle mask for testing
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # If points are provided, create a mask based on them
        if request.points:
            # Calculate the center of all points
            center_x = int(sum(p[0] for p in request.points) / len(request.points))
            center_y = int(sum(p[1] for p in request.points) / len(request.points))
            
            # Calculate the average distance from center to points
            distances = [np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in request.points]
            avg_distance = int(sum(distances) / len(distances)) if distances else 50
            
            # Create a circle around the center
            radius = max(avg_distance, 50)  # Minimum radius of 50 pixels
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        else:
            # No points provided, create a default circle in the center
            center_x, center_y = int(width/2), int(height/2)
            cv2.circle(mask, (center_x, center_y), 100, 255, -1)
        
        mask_path = os.path.join(masks_dir, f"{object_id}_{request.frame_index:05d}.png")
        cv2.imwrite(mask_path, mask)
        
        mask_url = f"/uploads/{request.video_id}/masks/{object_id}_{request.frame_index:05d}.png"
        print(f"Fallback mask saved to {mask_path}")
        print(f"Fallback mask URL: {mask_url}")
        print(f"Fallback mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Fallback mask min: {mask.min()}, max: {mask.max()}")
        
        return {
            "video_id": request.video_id,
            "object_id": object_id,
            "frame_index": request.frame_index,
            "mask_url": mask_url,
            "error": str(e)
        }

@app.post("/track")
async def track_objects(request: TrackingRequest):
    """Track objects across all frames using MatAnyone"""
    video_dir = os.path.join("uploads", request.video_id)
    if not os.path.exists(video_dir):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get the frames
    frames_dir = os.path.join(video_dir, "frames")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    
    # Create a directory for tracking results
    tracks_dir = os.path.join(video_dir, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    
    # Load all frames
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        frames.append(frame)
    
    # For each object, track it across all frames
    results = []
    for object_id in request.object_ids:
        try:
            # Get the first frame mask as a reference
            masks_dir = os.path.join(video_dir, "masks")
            mask_files = [f for f in os.listdir(masks_dir) if f.startswith(f"{object_id}_")]
            
            if not mask_files:
                continue
            
            first_mask_path = os.path.join(masks_dir, mask_files[0])
            first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Use MatAnyone to track the object across all frames
            tracked_masks = matanyone_wrapper.track_object(frames, first_mask)
            
            # Save the masks for each frame
            object_tracks = []
            for i, mask in enumerate(tracked_masks):
                track_path = os.path.join(tracks_dir, f"{object_id}_{i:05d}.png")
                cv2.imwrite(track_path, mask)
                
                object_tracks.append({
                    "frame_index": i,
                    "mask_url": f"/uploads/{request.video_id}/tracks/{object_id}_{i:05d}.png"
                })
            
            results.append({
                "object_id": object_id,
                "tracks": object_tracks
            })
        except Exception as e:
            # If MatAnyone fails, fall back to a simple tracking for testing
            object_tracks = []
            
            # Get the first frame mask as a reference
            masks_dir = os.path.join(video_dir, "masks")
            mask_files = [f for f in os.listdir(masks_dir) if f.startswith(f"{object_id}_")]
            
            if not mask_files:
                continue
            
            first_mask_path = os.path.join(masks_dir, mask_files[0])
            first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # For now, just copy the first mask to all frames with slight variations
            for i in range(len(frames)):
                # Add some random variation to simulate tracking
                mask = first_mask.copy()
                if i > 0:
                    # Shift the mask slightly
                    M = np.float32([[1, 0, i % 10], [0, 1, i % 5]])
                    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
                
                track_path = os.path.join(tracks_dir, f"{object_id}_{i:05d}.png")
                cv2.imwrite(track_path, mask)
                
                object_tracks.append({
                    "frame_index": i,
                    "mask_url": f"/uploads/{request.video_id}/tracks/{object_id}_{i:05d}.png"
                })
            
            results.append({
                "object_id": object_id,
                "tracks": object_tracks,
                "error": str(e)
            })
    
    return {
        "video_id": request.video_id,
        "objects": results
    }

@app.post("/apply-effect")
async def apply_effect(request: EffectRequest):
    """Apply an effect to a segmented object"""
    video_dir = os.path.join("uploads", request.video_id)
    if not os.path.exists(video_dir):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Create a directory for effects
    effects_dir = os.path.join(video_dir, "effects")
    os.makedirs(effects_dir, exist_ok=True)
    
    # Get the tracking masks for this object
    tracks_dir = os.path.join(video_dir, "tracks")
    track_files = sorted([f for f in os.listdir(tracks_dir) if f.startswith(f"{request.object_id}_")])
    
    if not track_files:
        raise HTTPException(status_code=404, detail="Object tracks not found")
    
    # Apply the effect to each frame
    effect_frames = []
    for track_file in track_files:
        frame_index = int(track_file.split("_")[1].split(".")[0])
        
        # Load the original frame
        frame_path = os.path.join(video_dir, "frames", f"frame_{frame_index:05d}.jpg")
        frame = cv2.imread(frame_path)
        
        # Load the mask
        track_path = os.path.join(tracks_dir, track_file)
        mask = cv2.imread(track_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply the effect based on the type
        if request.effect_type == "blur":
            # Apply blur to the masked region
            blur_amount = request.effect_params.get("amount", 15)
            blurred = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
            # Apply the mask
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = frame * (1 - mask_3ch) + blurred * mask_3ch
            result = result.astype(np.uint8)
        elif request.effect_type == "bw":
            # Convert to black and white
            bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            # Apply the mask
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = frame * (1 - mask_3ch) + bw * mask_3ch
            result = result.astype(np.uint8)
        elif request.effect_type == "chroma":
            # Apply chromatic aberration
            # Split the channels
            b, g, r = cv2.split(frame)
            # Shift the red channel to the right
            shift_amount = request.effect_params.get("amount", 5)
            M_r = np.float32([[1, 0, shift_amount], [0, 1, 0]])
            r_shifted = cv2.warpAffine(r, M_r, (frame.shape[1], frame.shape[0]))
            # Shift the blue channel to the left
            M_b = np.float32([[1, 0, -shift_amount], [0, 1, 0]])
            b_shifted = cv2.warpAffine(b, M_b, (frame.shape[1], frame.shape[0]))
            # Merge the channels
            chroma = cv2.merge([b_shifted, g, r_shifted])
            # Apply the mask
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = frame * (1 - mask_3ch) + chroma * mask_3ch
            result = result.astype(np.uint8)
        elif request.effect_type == "glow":
            # Apply glow effect
            glow_amount = request.effect_params.get("amount", 10)
            # Blur the image
            blurred = cv2.GaussianBlur(frame, (glow_amount*2+1, glow_amount*2+1), 0)
            # Increase brightness
            brightness = np.ones(blurred.shape, dtype="uint8") * 50
            brightened = cv2.add(blurred, brightness)
            # Apply the mask
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = frame * (1 - mask_3ch) + brightened * mask_3ch
            result = result.astype(np.uint8)
        else:
            # Default: no effect
            result = frame
        
        # Save the result
        effect_path = os.path.join(effects_dir, f"{request.object_id}_{request.effect_type}_{frame_index:05d}.jpg")
        cv2.imwrite(effect_path, result)
        
        effect_frames.append({
            "frame_index": frame_index,
            "effect_url": f"/uploads/{request.video_id}/effects/{request.object_id}_{request.effect_type}_{frame_index:05d}.jpg"
        })
    
    return {
        "video_id": request.video_id,
        "object_id": request.object_id,
        "effect_type": request.effect_type,
        "frames": effect_frames
    }

@app.post("/export-video")
async def export_video(
    video_id: str = Form(...),
    object_ids: str = Form(...),
    effect_types: str = Form(...),
    export_type: str = Form(...)
):
    """Export the final video with all effects applied"""
    
    video_dir = os.path.join("uploads", video_id)
    if not os.path.exists(video_dir):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Parse the object IDs and effect types
    object_ids = object_ids.split(",")
    effect_types = effect_types.split(",")
    
    # Get the original video properties
    frames_dir = os.path.join(video_dir, "frames")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    
    if not frame_files:
        raise HTTPException(status_code=404, detail="No frames found")
    
    # Get the first frame to determine dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    # Create a directory for the export
    export_dir = os.path.join("results", video_id)
    os.makedirs(export_dir, exist_ok=True)
    
    # Create a video writer
    export_path = os.path.join(export_dir, f"{export_type}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(export_path, fourcc, 30.0, (width, height))
    
    # Process each frame
    for i, frame_file in enumerate(frame_files):
        # Load the original frame
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if export_type == "mask":
            # For mask export, create a black frame with white masks
            result = np.zeros_like(frame)
            
            # Add each object's mask
            for object_id in object_ids:
                tracks_dir = os.path.join(video_dir, "tracks")
                track_path = os.path.join(tracks_dir, f"{object_id}_{i:05d}.png")
                
                if os.path.exists(track_path):
                    mask = cv2.imread(track_path, cv2.IMREAD_GRAYSCALE)
                    # Convert mask to white
                    white_mask = np.zeros_like(frame)
                    white_mask[mask > 0] = [255, 255, 255]
                    result = cv2.add(result, white_mask)
        else:
            # For FX export, apply all effects
            result = frame.copy()
            
            # Apply each object's effect
            for object_id, effect_type in zip(object_ids, effect_types):
                effects_dir = os.path.join(video_dir, "effects")
                effect_path = os.path.join(effects_dir, f"{object_id}_{effect_type}_{i:05d}.jpg")
                
                if os.path.exists(effect_path):
                    effect_frame = cv2.imread(effect_path)
                    # Replace the corresponding region in the result
                    tracks_dir = os.path.join(video_dir, "tracks")
                    track_path = os.path.join(tracks_dir, f"{object_id}_{i:05d}.png")
                    
                    if os.path.exists(track_path):
                        mask = cv2.imread(track_path, cv2.IMREAD_GRAYSCALE)
                        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                        result = result * (1 - mask_3ch) + effect_frame * mask_3ch
                        result = result.astype(np.uint8)
        
        # Write the frame to the video
        out.write(result)
    
    out.release()
    
    return {
        "video_id": video_id,
        "export_type": export_type,
        "video_url": f"/results/{video_id}/{export_type}.mp4"
    }

@app.get("/check-mask/{video_id}/{mask_file}")
async def check_mask(video_id: str, mask_file: str):
    """Check if a mask file exists and can be served"""
    mask_path = os.path.join("uploads", video_id, "masks", mask_file)
    
    if not os.path.exists(mask_path):
        return {
            "exists": False,
            "error": "Mask file not found"
        }
    
    try:
        # Try to read the mask file
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            return {
                "exists": True,
                "valid": False,
                "error": "Mask file exists but cannot be read"
            }
        
        return {
            "exists": True,
            "valid": True,
            "shape": mask.shape,
            "dtype": str(mask.dtype),
            "min": int(mask.min()),
            "max": int(mask.max())
        }
    except Exception as e:
        return {
            "exists": True,
            "valid": False,
            "error": str(e)
        }

@app.get("/uploads/{video_id}/masks/{mask_file}")
async def get_mask(video_id: str, mask_file: str):
    """Serve a mask file with the correct content type"""
    mask_path = os.path.join("uploads", video_id, "masks", mask_file)
    
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Mask not found")
    
    # Return the file with the correct content type
    return FileResponse(mask_path, media_type="image/png")

@app.get("/health-check")
async def health_check():
    """Simple endpoint to check if the API is running"""
    return {"status": "ok", "message": "API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 