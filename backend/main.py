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
        # Return a relative URL path that will be handled by the frontend
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
        
        # Construct the mask URL - use a relative path that will be handled by the frontend
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
        
        # Use a relative URL that will be handled by the frontend
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
    
    # Create a directory for tracked masks
    masks_dir = os.path.join(video_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # Load all frames
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        frames.append(frame)
    
    # Track each object
    tracked_objects = []
    for object_id in request.object_ids:
        # Find the initial mask
        initial_masks = [f for f in os.listdir(masks_dir) if f.startswith(object_id)]
        if not initial_masks:
            continue
        
        # Get the initial frame index (assuming mask filename format: object_id_frameindex.png)
        initial_mask_file = sorted(initial_masks)[0]
        initial_frame_index = int(initial_mask_file.split("_")[1].split(".")[0])
        
        # Load the initial mask
        initial_mask_path = os.path.join(masks_dir, initial_mask_file)
        initial_mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE)
        
        try:
            # Use MatAnyone to track the object
            print(f"Tracking object {object_id} starting from frame {initial_frame_index}")
            tracked_masks = matanyone_wrapper.track_object(frames, initial_mask, initial_frame_index)
            
            # Save all tracked masks
            tracks = []
            for frame_idx, mask in tracked_masks.items():
                # Convert mask to binary
                binary_mask = np.where(mask > 127, 255, 0).astype(np.uint8)
                
                # Save the mask
                mask_path = os.path.join(masks_dir, f"{object_id}_{frame_idx:05d}.png")
                cv2.imwrite(mask_path, binary_mask)
                
                # Use a relative URL that will be handled by the frontend
                mask_url = f"/uploads/{request.video_id}/masks/{object_id}_{frame_idx:05d}.png"
                
                tracks.append({
                    "frame_index": frame_idx,
                    "mask_url": mask_url
                })
                
                print(f"Saved tracked mask for object {object_id} at frame {frame_idx}")
            
            tracked_objects.append({
                "object_id": object_id,
                "tracks": tracks
            })
        except Exception as e:
            print(f"Error tracking object {object_id}: {e}")
            # Fall back to a simple tracking
            
            # Get all frames
            tracks = []
            for i, frame in enumerate(frames):
                # Skip the initial frame because we already have it
                if i == initial_frame_index:
                    # Add the initial mask
                    tracks.append({
                        "frame_index": initial_frame_index,
                        "mask_url": f"/uploads/{request.video_id}/masks/{initial_mask_file}"
                    })
                    continue
                
                # For other frames, create a copy of the initial mask (no tracking)
                mask_path = os.path.join(masks_dir, f"{object_id}_{i:05d}.png")
                cv2.imwrite(mask_path, initial_mask)
                
                # Use a relative URL that will be handled by the frontend
                mask_url = f"/uploads/{request.video_id}/masks/{object_id}_{i:05d}.png"
                
                tracks.append({
                    "frame_index": i,
                    "mask_url": mask_url
                })
                
                print(f"Saved fallback mask for object {object_id} at frame {i}")
            
            tracked_objects.append({
                "object_id": object_id,
                "tracks": tracks
            })
    
    return {
        "video_id": request.video_id,
        "objects": tracked_objects
    }

@app.post("/apply-effect")
async def apply_effect(request: EffectRequest):
    """Apply a visual effect to an object"""
    video_dir = os.path.join("uploads", request.video_id)
    if not os.path.exists(video_dir):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get the frames
    frames_dir = os.path.join(video_dir, "frames")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    
    # Get the masks
    masks_dir = os.path.join(video_dir, "masks")
    mask_files = [f for f in os.listdir(masks_dir) if f.startswith(request.object_id)]
    
    # Create a directory for effects
    effects_dir = os.path.join(video_dir, "effects")
    os.makedirs(effects_dir, exist_ok=True)
    
    # Apply effect to each mask
    result_frames = []
    for mask_file in mask_files:
        # Get the frame index from the mask filename
        frame_index = int(mask_file.split("_")[1].split(".")[0])
        
        # Check if the frame exists
        if frame_index >= len(frame_files):
            continue
        
        # Load the frame
        frame_path = os.path.join(frames_dir, frame_files[frame_index])
        frame = cv2.imread(frame_path)
        
        # Load the mask
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        try:
            # Apply the effect based on the type
            effect_type = request.effect_type
            effect_params = request.effect_params
            
            # Create effect image (same size as frame)
            effect_img = np.zeros_like(frame)
            
            # For demonstration, we'll just apply some simple effects
            if effect_type == "blur":
                # Apply blur where the mask is
                blur_amount = effect_params.get("amount", 15)
                blurred = cv2.GaussianBlur(frame, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                effect_img = np.where(mask_3ch > 0, blurred, 0)
            
            elif effect_type == "bw":
                # Convert to grayscale where the mask is
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                effect_img = np.where(mask_3ch > 0, gray_3ch, 0)
            
            elif effect_type == "chroma":
                # Simple chromatic aberration effect
                amount = effect_params.get("amount", 5)
                
                # Split into channels
                b, g, r = cv2.split(frame)
                
                # Shift red channel
                M = np.float32([[1, 0, amount], [0, 1, 0]])
                r_shifted = cv2.warpAffine(r, M, (frame.shape[1], frame.shape[0]))
                
                # Shift blue channel
                M = np.float32([[1, 0, -amount], [0, 1, 0]])
                b_shifted = cv2.warpAffine(b, M, (frame.shape[1], frame.shape[0]))
                
                # Merge channels
                chroma = cv2.merge([b_shifted, g, r_shifted])
                
                # Apply only where the mask is
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                effect_img = np.where(mask_3ch > 0, chroma, 0)
            
            elif effect_type == "glow":
                # Simple glow effect
                amount = effect_params.get("amount", 10)
                
                # Dilate the mask
                kernel = np.ones((amount, amount), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Create a gradient from the original mask to the dilated mask
                gradient_mask = dilated_mask.copy()
                gradient_mask[mask > 0] = 0
                
                # Apply a gaussian blur to create a glow effect
                blurred = cv2.GaussianBlur(frame, (21, 21), 0)
                
                # Combine with original frame
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                gradient_mask_3ch = cv2.cvtColor(gradient_mask, cv2.COLOR_GRAY2BGR)
                
                # Use original colors for the object and blurred colors for the glow
                effect_img = np.where(mask_3ch > 0, frame, 0)
                effect_img = np.where(gradient_mask_3ch > 0, blurred, effect_img)
            
            else:
                # Unknown effect type, just use the mask as is
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                effect_img = mask_3ch
            
            # Save the effect image
            effect_filename = f"{request.object_id}_{effect_type}_{frame_index:05d}.png"
            effect_path = os.path.join(effects_dir, effect_filename)
            cv2.imwrite(effect_path, effect_img)
            
            # Use a relative URL that will be handled by the frontend
            effect_url = f"/uploads/{request.video_id}/effects/{effect_filename}"
            
            result_frames.append({
                "frame_index": frame_index,
                "effect_url": effect_url
            })
        except Exception as e:
            print(f"Error applying effect {effect_type} to frame {frame_index}: {e}")
    
    return {
        "video_id": request.video_id,
        "object_id": request.object_id,
        "effect_type": request.effect_type,
        "frames": result_frames
    }

@app.post("/export-video")
async def export_video(
    video_id: str = Form(...),
    object_ids: str = Form(...),
    effect_types: str = Form(...),
    export_type: str = Form(...)
):
    """Export a video with effects applied"""
    video_dir = os.path.join("uploads", video_id)
    if not os.path.exists(video_dir):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Parse the object IDs and effect types
    object_id_list = object_ids.split(",")
    effect_type_list = effect_types.split(",")
    
    # Get all frames
    frames_dir = os.path.join(video_dir, "frames")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    
    # Create a directory for exported videos
    exports_dir = os.path.join("results", video_id)
    os.makedirs(exports_dir, exist_ok=True)
    
    # Generate a unique filename for the output video
    output_filename = f"{export_type}_{tempfile.NamedTemporaryFile().name.split('/')[-1]}.mp4"
    output_path = os.path.join(exports_dir, output_filename)
    
    # Load the first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    fps = 30  # Assuming 30 fps
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        # Process each frame
        for i, frame_file in enumerate(frame_files):
            # Load the original frame
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if export_type == "mask":
                # Export with visible masks
                result = frame.copy()
                
                # Apply each object's mask
                for j, object_id in enumerate(object_id_list):
                    # Look for this frame's mask
                    mask_path = os.path.join(video_dir, "masks", f"{object_id}_{i:05d}.png")
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        
                        # Convert to color mask using the object's color
                        color_index = j % len([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)])
                        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)][color_index]
                        
                        colored_mask = np.zeros_like(frame)
                        colored_mask[:, :, 0] = mask // 255 * color[0]
                        colored_mask[:, :, 1] = mask // 255 * color[1]
                        colored_mask[:, :, 2] = mask // 255 * color[2]
                        
                        # Blend with the original frame
                        alpha = 0.5  # Transparency level
                        result = cv2.addWeighted(result, 1, colored_mask, alpha, 0)
                
                video_writer.write(result)
            
            elif export_type == "fx":
                # Export with effects applied
                result = frame.copy()
                
                # Apply each object's effect
                for j, (object_id, effect_type) in enumerate(zip(object_id_list, effect_type_list)):
                    # Look for this frame's effect
                    effect_path = os.path.join(video_dir, "effects", f"{object_id}_{effect_type}_{i:05d}.png")
                    if os.path.exists(effect_path):
                        effect = cv2.imread(effect_path)
                        
                        # Add the effect to the result
                        if effect is not None and effect.shape == result.shape:
                            # Only apply the effect where it's not black (0, 0, 0)
                            mask = np.all(effect > 0, axis=2).astype(np.uint8) * 255
                            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                            
                            # Blend the effect with the result
                            result = np.where(mask_3ch > 0, effect, result)
                
                video_writer.write(result)
        
        video_writer.release()
        
        # Return the URL to the exported video
        # Use a relative URL that will be handled by the frontend
        video_url = f"/results/{video_id}/{output_filename}"
        
        return {
            "video_id": video_id,
            "export_type": export_type,
            "video_url": video_url
        }
    except Exception as e:
        video_writer.release()
        raise HTTPException(status_code=500, detail=f"Error exporting video: {str(e)}")

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