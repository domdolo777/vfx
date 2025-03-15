import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import shutil
from omegaconf import open_dict, OmegaConf
import yaml

# Add MatAnyone to the Python path
matanyone_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MatAnyone"))
sys.path.insert(0, matanyone_path)
print(f"Added MatAnyone path: {matanyone_path}")

# Flag to track if imports were successful
matanyone_imports_successful = False

try:
    # Import MatAnyone modules
    from matanyone.model.matanyone import MatAnyone
    from matanyone.inference.inference_core import InferenceCore
    # Import the get_matanyone_model function
    from matanyone.utils.get_default_model import get_matanyone_model
    matanyone_imports_successful = True
    print("Successfully imported MatAnyone modules")
except ImportError as e:
    print(f"Failed to import MatAnyone modules: {e}")
    print("Make sure the MatAnyone repository is correctly installed")

# Copy the original configuration files to our location
config_dir = os.path.join(os.path.dirname(__file__), "matanyone_config")
os.makedirs(config_dir, exist_ok=True)

# Source config files - using relative paths for better portability
original_eval_config = os.path.join(matanyone_path, "matanyone", "config", "eval_matanyone_config.yaml")
original_base_config = os.path.join(matanyone_path, "matanyone", "config", "model", "base.yaml")

# Target config files
local_eval_config = os.path.join(config_dir, "eval_matanyone_config.yaml")
local_base_config = os.path.join(config_dir, "base.yaml")

# Copy the configuration files if they don't exist
if not os.path.exists(local_eval_config) and os.path.exists(original_eval_config):
    print(f"Copying {original_eval_config} to {local_eval_config}")
    shutil.copy(original_eval_config, local_eval_config)
else:
    print(f"Using existing config file at {local_eval_config}" if os.path.exists(local_eval_config) else f"Could not find original config at {original_eval_config}")

# Create model directory if it doesn't exist
os.makedirs(os.path.join(config_dir, "model"), exist_ok=True)

if not os.path.exists(os.path.join(config_dir, "model", "base.yaml")) and os.path.exists(original_base_config):
    print(f"Copying {original_base_config} to {os.path.join(config_dir, 'model', 'base.yaml')}")
    shutil.copy(original_base_config, os.path.join(config_dir, "model", "base.yaml"))
else:
    print(f"Using existing config file at {os.path.join(config_dir, 'model', 'base.yaml')}" if os.path.exists(os.path.join(config_dir, "model", "base.yaml")) else f"Could not find original config at {original_base_config}")

class MatAnyoneWrapper:
    def __init__(self):
        """Initialize the MatAnyone model and processor"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.initialized = False
        # Use the correct path for the model file according to MatAnyone documentation
        self.model_path = os.path.join(matanyone_path, "pretrained_models", "matanyone.pth")
        print(f"Model path set to: {self.model_path}")
    
    def initialize(self):
        """Load the MatAnyone model"""
        if not self.initialized:
            try:
                # Check if we have the needed modules
                if not matanyone_imports_successful:
                    print("MatAnyone modules not properly imported. Using fallback.")
                    return False
                
                # Check if the model file exists
                if not os.path.exists(self.model_path):
                    print(f"Model file not found at {self.model_path}")
                    print("Using fallback segmentation")
                    return False
                
                print(f"Loading MatAnyone model from: {self.model_path}")
                
                try:
                    # Load the original configuration files
                    if os.path.exists(local_eval_config) and os.path.exists(os.path.join(config_dir, "model", "base.yaml")):
                        print("Using original MatAnyone configuration files")
                        
                        # Load the base model config
                        with open(os.path.join(config_dir, "model", "base.yaml"), 'r') as f:
                            base_config = yaml.safe_load(f)
                        
                        # Load the eval config
                        with open(local_eval_config, 'r') as f:
                            eval_config = yaml.safe_load(f)
                        
                        # Create a complete configuration
                        # We need to keep ALL the original parameters
                        config = eval_config.copy()
                        # Override the model with the base config
                        config['model'] = base_config
                        # Make sure we have the weights path
                        config['weights'] = self.model_path
                        
                        # Fix the pixel_encoder configuration
                        # The model weights expect specific dimensions
                        if 'model' in config and 'pixel_encoder' in config['model']:
                            config['model']['pixel_encoder']['type'] = 'resnet50'
                            # Use the exact dimensions from the error message
                            config['model']['pixel_encoder']['ms_dims'] = [1024, 512, 256, 64, 3]  # Dimensions from the base.yaml file
                        
                        # Remove any top-level pixel_encoder to avoid confusion
                        if 'pixel_encoder' in config:
                            del config['pixel_encoder']
                        
                        # Add all required parameters for InferenceCore
                        # These are the parameters that the error messages have shown are required
                        required_params = {
                            'mem_every': 5,
                            'max_mem_frames': 5,
                            'stagger_updates': 5,
                            'top_k': 30,
                            'chunk_size': -1,
                            'max_internal_size': -1,
                            'save_aux': False,
                            'save_scores': False,
                            'visualize': False,
                            'flip_aug': False,
                            'use_long_term': False,
                            'use_all_masks': False,
                            'save_all': True,
                            'amp': False,
                            'output_dir': None
                        }
                        
                        # Add all required parameters
                        for key, value in required_params.items():
                            if key not in config:
                                config[key] = value
                        
                        # Make sure long_term config is present
                        if 'long_term' not in config:
                            config['long_term'] = {
                                'count_usage': True,
                                'max_mem_frames': 10,
                                'min_mem_frames': 5,
                                'num_prototypes': 128,
                                'max_num_tokens': 10000,
                                'buffer_tokens': 2000
                            }
                        
                        # Convert to OmegaConf object
                        cfg = OmegaConf.create(config)
                        
                        print("Configuration:", OmegaConf.to_yaml(cfg))
                        
                        # Initialize MatAnyone model directly
                        print("Initializing MatAnyone model directly...")
                        self.model = MatAnyone(cfg, single_object=True).to(self.device).eval()
                        
                        # Load model weights
                        print(f"Loading model weights from: {self.model_path}")
                        model_weights = torch.load(self.model_path, map_location=self.device)
                        self.model.load_weights(model_weights)
                        
                        # Initialize the processor with the model
                        print("Creating inference processor...")
                        self.processor = InferenceCore(self.model, cfg=cfg)
                        
                        self.initialized = True
                        print("MatAnyone model loaded successfully")
                        return True
                    else:
                        print("Original configuration files not found. Using fallback segmentation.")
                        return False
                except Exception as e:
                    print(f"Error initializing MatAnyone model directly: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Using fallback segmentation instead")
                    return False
            except Exception as e:
                print(f"Error loading MatAnyone model: {e}")
                print("Using fallback segmentation instead")
                return False
        return self.initialized
    
    def create_fallback_mask(self, frame: np.ndarray, points: List[List[float]]) -> np.ndarray:
        """Create a fallback mask based on the points"""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if len(points) > 0:
            # Calculate the center of all points
            center_x = int(sum(p[0] for p in points) / len(points))
            center_y = int(sum(p[1] for p in points) / len(points))
            
            # Calculate the average distance from center to points
            distances = [np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in points]
            avg_distance = int(sum(distances) / len(distances)) if distances else 50
            
            # Create a circle around the center
            radius = max(avg_distance, 50)  # Minimum radius of 50 pixels
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        else:
            # No points provided, create a default circle in the center
            center_x, center_y = width // 2, height // 2
            cv2.circle(mask, (center_x, center_y), 100, 255, -1)
        
        return mask
    
    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def segment_frame(self, frame, points, labels):
        """
        Segment a frame using MatAnyone
        
        Args:
            frame: The input frame (numpy array)
            points: List of points [[x1, y1], [x2, y2], ...]
            labels: List of labels [1, 1, 0, ...] (1 for foreground, 0 for background)
            
        Returns:
            mask: Binary mask (numpy array)
        """
        # Try to initialize the model if not already initialized
        if not self.initialized:
            self.initialize()
            
        # Check again if initialization succeeded
        if not self.initialized:
            print("MatAnyone model not loaded, using fallback segmentation")
            return self.create_fallback_mask(frame, points)
        
        try:
            # Create a proper segmentation mask from the points
            height, width = frame.shape[:2]
            
            # First, create a mask with the points
            point_mask = np.zeros((height, width), dtype=np.uint8)
            for point, label in zip(points, labels):
                x, y = int(point[0]), int(point[1])
                if 0 <= x < width and 0 <= y < height:
                    # Draw a circle for each point
                    radius = 10
                    color = 255 if label == 1 else 128
                    cv2.circle(point_mask, (x, y), radius, color, -1)
            
            # Create a more refined mask using GrabCut algorithm
            # This will create a better initial mask for MatAnyone
            try:
                # Convert frame to RGB for GrabCut
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create initial mask for GrabCut
                # 0 = background, 1 = foreground, 2 = probable background, 3 = probable foreground
                grabcut_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Mark points as definite foreground or background
                for point, label in zip(points, labels):
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < width and 0 <= y < height:
                        # 0 = background, 1 = foreground
                        value = 1 if label == 1 else 0
                        # Convert to GrabCut values: 0 = background, 1 = foreground
                        grabcut_value = cv2.GC_FGD if value == 1 else cv2.GC_BGD
                        cv2.circle(grabcut_mask, (x, y), 30, grabcut_value, -1)  # Increased radius
                
                # Mark the rest as probable background
                grabcut_mask[grabcut_mask == 0] = cv2.GC_PR_BGD
                
                # Create a rectangle around all foreground points
                if np.any(grabcut_mask == cv2.GC_FGD):
                    fg_points = np.where(grabcut_mask == cv2.GC_FGD)
                    min_y, max_y = np.min(fg_points[0]), np.max(fg_points[0])
                    min_x, max_x = np.min(fg_points[1]), np.max(fg_points[1])
                    
                    # Expand the rectangle
                    padding = 100  # Increased padding
                    min_y = max(0, min_y - padding)
                    max_y = min(height - 1, max_y + padding)
                    min_x = max(0, min_x - padding)
                    max_x = min(width - 1, max_x + padding)
                    
                    # Mark the rectangle as probable foreground
                    grabcut_mask[min_y:max_y, min_x:max_x] = cv2.GC_PR_FGD
                
                # Run GrabCut algorithm
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Run for more iterations
                try:
                    cv2.grabCut(frame_rgb, grabcut_mask, None, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK)
                except Exception as e:
                    print(f"Error in GrabCut: {e}")
                
                # Create mask from GrabCut result
                refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                
                # Apply morphological operations to clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
                
                # If GrabCut didn't produce a good mask, fall back to dilated points
                if np.sum(refined_mask) < 1000:  # Increased threshold
                    print("GrabCut produced an empty or small mask, using dilated points instead")
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))  # Increased kernel size
                    refined_mask = cv2.dilate(point_mask, kernel, iterations=3)  # More iterations
            except Exception as e:
                print(f"Error in GrabCut segmentation: {e}")
                # Fall back to dilated points
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))  # Increased kernel size
                refined_mask = cv2.dilate(point_mask, kernel, iterations=3)  # More iterations
            
            # Convert frame to tensor (HWC -> CHW, normalize)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Convert mask to tensor
            mask_tensor = torch.from_numpy(refined_mask).float().to(self.device) / 255.0
            
            # Process the frame with MatAnyone
            with torch.no_grad():
                # Initialize with the first frame
                objects = [1]  # Single object
                
                try:
                    # Process the image using the refined mask
                    output_prob = self.processor.step(
                        frame_tensor, 
                        mask_tensor,
                        objects=objects
                    )
                    
                    # Get the mask
                    result_mask = self.processor.output_prob_to_mask(output_prob)
                    
                    # Convert tensor to numpy array
                    mask_np = result_mask.cpu().numpy()
                    
                    # Ensure mask is binary (0 or 255)
                    mask_binary = np.where(mask_np > 0.5, 255, 0).astype(np.uint8)
                    
                    print(f"Generated mask with shape: {mask_binary.shape}, dtype: {mask_binary.dtype}")
                    print(f"Mask min: {mask_binary.min()}, max: {mask_binary.max()}")
                    
                    # Check if the mask is empty (all zeros)
                    if mask_binary.max() == 0:
                        print("MatAnyone generated an empty mask, using refined mask instead")
                        return refined_mask
                    
                    return mask_binary
                except Exception as e:
                    print(f"Error in processor.step: {e}")
                    print("Using refined mask instead")
                    return refined_mask
        except Exception as e:
            print(f"Error in segment_frame: {e}")
            print(f"Exception details: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.create_fallback_mask(frame, points)
    
    def create_fallback_tracking(self, frames: List[np.ndarray], first_frame_mask: np.ndarray) -> List[np.ndarray]:
        """Create a fallback tracking by using the first frame mask with slight variations"""
        masks = []
        
        for i, frame in enumerate(frames):
            # Add some random variation to simulate tracking
            mask = first_frame_mask.copy()
            if i > 0:
                # Shift the mask slightly based on the frame index
                M = np.float32([[1, 0, (i % 10) - 5], [0, 1, (i % 8) - 4]])
                mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            
            masks.append(mask)
        
        return masks
    
    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def track_object(self, frames: List[np.ndarray], first_frame_mask: np.ndarray, initial_frame_index: int = 0) -> Dict[int, np.ndarray]:
        """
        Track the object with the mask across all frames using MatAnyone's processor
        
        Args:
            frames: List of frames
            first_frame_mask: Binary mask for the first frame
            initial_frame_index: Index of the frame that contains the first mask
            
        Returns:
            Dictionary of frame indices to masks
        """
        # Check if MatAnyone is initialized
        if not self.initialized:
            self.initialize()
        
        # If still not initialized, fall back to optical flow
        if not self.initialized:
            print("MatAnyone not initialized, falling back to optical flow tracking")
            return self._track_with_optical_flow(frames, first_frame_mask, initial_frame_index)
        
        print("Using MatAnyone for tracking")
        result = {}
        
        try:
            # Convert the mask to a binary mask if it's not already
            binary_mask = np.where(first_frame_mask > 127, 255, 0).astype(np.uint8)
            
            # Process frames in order
            # We need to normalize frames for MatAnyone
            frame_tensors = []
            for frame in frames:
                # Convert to RGB and normalize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.to(self.device)
                frame_tensors.append(frame_tensor)
            
            # Convert mask to tensor
            mask_tensor = torch.from_numpy(binary_mask).float().to(self.device) / 255.0
            
            # Record the result for initial frame
            result[initial_frame_index] = binary_mask
            
            # Start with the initial frame - this initializes the processor with our mask
            objects = [1]  # Single object
            
            # Reset the processor state to start fresh
            self.processor.reset()
                
            # First, process the initial frame with the mask
            print(f"Initializing tracking with frame {initial_frame_index}")
            try:
                # Initialize with mask on the first frame
                output_prob = self.processor.step(
                    frame_tensors[initial_frame_index],
                    mask_tensor,
                    objects=objects
                )
            except Exception as e:
                print(f"Error initializing tracking: {e}")
                return self._track_with_optical_flow(frames, first_frame_mask, initial_frame_index)
            
            # Now track forward from the initial frame
            for i in range(initial_frame_index + 1, len(frames)):
                try:
                    # Process the next frame
                    output_prob = self.processor.step(frame_tensors[i])
                    
                    # Convert output probability to mask
                    result_mask = self.processor.output_prob_to_mask(output_prob)
                    
                    # Convert tensor to numpy array
                    mask_np = result_mask.cpu().numpy()
                    
                    # Ensure mask is binary (0 or 255)
                    mask_binary = np.where(mask_np > 0.5, 255, 0).astype(np.uint8)
                    
                    # Store the result
                    result[i] = mask_binary
                    print(f"Tracked to frame {i} successfully")
                    
                except Exception as e:
                    print(f"Error tracking to frame {i}: {e}")
                    # If tracking fails, use optical flow for the rest
                    remaining_frames = frames[i:]
                    optical_flow_results = self._track_with_optical_flow(
                        remaining_frames, 
                        result[i-1] if i-1 in result else first_frame_mask,
                        0  # Start from the first frame in the remaining frames
                    )
                    
                    # Merge the results
                    for j, frame_idx in enumerate(range(i, len(frames))):
                        if j in optical_flow_results:
                            result[frame_idx] = optical_flow_results[j]
                    
                    # Stop the loop as we've handled the remaining frames
                    break
            
            # Now track backward from the initial frame if needed
            if initial_frame_index > 0:
                # Reset the processor for backward tracking
                self.processor.reset()
                
                # Initialize with the initial frame and mask again
                try:
                    output_prob = self.processor.step(
                        frame_tensors[initial_frame_index],
                        mask_tensor,
                        objects=objects
                    )
                except Exception as e:
                    print(f"Error initializing backward tracking: {e}")
                    # Use optical flow for backward frames
                    backward_frames = frames[:initial_frame_index+1]
                    backward_frames.reverse()  # Reverse to track backward
                    optical_flow_results = self._track_with_optical_flow(
                        backward_frames,
                        first_frame_mask,
                        0  # Initial frame is now at index 0 in the reversed list
                    )
                    
                    # Merge the results, adjusting indices
                    for j in range(initial_frame_index):
                        if initial_frame_index-j-1 in optical_flow_results:
                            result[j] = optical_flow_results[initial_frame_index-j-1]
                    
                    return result
                
                # Track backward
                for i in range(initial_frame_index - 1, -1, -1):
                    try:
                        # Process the previous frame
                        output_prob = self.processor.step(frame_tensors[i])
                        
                        # Convert output probability to mask
                        result_mask = self.processor.output_prob_to_mask(output_prob)
                        
                        # Convert tensor to numpy array
                        mask_np = result_mask.cpu().numpy()
                        
                        # Ensure mask is binary (0 or 255)
                        mask_binary = np.where(mask_np > 0.5, 255, 0).astype(np.uint8)
                        
                        # Store the result
                        result[i] = mask_binary
                        print(f"Tracked back to frame {i} successfully")
                        
                    except Exception as e:
                        print(f"Error tracking back to frame {i}: {e}")
                        # If tracking fails, use optical flow for the remaining backward frames
                        backward_frames = frames[:i+1]
                        backward_frames.reverse()  # Reverse to track backward
                        optical_flow_results = self._track_with_optical_flow(
                            backward_frames,
                            result[i+1] if i+1 in result else first_frame_mask,
                            0  # Initial frame is now at index 0 in the reversed list
                        )
                        
                        # Merge the results, adjusting indices
                        for j in range(i+1):
                            if i-j in optical_flow_results:
                                result[j] = optical_flow_results[i-j]
                        
                        # Stop the loop as we've handled the remaining backward frames
                        break
            
            return result
        
        except Exception as e:
            print(f"Error in MatAnyone tracking: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to optical flow tracking")
            return self._track_with_optical_flow(frames, first_frame_mask, initial_frame_index)

    def _track_with_optical_flow(self, frames: List[np.ndarray], first_frame_mask: np.ndarray, initial_frame_index: int = 0) -> Dict[int, np.ndarray]:
        """
        Track the object with optical flow as a fallback method
        
        Args:
            frames: List of frames
            first_frame_mask: Binary mask for the first frame
            initial_frame_index: Index of the frame that contains the first mask
            
        Returns:
            Dictionary of frame indices to masks
        """
        print(f"Using optical flow tracking as fallback")
        result = {}
        
        # Convert the mask to a binary mask if it's not already
        binary_mask = np.where(first_frame_mask > 127, 255, 0).astype(np.uint8)
        
        # Find contours in the mask to get the bounding box
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return the original mask for all frames
        if not contours:
            print("No contours found in mask, using original mask for all frames")
            return {i: binary_mask.copy() for i in range(len(frames))}
        
        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box from the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Ensure the bounding box is valid
        if w <= 0 or h <= 0:
            print("Invalid bounding box, using original mask for all frames")
            return {i: binary_mask.copy() for i in range(len(frames))}
        
        # Initialize sparse optical flow
        initial_frame_gray = cv2.cvtColor(frames[initial_frame_index], cv2.COLOR_BGR2GRAY)
        
        # Sample points from within the mask for tracking
        mask_indices = np.where(binary_mask > 0)
        
        # If mask is empty, fallback to original implementation
        if len(mask_indices[0]) == 0:
            print("Empty mask, using original mask for all frames")
            return {i: binary_mask.copy() for i in range(len(frames))}
        
        # Sample up to 100 points from the mask
        max_points = min(100, len(mask_indices[0]))
        sample_indices = np.random.choice(len(mask_indices[0]), max_points, replace=False)
        
        # Create points array for optical flow
        points = np.array([[mask_indices[1][i], mask_indices[0][i]] for i in sample_indices], dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        
        # First save the original mask for the initial frame
        result[initial_frame_index] = binary_mask.copy()
        
        # Track forward
        prev_gray = initial_frame_gray
        prev_points = points.copy()
        current_mask = binary_mask.copy()
        
        for i in range(initial_frame_index + 1, len(frames)):
            current_frame = frames[i]
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, None)
            
            # If no points were tracked successfully, use the previous mask
            if new_points is None or np.sum(status) == 0:
                print(f"No points tracked at frame {i}, using previous mask")
                result[i] = current_mask.copy()
                continue
            
            # Select good points
            good_old = prev_points[status == 1]
            good_new = new_points[status == 1]
            
            if len(good_old) == 0 or len(good_new) == 0:
                print(f"No good points at frame {i}, using previous mask")
                result[i] = current_mask.copy()
                continue
            
            # Calculate the average movement of all tracked points
            dx = np.mean(good_new[:, 0, 0] - good_old[:, 0, 0])
            dy = np.mean(good_new[:, 0, 1] - good_old[:, 0, 1])
            
            # Apply translation to the mask
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            current_mask = cv2.warpAffine(current_mask, M, (current_mask.shape[1], current_mask.shape[0]))
            
            # Save the result
            result[i] = current_mask.copy()
            
            # Update variables for next iteration
            prev_gray = current_gray
            prev_points = good_new.reshape(-1, 1, 2)
        
        # Track backward if needed
        if initial_frame_index > 0:
            prev_gray = initial_frame_gray
            prev_points = points.copy()
            current_mask = binary_mask.copy()
            
            for i in range(initial_frame_index - 1, -1, -1):
                current_frame = frames[i]
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow (backward)
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, None)
                
                # If no points were tracked successfully, use the previous mask
                if new_points is None or np.sum(status) == 0:
                    print(f"No points tracked at backward frame {i}, using previous mask")
                    result[i] = current_mask.copy()
                    continue
                
                # Select good points
                good_old = prev_points[status == 1]
                good_new = new_points[status == 1]
                
                if len(good_old) == 0 or len(good_new) == 0:
                    print(f"No good points at backward frame {i}, using previous mask")
                    result[i] = current_mask.copy()
                    continue
                
                # Calculate the average movement of all tracked points
                dx = np.mean(good_new[:, 0, 0] - good_old[:, 0, 0])
                dy = np.mean(good_new[:, 0, 1] - good_old[:, 0, 1])
                
                # Apply translation to the mask
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                current_mask = cv2.warpAffine(current_mask, M, (current_mask.shape[1], current_mask.shape[0]))
                
                # Save the result
                result[i] = current_mask.copy()
                
                # Update variables for next iteration
                prev_gray = current_gray
                prev_points = good_new.reshape(-1, 1, 2)
        
        return result

# Create a singleton instance
matanyone_wrapper = MatAnyoneWrapper()

# Initialize the model when the module is imported
print("Attempting to initialize MatAnyone on startup...")
if matanyone_wrapper.initialize():
    print("MatAnyone initialized successfully on startup")
else:
    print("Failed to initialize MatAnyone on startup, will use fallback segmentation") 