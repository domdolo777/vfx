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

# Source config files
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
    def track_object(self, frames: List[np.ndarray], first_frame_mask: np.ndarray, n_warmup: int = 10) -> List[np.ndarray]:
        """Track an object across frames using the first frame mask"""
        # Try to initialize MatAnyone
        if not self.initialized:
            # If MatAnyone initialization failed, use a simple fallback tracking
            print("Using fallback tracking")
            return self.create_fallback_tracking(frames, first_frame_mask)
        
        try:
            # Prepare frames
            frame_tensors = []
            for frame in frames:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.to(self.device)
                frame_tensors.append(frame_tensor)
            
            # Add warmup frames (repeat first frame)
            warmup_frames = [frame_tensors[0]] * n_warmup
            frame_tensors = warmup_frames + frame_tensors
            
            # Convert mask to tensor
            mask_tensor = torch.from_numpy(first_frame_mask).float().to(self.device)
            
            # Process with MatAnyone
            objects = [1]  # Single object
            masks = []
            
            for ti, frame_tensor in enumerate(frame_tensors):
                if ti == 0:
                    output_prob = self.processor.step(frame_tensor, mask_tensor, objects=objects)
                    output_prob = self.processor.step(frame_tensor, first_frame_pred=True)
                else:
                    if ti <= n_warmup:
                        output_prob = self.processor.step(frame_tensor, first_frame_pred=True)
                    else:
                        output_prob = self.processor.step(frame_tensor)
                
                # Convert output to mask
                result_mask = self.processor.output_prob_to_mask(output_prob)
                
                # Skip warmup frames
                if ti > (n_warmup - 1):
                    # Convert to numpy
                    result_mask = result_mask.cpu().numpy() * 255
                    result_mask = result_mask.astype(np.uint8)
                    masks.append(result_mask)
            
            return masks
        except Exception as e:
            print(f"Error during tracking: {e}")
            return self.create_fallback_tracking(frames, first_frame_mask)

# Create a singleton instance
matanyone_wrapper = MatAnyoneWrapper()

# Initialize the model when the module is imported
print("Attempting to initialize MatAnyone on startup...")
if matanyone_wrapper.initialize():
    print("MatAnyone initialized successfully on startup")
else:
    print("Failed to initialize MatAnyone on startup, will use fallback segmentation") 