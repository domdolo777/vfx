import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Paper,
  Slider,
  IconButton,
  CircularProgress,
  Alert,
  Grid,
  Divider,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Checkbox,
  FormControlLabel,
  TextField
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import DeleteIcon from '@mui/icons-material/Delete';
import SaveIcon from '@mui/icons-material/Save';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import MovieIcon from '@mui/icons-material/Movie';
import VideocamIcon from '@mui/icons-material/Videocam';
import SegmentIcon from '@mui/icons-material/Segment';
import axios from 'axios';
import config from '../config';

// Define colors for object highlighting
const objectColors = [
  '#FF5252', // Red
  '#4CAF50', // Green
  '#2196F3', // Blue
  '#FFC107', // Amber
  '#9C27B0', // Purple
  '#00BCD4', // Cyan
  '#FF9800', // Orange
  '#795548'  // Brown
];

const EFFECT_TYPES = [
  { id: 'blur', name: 'Blur', params: { amount: { default: 15, min: 1, max: 50, step: 1 } } },
  { id: 'bw', name: 'Black & White', params: {} },
  { id: 'chroma', name: 'Chromatic Aberration', params: { amount: { default: 5, min: 1, max: 20, step: 1 } } },
  { id: 'glow', name: 'Glow', params: { amount: { default: 10, min: 1, max: 30, step: 1 } } }
];

// Add the missing applyEffect function
const applyEffect = (ctx, maskImg, effectType, params) => {
  // This is a simplified implementation for demonstration
  // In a real application, you would implement actual visual effects
  
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  
  // Create a temporary canvas to apply the effect
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  const tempCtx = tempCanvas.getContext('2d');
  
  // Draw the mask
  tempCtx.drawImage(maskImg, 0, 0);
  
  // Get the mask data
  const maskData = tempCtx.getImageData(0, 0, width, height);
  
  // Get the current image data
  const imageData = ctx.getImageData(0, 0, width, height);
  
  // Apply the effect based on the mask
  for (let i = 0; i < maskData.data.length; i += 4) {
    const alpha = maskData.data[i + 3]; // Alpha channel of the mask
    
    if (alpha > 0) {
      // Only apply effect where the mask is visible
      const r = imageData.data[i];
      const g = imageData.data[i + 1];
      const b = imageData.data[i + 2];
      
      switch (effectType) {
        case 'blur':
          // Simple blur effect (just for demonstration)
          // In a real app, you would implement a proper blur algorithm
          imageData.data[i] = r;
          imageData.data[i + 1] = g;
          imageData.data[i + 2] = b;
          break;
          
        case 'bw':
          // Convert to black and white
          const gray = 0.3 * r + 0.59 * g + 0.11 * b;
          imageData.data[i] = gray;
          imageData.data[i + 1] = gray;
          imageData.data[i + 2] = gray;
          break;
          
        case 'chroma':
          // Simple chromatic aberration (just for demonstration)
          imageData.data[i] = r; // Red channel stays
          imageData.data[i + 1] = g; // Green channel stays
          imageData.data[i + 2] = b; // Blue channel stays
          break;
          
        case 'glow':
          // Simple glow effect (just for demonstration)
          const amount = params?.amount || 10;
          imageData.data[i] = Math.min(255, r + amount);
          imageData.data[i + 1] = Math.min(255, g + amount);
          imageData.data[i + 2] = Math.min(255, b + amount);
          break;
          
        default:
          break;
      }
    }
  }
  
  // Put the modified image data back
  ctx.putImageData(imageData, 0, 0);
};

const FXMode = () => {
  const { videoId } = useParams();
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const [videoInfo, setVideoInfo] = useState(null);
  const [frames, setFrames] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [objects, setObjects] = useState([]);
  const [selectedObjectIndex, setSelectedObjectIndex] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [effectsInProgress, setEffectsInProgress] = useState(false);
  const [exportInProgress, setExportInProgress] = useState(false);
  
  // FX parameters
  const [selectedEffect, setSelectedEffect] = useState('');
  const [effectParams, setEffectParams] = useState({});
  const [globalParams, setGlobalParams] = useState({
    maskOpacity: 50,
    featherRadius: 5,
    featherExpand: 0,
    fxOpacity: 100,
    invertFX: false
  });
  const [presets, setPresets] = useState([]);
  const [presetName, setPresetName] = useState('');

  // Load video info and frames
  useEffect(() => {
    const fetchVideoInfo = async () => {
      try {
        const response = await axios.get(`${config.apiUrl}/video-frames/${videoId}?start=0&count=30`);
        setVideoInfo({
          videoId: response.data.video_id,
          totalFrames: response.data.total_frames
        });
        setFrames(response.data.frames);
      } catch (error) {
        console.error('Error fetching video info:', error);
        setError('Failed to load video information');
      }
    };

    fetchVideoInfo();
  }, [videoId]);

  // Handle video playback
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentFrameIndex((prevIndex) => {
          const nextIndex = prevIndex + 1;
          if (nextIndex >= frames.length) {
            setIsPlaying(false);
            return prevIndex;
          }
          return nextIndex;
        });
      }, 1000 / 30); // Assuming 30 fps
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, frames.length]);

  // Draw the current frame on the canvas
  useEffect(() => {
    const renderFrame = async () => {
      try {
        if (frames.length > 0 && currentFrameIndex < frames.length) {
          const frameUrl = frames[currentFrameIndex].url;
          const fullUrl = frameUrl.startsWith('http') 
            ? frameUrl 
            : `${config.apiUrl}${frameUrl}`;
          
          console.log('Loading frame', currentFrameIndex, 'from', fullUrl);
          
          const img = new Image();
          img.crossOrigin = "anonymous"; // Critical for CORS handling
          img.src = fullUrl;
          
          // Force debug flag for testing
          const DEBUG_MASKS = true;
          
          img.onload = () => {
            console.log('Frame loaded successfully:', fullUrl);
            const canvas = canvasRef.current;
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            
            // Clear the canvas first
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw the base frame
            ctx.drawImage(img, 0, 0);
            
            // Debug log the objects
            console.log('Objects to draw:', objects);
            
            // Draw debug info on canvas
            if (DEBUG_MASKS) {
              ctx.font = '20px Arial';
              ctx.fillStyle = 'white';
              ctx.fillText(`Frame: ${currentFrameIndex}`, 20, 30);
              ctx.fillText(`Objects: ${objects ? objects.length : 0}`, 20, 60);
              if (objects && objects.length > 0) {
                objects.forEach((obj, i) => {
                  ctx.fillText(`Object ${i+1}: ${obj.id} - Visible: ${obj.visible}`, 20, 90 + i * 30);
                  ctx.fillText(`  Has mask for current frame: ${obj.masks && obj.masks[currentFrameIndex] ? 'YES' : 'NO'}`, 40, 120 + i * 30);
                });
              }
            }
            
            // Draw masks and effects for the current frame
            if (objects && objects.length > 0) {
              // We'll use Promise.all to wait for all mask loading operations
              const maskPromises = objects.map((object, index) => {
                return new Promise(async (resolve) => {
                  try {
                    console.log(`Processing object ${index}:`, object);
                    
                    if (!object.visible || !object.masks) {
                      console.log(`Object ${object.id} is not visible or has no masks`);
                      return resolve();
                    }
                    
                    // Check if we have a mask for this frame
                    const maskUrl = object.masks[currentFrameIndex];
                    if (!maskUrl) {
                      console.log(`No mask found for object ${object.id} at frame ${currentFrameIndex}`);
                      return resolve();
                    }
                    
                    const fullMaskUrl = maskUrl.startsWith('http') 
                      ? maskUrl 
                      : `${config.apiUrl}${maskUrl}`;
                    
                    console.log('Loading mask for object', index, 'from', fullMaskUrl);
                    
                    try {
                      // Check if mask exists via fetch HEAD request
                      const checkResponse = await fetch(fullMaskUrl, { method: 'HEAD' });
                      console.log(`Mask URL ${fullMaskUrl} status: ${checkResponse.status}`);
                      
                      if (checkResponse.status !== 200) {
                        throw new Error(`Mask not found: ${checkResponse.status}`);
                      }
                    } catch (e) {
                      console.error(`Error checking mask URL: ${e.message}`);
                    }
                    
                    const maskImg = new Image();
                    maskImg.crossOrigin = "anonymous"; // Critical for CORS handling
                    
                    // Set up onload handler before setting src
                    maskImg.onload = () => {
                      console.log('Mask loaded successfully:', fullMaskUrl);
                      
                      // Create a temporary canvas for the mask
                      const tempCanvas = document.createElement('canvas');
                      tempCanvas.width = canvas.width;
                      tempCanvas.height = canvas.height;
                      const tempCtx = tempCanvas.getContext('2d');
                      
                      // Draw the mask on the temporary canvas
                      tempCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
                      
                      // Get the mask data
                      const maskData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
                      
                      // Debug: Log mask data stats
                      let nonZeroPixels = 0;
                      for (let i = 0; i < maskData.data.length; i += 4) {
                        if (maskData.data[i] > 0 || maskData.data[i+1] > 0 || 
                            maskData.data[i+2] > 0 || maskData.data[i+3] > 0) {
                          nonZeroPixels++;
                        }
                      }
                      console.log(`Mask has ${nonZeroPixels} non-zero pixels out of ${maskData.data.length/4} total pixels`);
                      
                      // Create a colored version of the mask
                      const coloredMaskData = new ImageData(canvas.width, canvas.height);
                      const objectColor = objectColors[index % objectColors.length];
                      
                      // Parse the hex color to RGB
                      const r = parseInt(objectColor.slice(1, 3), 16);
                      const g = parseInt(objectColor.slice(3, 5), 16);
                      const b = parseInt(objectColor.slice(5, 7), 16);
                      
                      console.log(`Using color ${objectColor} (${r},${g},${b}) for object ${object.id}`);
                      
                      // Apply the color to the mask with the global opacity setting
                      // For debugging, use high opacity to make mask clearly visible
                      const opacity = DEBUG_MASKS ? 0.9 : (globalParams.maskOpacity / 100);
                      
                      for (let i = 0; i < maskData.data.length; i += 4) {
                        // Check if ANY channel has a value (not just the first one)
                        if (maskData.data[i] > 0 || maskData.data[i+1] > 0 || 
                            maskData.data[i+2] > 0 || maskData.data[i+3] > 0) {
                          coloredMaskData.data[i] = r;     // R
                          coloredMaskData.data[i + 1] = g; // G
                          coloredMaskData.data[i + 2] = b; // B
                          coloredMaskData.data[i + 3] = Math.round(255 * opacity); // Alpha based on global setting or debug value
                        }
                      }
                      
                      // If debug and no nonZeroPixels, add a visible square for debugging
                      if (DEBUG_MASKS && nonZeroPixels < 10) {
                        console.log("WARNING: Mask has very few non-zero pixels, adding debug rectangle");
                        // Add a visible rectangle to show the mask location
                        const centerX = Math.floor(canvas.width / 2);
                        const centerY = Math.floor(canvas.height / 2);
                        const size = 100;
                        
                        for (let y = centerY - size/2; y < centerY + size/2; y++) {
                          for (let x = centerX - size/2; x < centerX + size/2; x++) {
                            const i = (y * canvas.width + x) * 4;
                            coloredMaskData.data[i] = r;
                            coloredMaskData.data[i + 1] = g;
                            coloredMaskData.data[i + 2] = b;
                            coloredMaskData.data[i + 3] = 200; // Semi-transparent
                          }
                        }
                      }
                      
                      // Put the colored mask back on the temporary canvas
                      tempCtx.putImageData(coloredMaskData, 0, 0);
                      
                      // Draw the colored mask on the main canvas
                      ctx.save();
                      ctx.globalCompositeOperation = 'source-over';
                      ctx.drawImage(tempCanvas, 0, 0);
                      ctx.restore();
                      
                      console.log(`Mask for object ${object.id} drawn successfully`);
                      resolve();
                    };
                    
                    maskImg.onerror = async (e) => {
                      console.error('Failed to load mask image:', fullMaskUrl, e);
                      
                      // Try an alternative URL format as fallback
                      const altMaskUrl = `${config.apiUrl}/uploads/${videoId}/tracks/${object.id}_${currentFrameIndex.toString().padStart(5, '0')}.png`;
                      console.log('Trying alternative mask URL:', altMaskUrl);
                      
                      try {
                        // Check if alt mask exists
                        const altCheckResponse = await fetch(altMaskUrl, { method: 'HEAD' });
                        console.log(`Alternative mask URL ${altMaskUrl} status: ${altCheckResponse.status}`);
                        
                        if (altCheckResponse.status !== 200) {
                          // Try one more format
                          const alt2MaskUrl = `${config.apiUrl}/uploads/${videoId}/masks/${object.id}_${currentFrameIndex.toString().padStart(5, '0')}.png`;
                          console.log('Trying second alternative mask URL:', alt2MaskUrl);
                          
                          const alt2CheckResponse = await fetch(alt2MaskUrl, { method: 'HEAD' });
                          console.log(`Second alternative mask URL ${alt2MaskUrl} status: ${alt2CheckResponse.status}`);
                          
                          if (alt2CheckResponse.status !== 200) {
                            throw new Error('All mask URLs failed');
                          } else {
                            loadAltMask(alt2MaskUrl);
                          }
                        } else {
                          loadAltMask(altMaskUrl);
                        }
                      } catch (error) {
                        console.error('All mask URLs failed:', error);
                        
                        if (DEBUG_MASKS) {
                          // Draw a debug rectangle as a placeholder for the missing mask
                          ctx.save();
                          const objectColor = objectColors[index % objectColors.length];
                          ctx.fillStyle = objectColor + '80'; // 50% opacity
                          ctx.fillRect(50, 50 + index * 100, 200, 100);
                          ctx.fillStyle = 'white';
                          ctx.fillText(`Missing mask for ${object.id}`, 60, 100 + index * 100);
                          ctx.restore();
                        }
                        
                        resolve();
                      }
                      
                      // Function to load and draw the alternative mask
                      function loadAltMask(url) {
                        const altMaskImg = new Image();
                        altMaskImg.crossOrigin = "anonymous";
                        
                        altMaskImg.onload = () => {
                          console.log('Alternative mask loaded successfully:', url);
                          // Create a temporary canvas for the mask
                          const tempCanvas = document.createElement('canvas');
                          tempCanvas.width = canvas.width;
                          tempCanvas.height = canvas.height;
                          const tempCtx = tempCanvas.getContext('2d');
                          
                          // Draw the mask on the temporary canvas
                          tempCtx.drawImage(altMaskImg, 0, 0, canvas.width, canvas.height);
                          
                          // Debug: draw the raw mask directly on canvas for visibility
                          if (DEBUG_MASKS) {
                            ctx.save();
                            ctx.globalAlpha = 0.8;
                            ctx.drawImage(altMaskImg, 0, 0, canvas.width, canvas.height);
                            ctx.restore();
                          }
                          
                          // Get the mask data
                          const maskData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
                          
                          // Create a colored version of the mask
                          const coloredMaskData = new ImageData(canvas.width, canvas.height);
                          const objectColor = objectColors[index % objectColors.length];
                          
                          // Parse the hex color to RGB
                          const r = parseInt(objectColor.slice(1, 3), 16);
                          const g = parseInt(objectColor.slice(3, 5), 16);
                          const b = parseInt(objectColor.slice(5, 7), 16);
                          
                          // Apply the color to the mask with the global opacity setting
                          const opacity = DEBUG_MASKS ? 0.9 : (globalParams.maskOpacity / 100);
                          
                          for (let i = 0; i < maskData.data.length; i += 4) {
                            if (maskData.data[i] > 0 || maskData.data[i+1] > 0 || 
                                maskData.data[i+2] > 0 || maskData.data[i+3] > 0) {
                              coloredMaskData.data[i] = r;     // R
                              coloredMaskData.data[i + 1] = g; // G
                              coloredMaskData.data[i + 2] = b; // B
                              coloredMaskData.data[i + 3] = Math.round(255 * opacity); // Alpha with debug opacity
                            }
                          }
                          
                          // Put the colored mask back on the temporary canvas
                          tempCtx.putImageData(coloredMaskData, 0, 0);
                          
                          // Draw the colored mask on the main canvas
                          ctx.save();
                          ctx.globalCompositeOperation = 'source-over';
                          ctx.drawImage(tempCanvas, 0, 0);
                          ctx.restore();
                          
                          resolve();
                        };
                        
                        altMaskImg.onerror = () => {
                          console.error('Failed to load alternative mask image:', url);
                          resolve();
                        };
                        
                        altMaskImg.src = url;
                      }
                    };
                    
                    maskImg.src = fullMaskUrl;
                  } catch (error) {
                    console.error("Error processing object mask:", error);
                    resolve();
                  }
                });
              });
              
              // Wait for all masks to be processed before handling effects
              Promise.all(maskPromises).then(() => {
                console.log("All masks processed");
                // Process effects if needed
              });
            } else {
              console.log('No objects to draw or objects are empty');
            }
          };
          
          img.onerror = (e) => {
            console.error('Failed to load frame image:', fullUrl, e);
            setError('Failed to load frame');
          };
        }
      } catch (error) {
        console.error("Error in renderFrame:", error);
      }
    };

    renderFrame();
  }, [frames, currentFrameIndex, objects, globalParams, objectColors, videoId, config.apiUrl]);

  // Load objects from segmentation mode
  useEffect(() => {
    const fetchObjects = async () => {
      try {
        // Fetch the tracked objects from the backend
        console.log('Fetching objects for FX mode...');
        
        // First, get frames to ensure video info is loaded
        const response = await axios.get(`${config.apiUrl}/video-frames/${videoId}?start=0&count=1`);
        
        // Get all masks from the uploads directory for this video
        console.log(`Fetching masks from ${config.apiUrl}/list-masks/${videoId}`);
        const masksResponse = await axios.get(`${config.apiUrl}/list-masks/${videoId}`);
        console.log('Masks response:', masksResponse.data);
        
        if (masksResponse.data && masksResponse.data.objects && masksResponse.data.objects.length > 0) {
          console.log(`Found ${masksResponse.data.objects.length} objects with masks`);
          
          // Log the first object's masks to debug
          const firstObj = masksResponse.data.objects[0];
          console.log(`First object ${firstObj.id} has ${Object.keys(firstObj.masks).length} masks`);
          console.log('Sample mask URL:', firstObj.masks[Object.keys(firstObj.masks)[0]]);
          
          setObjects(masksResponse.data.objects.map(obj => ({
            id: obj.id,
            name: obj.name || `Object ${obj.id.substring(4, 10)}`,
            visible: true,
            masks: obj.masks,
            effects: {}
          })));
          
          setSelectedObjectIndex(0);
        } else {
          // If no objects found, check if we have any masks in the backend that we can detect
          console.log('No objects returned from API, checking for masks manually...');
          
          // As a fallback, attempt to detect masks from the logs
          const objectsWithMasks = [];
          const regexPattern = /obj_[a-zA-Z0-9_]+/g;
          
          // Check for object IDs in the console logs (not ideal but can work as a fallback)
          let objectIds = new Set();
          let consoleOutput = '';
          try {
            const consoleResponse = await axios.get(`${config.apiUrl}/console-log`);
            consoleOutput = consoleResponse.data.log;
            const matches = consoleOutput.match(regexPattern);
            if (matches) {
              matches.forEach(match => objectIds.add(match));
            }
          } catch (err) {
            console.error('Error fetching console logs:', err);
          }
          
          // If we found object IDs, create objects with masks
          if (objectIds.size > 0) {
            let index = 0;
            for (const objId of objectIds) {
              const masks = {};
              // Assume masks exist for frames 0-30 (for simplicity)
              for (let i = 0; i < 30; i++) {
                const paddedIndex = i.toString().padStart(5, '0');
                masks[i] = `${config.apiUrl}/uploads/${videoId}/masks/${objId}_${paddedIndex}.png`;
                console.log(`Created mask URL for ${objId}, frame ${i}: ${masks[i]}`);
              }
              
              objectsWithMasks.push({
                id: objId,
                name: `Object ${index + 1}`,
                visible: true,
                masks: masks,
                effects: {}
              });
              index++;
            }
            
            if (objectsWithMasks.length > 0) {
              console.log(`Created ${objectsWithMasks.length} objects with masks from logs`);
              setObjects(objectsWithMasks);
              setSelectedObjectIndex(0);
            } else {
              setError('No objects found. Please go back to Segmentation Mode and create objects first.');
            }
          } else {
            setError('No objects found. Please go back to Segmentation Mode and create objects first.');
          }
        }
      } catch (error) {
        console.error('Error fetching objects:', error);
        setError('Failed to load objects. Please go back to Segmentation Mode and create objects first.');
      }
    };

    fetchObjects();
  }, [videoId]);

  const handleObjectSelect = (index) => {
    setSelectedObjectIndex(index);
  };

  const handleToggleVisibility = (index) => {
    const updatedObjects = [...objects];
    updatedObjects[index].visible = !updatedObjects[index].visible;
    setObjects(updatedObjects);
  };

  const handleGlobalParamChange = (param, value) => {
    setGlobalParams({
      ...globalParams,
      [param]: value
    });
  };

  const handleEffectParamChange = (param, value) => {
    setEffectParams({
      ...effectParams,
      [param]: value
    });
  };

  const handleSelectEffect = (effectId) => {
    setSelectedEffect(effectId);
    
    // Set default parameters for the selected effect
    const effect = EFFECT_TYPES.find(e => e.id === effectId);
    if (effect) {
      const defaultParams = {};
      Object.keys(effect.params).forEach(paramName => {
        defaultParams[paramName] = effect.params[paramName].default;
      });
      setEffectParams(defaultParams);
    } else {
      setEffectParams({});
    }
  };

  const handleApplyEffect = async () => {
    if (selectedObjectIndex === null) {
      setError('Please select an object first');
      return;
    }
    
    if (!selectedEffect) {
      setError('Please select an effect first');
      return;
    }
    
    setEffectsInProgress(true);
    setError('');
    
    try {
      const response = await axios.post(`${config.apiUrl}/apply-effect`, {
        video_id: videoId,
        object_id: objects[selectedObjectIndex].id,
        effect_type: selectedEffect,
        effect_params: {
          ...effectParams,
          globalParams: globalParams
        }
      });
      
      // Update the object with the new effect
      const updatedObjects = [...objects];
      if (!updatedObjects[selectedObjectIndex].effects) {
        updatedObjects[selectedObjectIndex].effects = {};
      }
      
      // Store effect frames
      const effectFrames = {};
      response.data.frames.forEach(frame => {
        effectFrames[frame.frame_index] = frame.effect_url;
      });
      
      updatedObjects[selectedObjectIndex].effects[selectedEffect] = effectFrames;
      setObjects(updatedObjects);
    } catch (error) {
      console.error('Error applying effect:', error);
      setError('Failed to apply effect. Please try again.');
    } finally {
      setEffectsInProgress(false);
    }
  };

  const handleSavePreset = () => {
    if (!presetName.trim()) {
      setError('Please enter a preset name');
      return;
    }
    
    const newPreset = {
      id: `preset_${Date.now()}`,
      name: presetName,
      effect: selectedEffect,
      params: effectParams,
      globalParams: globalParams
    };
    
    setPresets([...presets, newPreset]);
    setPresetName('');
  };

  const handleLoadPreset = (preset) => {
    setSelectedEffect(preset.effect);
    setEffectParams(preset.params);
    setGlobalParams(preset.globalParams);
  };

  const handleExportVideo = async (exportType) => {
    setExportInProgress(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('video_id', videoId);
      
      // Get object IDs and their effect types
      const objectIds = [];
      const effectTypes = [];
      
      objects.forEach(obj => {
        if (obj.effects && Object.keys(obj.effects).length > 0) {
          objectIds.push(obj.id);
          // Use the first effect for each object (in a real app, you might want to handle multiple effects)
          effectTypes.push(Object.keys(obj.effects)[0]);
        }
      });
      
      formData.append('object_ids', objectIds.join(','));
      formData.append('effect_types', effectTypes.join(','));
      formData.append('export_type', exportType);
      
      const response = await axios.post(`${config.apiUrl}/export-video`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Open the exported video in a new tab
      window.open(response.data.video_url, '_blank');
    } catch (error) {
      console.error('Error exporting video:', error);
      setError('Failed to export video. Please try again.');
    } finally {
      setExportInProgress(false);
    }
  };

  const handleSwitchToSegmentationMode = () => {
    navigate(`/segmentation/${videoId}`);
  };

  if (!videoInfo) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" gutterBottom>
        FX Mode
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={8}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 2, 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center',
              backgroundColor: 'background.paper',
              borderRadius: 2,
              overflow: 'hidden'
            }}
          >
            <Box 
              sx={{ 
                position: 'relative', 
                width: '100%', 
                height: 'auto'
              }}
            >
              <canvas 
                ref={canvasRef} 
                style={{ 
                  width: '100%', 
                  height: 'auto',
                  backgroundColor: '#000'
                }}
              />
              
              {effectsInProgress && (
                <Box 
                  sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    right: 0, 
                    bottom: 0, 
                    display: 'flex', 
                    justifyContent: 'center', 
                    alignItems: 'center',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)'
                  }}
                >
                  <CircularProgress />
                </Box>
              )}
            </Box>
            
            <Box sx={{ width: '100%', mt: 2 }}>
              <Slider
                value={currentFrameIndex}
                min={0}
                max={frames.length - 1}
                onChange={(_, value) => setCurrentFrameIndex(value)}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `Frame ${value}`}
              />
              
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                <IconButton onClick={() => setIsPlaying(!isPlaying)}>
                  {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                </IconButton>
                <Typography variant="body2" sx={{ alignSelf: 'center', ml: 1 }}>
                  Frame {currentFrameIndex + 1} / {frames.length}
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 2, 
              backgroundColor: 'background.paper',
              borderRadius: 2,
              height: '100%',
              overflow: 'auto'
            }}
          >
            <Typography variant="h6" gutterBottom>
              FX Controls
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Objects
              </Typography>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                {objects.map((object, index) => (
                  <Button
                    key={object.id}
                    variant={selectedObjectIndex === index ? "contained" : "outlined"}
                    color="primary"
                    size="small"
                    onClick={() => handleObjectSelect(index)}
                    startIcon={object.visible ? <VisibilityIcon /> : <VisibilityOffIcon />}
                    sx={{ mb: 1 }}
                  >
                    {object.name}
                  </Button>
                ))}
              </Box>
              
              {selectedObjectIndex !== null && (
                <Box sx={{ mb: 2 }}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={objects[selectedObjectIndex].visible}
                        onChange={() => handleToggleVisibility(selectedObjectIndex)}
                      />
                    }
                    label="Show Object"
                  />
                </Box>
              )}
            </Box>
            
            <Divider sx={{ mb: 2 }} />
            
            <Typography variant="subtitle1" gutterBottom>
              Global Parameters
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                Mask Opacity: {globalParams.maskOpacity}%
              </Typography>
              <Slider
                value={globalParams.maskOpacity}
                min={0}
                max={100}
                step={1}
                onChange={(_, value) => handleGlobalParamChange('maskOpacity', value)}
              />
              
              <Typography variant="body2" gutterBottom>
                Feather Radius: {globalParams.featherRadius}px
              </Typography>
              <Slider
                value={globalParams.featherRadius}
                min={0}
                max={50}
                step={1}
                onChange={(_, value) => handleGlobalParamChange('featherRadius', value)}
              />
              
              <Typography variant="body2" gutterBottom>
                Feather Expand: {globalParams.featherExpand}px
              </Typography>
              <Slider
                value={globalParams.featherExpand}
                min={-20}
                max={20}
                step={1}
                onChange={(_, value) => handleGlobalParamChange('featherExpand', value)}
              />
              
              <Typography variant="body2" gutterBottom>
                FX Opacity: {globalParams.fxOpacity}%
              </Typography>
              <Slider
                value={globalParams.fxOpacity}
                min={0}
                max={100}
                step={1}
                onChange={(_, value) => handleGlobalParamChange('fxOpacity', value)}
              />
              
              <FormControlLabel
                control={
                  <Checkbox
                    checked={globalParams.invertFX}
                    onChange={(e) => handleGlobalParamChange('invertFX', e.target.checked)}
                  />
                }
                label="Invert FX"
              />
            </Box>
            
            <Divider sx={{ mb: 2 }} />
            
            <Typography variant="subtitle1" gutterBottom>
              FX Stack
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Select Effect</InputLabel>
                <Select
                  value={selectedEffect}
                  onChange={(e) => handleSelectEffect(e.target.value)}
                  label="Select Effect"
                >
                  {EFFECT_TYPES.map((effect) => (
                    <MenuItem key={effect.id} value={effect.id}>
                      {effect.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              {selectedEffect && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Effect Parameters
                  </Typography>
                  
                  {selectedEffect === 'blur' && (
                    <Box>
                      <Typography variant="body2" gutterBottom>
                        Blur Amount: {effectParams.amount || 15}
                      </Typography>
                      <Slider
                        value={effectParams.amount || 15}
                        min={1}
                        max={50}
                        step={1}
                        onChange={(_, value) => handleEffectParamChange('amount', value)}
                      />
                    </Box>
                  )}
                  
                  {selectedEffect === 'chroma' && (
                    <Box>
                      <Typography variant="body2" gutterBottom>
                        Shift Amount: {effectParams.amount || 5}px
                      </Typography>
                      <Slider
                        value={effectParams.amount || 5}
                        min={1}
                        max={20}
                        step={1}
                        onChange={(_, value) => handleEffectParamChange('amount', value)}
                      />
                    </Box>
                  )}
                  
                  {selectedEffect === 'glow' && (
                    <Box>
                      <Typography variant="body2" gutterBottom>
                        Glow Amount: {effectParams.amount || 10}
                      </Typography>
                      <Slider
                        value={effectParams.amount || 10}
                        min={1}
                        max={30}
                        step={1}
                        onChange={(_, value) => handleEffectParamChange('amount', value)}
                      />
                    </Box>
                  )}
                  
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={handleApplyEffect}
                    disabled={!selectedEffect || selectedObjectIndex === null || effectsInProgress}
                    fullWidth
                    sx={{ mt: 1 }}
                  >
                    {effectsInProgress ? <CircularProgress size={24} /> : 'Apply Effect'}
                  </Button>
                </Box>
              )}
            </Box>
            
            <Divider sx={{ mb: 2 }} />
            
            <Typography variant="subtitle1" gutterBottom>
              Presets
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <TextField
                label="Preset Name"
                value={presetName}
                onChange={(e) => setPresetName(e.target.value)}
                fullWidth
                size="small"
                sx={{ mb: 1 }}
              />
              
              <Button
                variant="outlined"
                startIcon={<SaveIcon />}
                onClick={handleSavePreset}
                disabled={!selectedEffect || !presetName.trim()}
                fullWidth
                sx={{ mb: 1 }}
              >
                Save Preset
              </Button>
              
              {presets.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="body2" gutterBottom>
                    Saved Presets
                  </Typography>
                  
                  {presets.map((preset) => (
                    <Button
                      key={preset.id}
                      variant="outlined"
                      size="small"
                      startIcon={<FolderOpenIcon />}
                      onClick={() => handleLoadPreset(preset)}
                      fullWidth
                      sx={{ mb: 0.5 }}
                    >
                      {preset.name}
                    </Button>
                  ))}
                </Box>
              )}
            </Box>
            
            <Divider sx={{ mb: 2 }} />
            
            <Typography variant="subtitle1" gutterBottom>
              Export
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<MovieIcon />}
                onClick={() => handleExportVideo('mask')}
                disabled={exportInProgress}
                fullWidth
                sx={{ mb: 1 }}
              >
                {exportInProgress ? <CircularProgress size={24} /> : 'Export Mask Video'}
              </Button>
              
              <Button
                variant="contained"
                color="secondary"
                startIcon={<VideocamIcon />}
                onClick={() => handleExportVideo('fx')}
                disabled={exportInProgress}
                fullWidth
              >
                {exportInProgress ? <CircularProgress size={24} /> : 'Export FX Video'}
              </Button>
            </Box>
            
            <Divider sx={{ mb: 2 }} />
            
            <Button
              variant="outlined"
              color="primary"
              startIcon={<SegmentIcon />}
              onClick={handleSwitchToSegmentationMode}
              fullWidth
            >
              Seg Mode
            </Button>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FXMode; 