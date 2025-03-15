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
    if (frames.length > 0 && currentFrameIndex < frames.length) {
      const frameUrl = frames[currentFrameIndex].url;
      const fullUrl = frameUrl.startsWith('http') 
        ? frameUrl 
        : `${config.apiUrl}${frameUrl}`;
      
      console.log('Loading frame', currentFrameIndex, 'from', fullUrl);
      
      const img = new Image();
      img.crossOrigin = "anonymous"; // Add cross-origin attribute
      img.src = fullUrl;
      
      img.onload = () => {
        console.log('Frame loaded successfully:', fullUrl);
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);
          
          // Draw masks and effects for the current frame
          objects.forEach((object, index) => {
            if (object.visible && object.masks && object.masks[currentFrameIndex]) {
              const maskUrl = object.masks[currentFrameIndex];
              const fullMaskUrl = maskUrl.startsWith('http') 
                ? maskUrl 
                : `${config.apiUrl}${maskUrl}`;
              
              console.log('Loading mask for object', index, 'from', fullMaskUrl);
              
              const maskImg = new Image();
              maskImg.crossOrigin = "anonymous"; // Add cross-origin attribute
              maskImg.src = fullMaskUrl;
              
              maskImg.onload = () => {
                console.log('Mask loaded successfully:', fullMaskUrl);
                // Apply the effect to the masked area
                ctx.save();
                ctx.globalAlpha = globalParams.maskOpacity / 100;
                ctx.drawImage(maskImg, 0, 0);
                ctx.restore();
                
                // If this object has effects, draw them
                if (object.effects) {
                  Object.entries(object.effects).forEach(([effectType, effectFrames]) => {
                    if (effectFrames[currentFrameIndex]) {
                      const effectUrl = effectFrames[currentFrameIndex];
                      const fullEffectUrl = effectUrl.startsWith('http') 
                        ? effectUrl 
                        : `${config.apiUrl}${effectUrl}`;
                      
                      console.log('Loading effect', effectType, 'from', fullEffectUrl);
                      
                      const effectImg = new Image();
                      effectImg.crossOrigin = "anonymous";
                      effectImg.src = fullEffectUrl;
                      
                      effectImg.onload = () => {
                        console.log('Effect loaded successfully:', fullEffectUrl);
                        // Apply the effect with proper opacity
                        ctx.save();
                        
                        // Apply the effect with the specified opacity
                        ctx.globalAlpha = globalParams.fxOpacity / 100;
                        
                        // If invert is enabled, we need to apply the effect to the non-masked area
                        if (globalParams.invertFX) {
                          // First draw the effect over the entire canvas
                          ctx.drawImage(effectImg, 0, 0);
                          
                          // Then use "destination-out" to remove the effect from the masked area
                          ctx.globalCompositeOperation = 'destination-out';
                          ctx.drawImage(maskImg, 0, 0);
                        } else {
                          // Normal mode: just draw the effect
                          ctx.drawImage(effectImg, 0, 0);
                        }
                        
                        ctx.restore();
                      };
                      
                      effectImg.onerror = () => {
                        console.error('Failed to load effect image:', fullEffectUrl);
                      };
                    }
                  });
                }
                
                // Highlight the selected object
                if (selectedObjectIndex === index) {
                  ctx.save();
                  ctx.strokeStyle = objectColors[index % objectColors.length];
                  ctx.lineWidth = 3;
                  ctx.globalAlpha = 0.8;
                  ctx.strokeRect(20, 20, canvas.width - 40, canvas.height - 40);
                  ctx.restore();
                }
              };
              
              maskImg.onerror = () => {
                console.error('Failed to load mask image:', fullMaskUrl);
              };
            }
          });
        }
      };
      
      img.onerror = () => {
        console.error('Failed to load frame image:', fullUrl);
        setError('Failed to load frame');
      };
    }
  }, [frames, currentFrameIndex, objects, selectedObjectIndex, globalParams]);

  // Load objects from segmentation mode
  useEffect(() => {
    const fetchObjects = async () => {
      try {
        // Fetch the tracked objects from the backend
        console.log('Fetching objects for FX mode...');
        
        // First, get frames to ensure video info is loaded
        const response = await axios.get(`${config.apiUrl}/video-frames/${videoId}?start=0&count=1`);
        
        // Get all masks from the uploads directory for this video
        const masksResponse = await axios.get(`${config.apiUrl}/list-masks/${videoId}`);
        
        if (masksResponse.data && masksResponse.data.objects && masksResponse.data.objects.length > 0) {
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
                masks[i] = `${config.apiUrl}/uploads/${videoId}/masks/${objId}_${i.toString().padStart(5, '0')}.png`;
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