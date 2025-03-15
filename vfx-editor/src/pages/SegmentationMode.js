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
  Tooltip
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import AddIcon from '@mui/icons-material/Add';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import DeleteIcon from '@mui/icons-material/Delete';
import ClearIcon from '@mui/icons-material/Clear';
import TrackChangesIcon from '@mui/icons-material/TrackChanges';
import MovieFilterIcon from '@mui/icons-material/MovieFilter';
import axios from 'axios';
import config from '../config';

const SegmentationMode = () => {
  const { videoId } = useParams();
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const [videoInfo, setVideoInfo] = useState(null);
  const [frames, setFrames] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [objects, setObjects] = useState([]);
  const [selectedObjectIndex, setSelectedObjectIndex] = useState(null);
  const [points, setPoints] = useState([]);
  const [labels, setLabels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [trackingInProgress, setTrackingInProgress] = useState(false);
  const [segmentationInProgress, setSegmentationInProgress] = useState(false);

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
      console.log('Loading frame', currentFrameIndex, 'from', frames[currentFrameIndex].url);
      const img = new Image();
      img.src = `${config.apiUrl}${frames[currentFrameIndex].url}`;
      img.onload = () => {
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          // Set canvas dimensions to match the image
          canvas.width = img.width;
          canvas.height = img.height;
          // Draw the image
          ctx.drawImage(img, 0, 0);
          
          // Draw points
          drawPoints(ctx);
          
          // Draw masks for the current frame
          drawMasks(ctx);
        }
      };
      img.onerror = (err) => {
        console.error('Error loading frame:', err);
        setError('Failed to load frame');
      };
    }
  }, [frames, currentFrameIndex, points, objects, selectedObjectIndex]);

  // Function to draw the points on the canvas
  const drawPoints = (ctx) => {
    points.forEach((point, index) => {
      const label = labels[index];
      ctx.beginPath();
      ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
      ctx.fillStyle = label === 1 ? 'green' : 'red';
      ctx.fill();
    });
  };

  // Function to draw the masks for the current frame
  const drawMasks = (ctx) => {
    if (!objects || objects.length === 0) return;
    
    objects.forEach((object, index) => {
      if (object.visible && object.masks && object.masks[currentFrameIndex]) {
        const maskImg = new Image();
        maskImg.src = `${config.apiUrl}${object.masks[currentFrameIndex]}`;
        maskImg.onload = () => {
          // Apply a semi-transparent colored overlay for the mask
          ctx.save();
          ctx.globalAlpha = 0.5;
          ctx.drawImage(maskImg, 0, 0);
          ctx.restore();
          
          // Highlight the selected object
          if (selectedObjectIndex === index) {
            ctx.save();
            ctx.strokeStyle = objectColors[index % objectColors.length];
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            // Draw outline around the mask (simplified)
            // In a real application, you would trace the actual contour of the mask
            ctx.rect(0, 0, maskImg.width, maskImg.height);
            ctx.stroke();
            ctx.restore();
          }
        };
      }
    });
  };

  const handleCanvasClick = (e) => {
    if (selectedObjectIndex === null) {
      setError('Please select or create an object first');
      return;
    }
    
    if (segmentationInProgress) {
      return;
    }

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Calculate click position relative to the canvas
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;
    
    // Account for scaling - convert from display coordinates to actual image coordinates
    const scaleX = canvas.width / canvas.clientWidth;
    const scaleY = canvas.height / canvas.clientHeight;
    
    const x = clientX * scaleX;
    const y = clientY * scaleY;
    
    console.log('Click coordinates:', { clientX, clientY, scaledX: x, scaledY: y });
    
    // Add point with positive label (foreground)
    setPoints([...points, [x, y]]);
    setLabels([...labels, 1]);
  };

  const handleAddObject = () => {
    const newObject = {
      id: `obj_${Date.now()}`,
      name: `Object ${objects.length + 1}`,
      visible: true,
      masks: {}
    };
    
    setObjects([...objects, newObject]);
    setSelectedObjectIndex(objects.length);
    setPoints([]);
    setLabels([]);
  };

  const handleObjectSelect = (index) => {
    setSelectedObjectIndex(index);
    setPoints([]);
    setLabels([]);
  };

  const handleToggleVisibility = (index) => {
    const updatedObjects = [...objects];
    updatedObjects[index].visible = !updatedObjects[index].visible;
    setObjects(updatedObjects);
  };

  const handleDeleteObject = (index) => {
    const updatedObjects = [...objects];
    updatedObjects.splice(index, 1);
    setObjects(updatedObjects);
    
    if (selectedObjectIndex === index) {
      setSelectedObjectIndex(null);
      setPoints([]);
      setLabels([]);
    } else if (selectedObjectIndex > index) {
      setSelectedObjectIndex(selectedObjectIndex - 1);
    }
  };

  const handleClearObject = (index) => {
    const updatedObjects = [...objects];
    updatedObjects[index].masks = {};
    setObjects(updatedObjects);
    
    if (selectedObjectIndex === index) {
      setPoints([]);
      setLabels([]);
    }
  };

  const handleSegment = async () => {
    if (selectedObjectIndex === null) {
      setError('Please select an object first');
      return;
    }
    
    if (points.length === 0) {
      setError('Please add at least one point');
      return;
    }
    
    setSegmentationInProgress(true);
    setError('');
    
    try {
      console.log('Sending segmentation request with points:', points);
      const response = await axios.post(`${config.apiUrl}/segment`, {
        video_id: videoId,
        frame_index: currentFrameIndex,
        points: points,
        labels: labels
      });
      
      console.log('Segmentation response:', response.data);
      
      // Update the object with the new mask
      const updatedObjects = [...objects];
      if (!updatedObjects[selectedObjectIndex].masks) {
        updatedObjects[selectedObjectIndex].masks = {};
      }
      updatedObjects[selectedObjectIndex].masks[currentFrameIndex] = response.data.mask_url;
      updatedObjects[selectedObjectIndex].id = response.data.object_id;
      setObjects(updatedObjects);
      
      // Clear points after segmentation
      setPoints([]);
      setLabels([]);
    } catch (error) {
      console.error('Error segmenting object:', error);
      setError('Failed to segment object. Please try again.');
    } finally {
      setSegmentationInProgress(false);
    }
  };

  const handleTrackObjects = async () => {
    if (objects.length === 0) {
      setError('Please create at least one object first');
      return;
    }
    
    const objectsToTrack = objects.filter(obj => 
      obj.masks && Object.keys(obj.masks).length > 0
    );
    
    if (objectsToTrack.length === 0) {
      setError('Please segment at least one object first');
      return;
    }
    
    setTrackingInProgress(true);
    setError('');
    
    try {
      const response = await axios.post(`${config.apiUrl}/track`, {
        video_id: videoId,
        object_ids: objectsToTrack.map(obj => obj.id)
      });
      
      // Update objects with tracking results
      const updatedObjects = [...objects];
      response.data.objects.forEach(trackedObject => {
        const objectIndex = updatedObjects.findIndex(obj => obj.id === trackedObject.object_id);
        if (objectIndex !== -1) {
          const masks = {};
          trackedObject.tracks.forEach(track => {
            masks[track.frame_index] = track.mask_url;
          });
          updatedObjects[objectIndex].masks = masks;
        }
      });
      
      setObjects(updatedObjects);
    } catch (error) {
      console.error('Error tracking objects:', error);
      setError('Failed to track objects. Please try again.');
    } finally {
      setTrackingInProgress(false);
    }
  };

  const handleSwitchToFXMode = () => {
    navigate(`/fx/${videoId}`);
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
        Segmentation Mode
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
                height: 'auto',
                cursor: selectedObjectIndex !== null ? 'crosshair' : 'default'
              }}
            >
              <canvas 
                ref={canvasRef} 
                onClick={handleCanvasClick}
                style={{ 
                  width: '100%', 
                  height: 'auto',
                  backgroundColor: '#000'
                }}
              />
              
              {segmentationInProgress && (
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
            </Box>
            
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
              <IconButton onClick={() => setIsPlaying(!isPlaying)}>
                {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
              </IconButton>
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Objects
            </Typography>
            
            <Button 
              variant="contained" 
              startIcon={<AddIcon />} 
              onClick={handleAddObject}
              sx={{ mb: 2 }}
            >
              Add Object
            </Button>
            
            {objects.map((object, index) => (
              <Paper 
                key={object.id}
                elevation={1} 
                sx={{ 
                  p: 1, 
                  mb: 1, 
                  display: 'flex', 
                  alignItems: 'center',
                  backgroundColor: selectedObjectIndex === index ? 'primary.light' : 'background.paper'
                }}
                onClick={() => handleObjectSelect(index)}
              >
                <Typography sx={{ flexGrow: 1 }}>
                  {object.name}
                </Typography>
                <Tooltip title="Toggle Visibility">
                  <IconButton onClick={(e) => { e.stopPropagation(); handleToggleVisibility(index); }}>
                    {object.visible ? <VisibilityIcon /> : <VisibilityOffIcon />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="Clear Object">
                  <IconButton onClick={(e) => { e.stopPropagation(); handleClearObject(index); }}>
                    <ClearIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Delete Object">
                  <IconButton onClick={(e) => { e.stopPropagation(); handleDeleteObject(index); }}>
                    <DeleteIcon />
                  </IconButton>
                </Tooltip>
              </Paper>
            ))}
            
            <Divider sx={{ my: 2 }} />
            
            <Button 
              variant="contained" 
              color="primary"
              disabled={selectedObjectIndex === null || points.length === 0 || segmentationInProgress}
              onClick={handleSegment}
              sx={{ mb: 1, width: '100%' }}
            >
              Segment Object
            </Button>
            
            <Button 
              variant="contained" 
              color="secondary"
              startIcon={<TrackChangesIcon />}
              disabled={objects.length === 0 || trackingInProgress}
              onClick={handleTrackObjects}
              sx={{ mb: 1, width: '100%' }}
            >
              Track All Objects
            </Button>
            
            <Button 
              variant="contained" 
              color="success"
              startIcon={<MovieFilterIcon />}
              disabled={objects.length === 0}
              onClick={handleSwitchToFXMode}
              sx={{ width: '100%' }}
            >
              Switch to FX Mode
            </Button>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SegmentationMode;