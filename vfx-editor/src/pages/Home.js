import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  Container, 
  CircularProgress,
  Alert
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import axios from 'axios';

const Home = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      // Check if the file is a video
      if (!selectedFile.type.startsWith('video/')) {
        setError('Please select a video file');
        return;
      }
      setFile(selectedFile);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Add a timestamp to prevent caching issues
      const timestamp = new Date().getTime();
      
      const response = await axios.post(`http://localhost:8000/upload-video?t=${timestamp}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Accept': 'application/json',
        },
        // Setting withCredentials to false can help with CORS issues
        withCredentials: false,
        // Add timeout to prevent hanging if the server doesn't respond
        timeout: 60000, // 60 seconds
      });

      console.log('Upload successful:', response.data);
      
      // Navigate to the segmentation mode with the video ID
      navigate(`/segmentation/${response.data.video_id}`);
    } catch (error) {
      console.error('Error uploading video:', error);
      
      // Provide more specific error messages based on the error type
      if (error.response) {
        // The server responded with an error status code
        console.error('Server error data:', error.response.data);
        setError(`Upload failed: ${error.response.data.detail || error.response.statusText}`);
      } else if (error.request) {
        // The request was made but no response was received
        console.error('No response received:', error.request);
        setError('Failed to connect to the server. Please check if the backend is running.');
      } else {
        // Something else caused the error
        setError(`Upload failed: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ mt: 8, mb: 4, textAlign: 'center' }}>
        <Typography variant="h2" component="h1" gutterBottom>
          VFX Editor
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          Upload a video, segment objects, and apply visual effects
        </Typography>
      </Box>

      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center',
          border: '2px dashed rgba(255, 255, 255, 0.12)',
          backgroundColor: 'background.paper',
          borderRadius: 2
        }}
      >
        <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          Upload a Video
        </Typography>
        
        <input
          accept="video/*"
          style={{ display: 'none' }}
          id="upload-video-button"
          type="file"
          onChange={handleFileChange}
        />
        <label htmlFor="upload-video-button">
          <Button 
            variant="contained" 
            component="span"
            disabled={loading}
          >
            Select Video
          </Button>
        </label>
        
        {file && (
          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Typography variant="body1" gutterBottom>
              Selected: {file.name}
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleUpload}
              disabled={loading}
              sx={{ mt: 1 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Upload & Continue'}
            </Button>
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mt: 2, width: '100%' }}>
            {error}
          </Alert>
        )}
      </Paper>

      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          How It Works
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2, flexWrap: 'wrap' }}>
          <Paper sx={{ p: 2, mb: 2, width: { xs: '100%', sm: '30%' } }}>
            <Typography variant="subtitle1" gutterBottom>
              1. Upload Video
            </Typography>
            <Typography variant="body2">
              Upload your video file to get started
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, mb: 2, width: { xs: '100%', sm: '30%' } }}>
            <Typography variant="subtitle1" gutterBottom>
              2. Segment Objects
            </Typography>
            <Typography variant="body2">
              Click on objects to segment and track them
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, mb: 2, width: { xs: '100%', sm: '30%' } }}>
            <Typography variant="subtitle1" gutterBottom>
              3. Apply Effects
            </Typography>
            <Typography variant="body2">
              Add visual effects to segmented objects
            </Typography>
          </Paper>
        </Box>
      </Box>
    </Container>
  );
};

export default Home; 