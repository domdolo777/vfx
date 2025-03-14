import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import VideoSettingsIcon from '@mui/icons-material/VideoSettings';

const Header = () => {
  return (
    <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: '1px solid rgba(255, 255, 255, 0.12)' }}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <VideoSettingsIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" color="inherit" noWrap component={RouterLink} to="/" sx={{ textDecoration: 'none', color: 'inherit' }}>
            VFX Editor
          </Typography>
        </Box>
        <Box sx={{ flexGrow: 1 }} />
        <Button color="inherit" component={RouterLink} to="/">
          Home
        </Button>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 