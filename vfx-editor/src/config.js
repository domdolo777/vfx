// Configuration for API endpoints
const config = {
    // Base API URL - will use environment variable if available, otherwise default to localhost
    apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:5000',
    
    // Endpoints
    endpoints: {
        upload: '/upload',
        segment: '/segment',
        track: '/track',
        apply_effect: '/apply_effect',
        export: '/export'
    }
};

export default config; 