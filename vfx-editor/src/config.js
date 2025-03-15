// Configuration for API endpoints
const config = {
    // Base API URL - will detect environment
    apiUrl: process.env.REACT_APP_API_URL || 
           (window.location.hostname.includes('runpod') 
            ? `https://${window.location.hostname.replace('3000', '8000')}` 
            : 'http://localhost:8000'),
    
    // Endpoints
    endpoints: {
        upload: '/upload',
        segment: '/segment',
        track: '/track',
        apply_effect: '/apply_effect',
        export: '/export'
    }
};

console.log('Using API URL:', config.apiUrl);

export default config; 