import numpy as np

def process_tle(tle_lines):
    """Converts TLE lines to ML features"""
    line2 = tle_lines[2].split()  # Split the second line
    
    # Extract orbital parameters
    inclination = float(line2[2])
    eccentricity = float(line2[4]) / 1e7  # Scaled
    mean_motion = float(line2[7])
    
    # Add synthetic size proxy (replace with real data if available)
    size_proxy = 0.5 if "DEB" in tle_lines[0] else 0.8  # Debris vs payload
    
    return np.array([inclination, eccentricity, mean_motion, size_proxy]).reshape(1, -1)
