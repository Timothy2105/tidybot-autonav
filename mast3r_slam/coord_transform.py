import numpy as np
from scipy.spatial.transform import Rotation

def get_rotation_from_directions(orientation_def):
    """Builds a 3x3 rotation matrix from a dictionary of axis directions."""
    direction_vectors = {
        'forward': [1, 0, 0], 'backward': [-1, 0, 0],
        'right':   [0, 1, 0], 'left':     [0, -1, 0],
        'up':      [0, 0, 1], 'down':     [0, 0, -1]
    }
    # The columns of the rotation matrix are the new basis vectors.
    return np.column_stack([
        direction_vectors[orientation_def['x']],
        direction_vectors[orientation_def['y']],
        direction_vectors[orientation_def['z']]
    ])

def calculate_world_tilt_from_config(config):
    """
    Calculates the net rotation in the world frame based on the phone's mounting
    orientation and its internal IMU reading.

    Returns:
        np.ndarray: A 3x3 rotation matrix representing the equivalent tilt in the world frame.
    """
    # Define the vector mapping legend.
    direction_vectors = {
        "forward": [1, 0, 0], "backward": [-1, 0, 0],
        "right":   [0, 1, 0], "left":     [0, -1, 0],
        "up":      [0, 0, 1], "down":     [0, 0, -1]
    }

    # --- Step 1: Define the actual (local) and reference (world) orientations ---
    phone_def = config['phone_orientation']
    world_def = config['reference_orientation']

    # Convert orientation definitions into sets of vectors
    local_vectors = np.array([
        direction_vectors[phone_def['x']],
        direction_vectors[phone_def['y']],
        direction_vectors[phone_def['z']]
    ])
    world_vectors = np.array([
        direction_vectors[world_def['x']],
        direction_vectors[world_def['y']],
        direction_vectors[world_def['z']]
    ])

    # --- Step 2: Calculate the "mounting error" rotation (R_correction) ---
    # This finds the rotation to align the phone's local frame with the world frame.
    R_local_to_world, _ = Rotation.align_vectors(world_vectors, local_vectors)

    # --- Step 3: Get the raw IMU rotation measured in the phone's local frame ---
    yaw = config['phone_orientation']['yaw']
    pitch = config['phone_orientation']['pitch']
    roll = config['phone_orientation']['roll']
    
    # This is the phone's tilt, expressed in its own local coordinate system.
    R_measured_local = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True)

    # --- Step 4: Calculate the equivalent tilt in the World Frame ("What happened?") ---
    # Use the change of basis formula: R_world = R_change * R_local * R_change_inverse
    R_equivalent_world_tilt = R_local_to_world * R_measured_local * R_local_to_world.inv()
    
    return R_equivalent_world_tilt.as_matrix()