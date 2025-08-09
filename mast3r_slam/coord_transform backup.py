import numpy as np
from scipy.spatial.transform import Rotation

def get_rotation_from_directions(orientation_def):
    """Builds a 3x3 rotation matrix from a dictionary of axis directions."""
    direction_vectors = {
        'forward': [1, 0, 0], 'backward': [-1, 0, 0],
        'right':   [0, 1, 0], 'left':     [0, -1, 0],
        'up':      [0, 0, 1], 'down':     [0, 0, -1]
    }
    # The columns of the rotation matrix are the basis vectors of the new frame.
    return np.column_stack([
        direction_vectors[orientation_def['x']],
        direction_vectors[orientation_def['y']],
        direction_vectors[orientation_def['z']]
    ])

def get_camera_correction_from_config(config):
    """
    COMPLETE PIPELINE:
    Computes the final camera correction matrix based on the phone's orientation
    and IMU readings specified in the configuration. This matrix, when applied to a
    camera pose, will align it with the ground/world frame.
    Includes detailed terminal output for debugging.
    """
    print("\n--- Starting Coordinate Transformation Calculation ---")

    # --- Step 1: Define the coordinate systems ---
    phone_actual_def = config['phone_orientation']
    phone_reference_def = config['reference_orientation']
    
    print(f"[Info] Phone's Reference (App Standard) Frame: {phone_reference_def}")
    print(f"[Info] Phone's Actual (During Recording) Frame: {phone_actual_def}")

    # Create the rotation matrices that define these frames relative to a standard world
    R_ref = get_rotation_from_directions(phone_reference_def)
    R_act = get_rotation_from_directions(phone_actual_def)

    # --- Step 2: Find the transformation from the Actual frame to the Reference frame ---
    # This matrix realigns the axes. R_act_to_ref * v_act = v_ref
    R_act_to_ref = R_ref.T @ R_act
    print("\n[Step 1] Rotation from Actual Frame to Reference Frame (R_act_to_ref):")
    print(np.round(R_act_to_ref, 2))

    # --- Step 3: Get the raw IMU rotation measured in the Actual frame ---
    yaw = config['phone_orientation']['yaw']
    pitch = config['phone_orientation']['pitch']
    roll = config['phone_orientation']['roll']
    
    # Convert YPR ('zyx' sequence) to a rotation matrix. This is the phone's tilt.
    R_local_tilt = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()
    print(f"\n[Step 2] Raw IMU angles (Yaw={yaw}, Pitch={pitch}, Roll={roll})")
    print("Resulting Local Tilt Rotation (R_local_tilt):")
    print(np.round(R_local_tilt, 2))

    # --- Step 4: Transform the local tilt into the Reference frame ---
    # This gives us the phone's true orientation in the standard reference system.
    # Formula: R_world = R_change_of_basis * R_local * R_change_of_basis_inverse
    R_tilt_in_ref_frame = R_act_to_ref @ R_local_tilt @ R_act_to_ref.T
    
    print("\n[Step 3] Phone's Tilt represented in the Reference Frame:")
    print(np.round(R_tilt_in_ref_frame, 2))

    # --- Step 5: Calculate the correction needed to undo this tilt ---
    # The correction is simply the inverse (transpose) of the tilt.
    R_leveling_correction_in_ref = R_tilt_in_ref_frame.T
    print("\n[Step 4] Required Leveling Correction in Reference Frame (Inverse of Step 3):")
    print(np.round(R_leveling_correction_in_ref, 2))
    
    # For debugging, let's see what this correction is as Yaw-Pitch-Roll
    ypr_ref = Rotation.from_matrix(R_leveling_correction_in_ref).as_euler('zyx', degrees=True)
    print(f" -> As Yaw-Pitch-Roll: Y={ypr_ref[0]:.2f}, P={ypr_ref[1]:.2f}, R={ypr_ref[2]:.2f}")


    # --- Step 6: Transform the correction into the Camera's coordinate system ---
    # Define the fixed rotation from the Phone Reference frame to the Camera frame.
    # Camera: Z-fwd, X-right, Y-down
    # Phone Reference: X-fwd, Y-right, Z-up
    R_ref_to_cam = np.array([
        [0, 1, 0],   # Camera X (right) = Phone Ref Y (right)
        [0, 0, -1],  # Camera Y (down)  = -Phone Ref Z (up)
        [1, 0, 0],   # Camera Z (fwd)   = Phone Ref X (fwd)
    ])
    print("\n[Step 5] Fixed Rotation from Phone Reference Frame to Camera Frame (R_ref_to_cam):")
    print(np.round(R_ref_to_cam, 2))

    # Apply the final change of basis.
    R_final_correction_in_cam = R_ref_to_cam @ R_leveling_correction_in_ref @ R_ref_to_cam.T
    
    print("\n[Step 6] FINAL RESULT: Leveling Correction in Camera's Local Frame:")
    print(np.round(R_final_correction_in_cam, 2))
    print("--- End of Calculation ---\n")

    return R_final_correction_in_cam