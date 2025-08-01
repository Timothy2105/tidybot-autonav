import re
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def rotation_matrix_z(degrees):
    """
    Converts an angle in degrees around the Z-axis to a 3x3 rotation matrix.
    """
    radians = np.deg2rad(degrees)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)

    R_z = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])
    return R_z

# convert quaternion to rotation matrix
def quaternion_wxyz_to_rotation_matrix(q_wxyz):
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    
    rotation = R.from_quat(q_xyzw)
    
    rotation_matrix = rotation.as_matrix()
    
    return rotation_matrix

# get wheel and camera poses from calibration log
def extract_poses_from_text(calibration_text):
    # extract wheel poses (x, y, theta in degrees)
    wheel_poses_raw = re.findall(
        r"WHEEL ODOMETRY END POSE:\s+\[x=([-\d.]+), y=([-\d.]+), theta=([-\d.]+)°\]", 
        calibration_text
    )
    
    # extract camera positions (x, y, z)
    cam_positions_raw = re.findall(
        r"CAMERA ESTIMATED END POSITION \(SLAM\):\s+\[x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\]", 
        calibration_text
    )
    
    # extract camera orientations (quaternion w, x, y, z)
    cam_orientations_raw = re.findall(
        r"CAMERA ESTIMATED END ORIENTATION \(SLAM\) \(quaternion\):\s+\[w=([-\d.]+), x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\]", 
        calibration_text
    )
    
    # convert to float arrays
    wheel_poses = [[float(x), float(y), float(theta)] for x, y, theta in wheel_poses_raw]
    cam_positions = [[float(x), float(y), float(z)] for x, y, z in cam_positions_raw]
    cam_orientations = [[float(w), float(x), float(y), float(z)] for w, x, y, z in cam_orientations_raw]
    
    return wheel_poses, cam_positions, cam_orientations

# hand-eye calibration
def calculate_hand_eye_calibration(wheel_poses, cam_positions, cam_orientations):
    # prepare wheel data (robot base to gripper)
    wheel_translations = [np.array([pose[0], pose[1], 0.0]) for pose in wheel_poses]
    wheel_rotations = [rotation_matrix_z(pose[2]) for pose in wheel_poses]
    
    # prepare camera data (camera to target)
    cam_translations = [np.array(pos) for pos in cam_positions]
    cam_rotations = [quaternion_wxyz_to_rotation_matrix(orient) for orient in cam_orientations]
    
    n_poses = min(len(wheel_translations), len(cam_translations))
    wheel_translations = wheel_translations[:n_poses]
    wheel_rotations = wheel_rotations[:n_poses]
    cam_translations = cam_translations[:n_poses]
    cam_rotations = cam_rotations[:n_poses]
    
    print(f"Using {n_poses} pose pairs for calibration")
    
    try:
        # calibrate hand-eye
        T_rot, T_trans = cv2.calibrateHandEye(
            R_gripper2base=wheel_rotations,
            t_gripper2base=wheel_translations,
            R_target2cam=cam_rotations,
            t_target2cam=cam_translations,
            method=cv2.CALIB_HAND_EYE_TSAI  # method 4
        )
        
        # construct 4x4 transformation matrix
        T_cam2base = np.eye(4)
        T_cam2base[0:3, 0:3] = T_rot
        T_cam2base[0:3, 3] = T_trans.flatten()
        
        print("\nHand-eye calibration result:")
        print("Transformation matrix (camera to base):")
        print(T_cam2base)
        
        rotation_euler = R.from_matrix(T_rot).as_euler('xyz', degrees=True)
        print(f"\nRotation (degrees): [x={rotation_euler[0]:.2f}, y={rotation_euler[1]:.2f}, z={rotation_euler[2]:.2f}]")
        print(f"Translation: [x={T_trans[0,0]:.3f}, y={T_trans[1,0]:.3f}, z={T_trans[2,0]:.3f}]")
        
        return T_cam2base
        
    except Exception as e:
        print(f"Error during calibration: {e}")
        return None

def main():
    # read calibration text from file
    calibration_file = "calib-results/calib.txt" 
    try:
        with open(calibration_file, 'r') as f:
            calibration_text = f.read()
        print(f"Successfully loaded calibration data from {calibration_file}")
    except FileNotFoundError:
        print(f"Error: Calibration file {calibration_file} not found!")
        return
    except Exception as e:
        print(f"Error reading calibration file: {e}")
        return
    
    # extract poses from text
    wheel_poses, cam_positions, cam_orientations = extract_poses_from_text(calibration_text)
    
    if len(wheel_poses) == 0 or len(cam_positions) == 0:
        print("No poses found in calibration text!")
        return
    
    print(f"Found {len(wheel_poses)} wheel poses")
    print(f"Found {len(cam_positions)} camera positions")
    print(f"Found {len(cam_orientations)} camera orientations")
    
    # calculate hand-eye calibration
    T_cam2base = calculate_hand_eye_calibration(wheel_poses, cam_positions, cam_orientations)
    
    if T_cam2base is not None:
        # save the calibration matrix
        np.save('calib-results/hand_eye_calibration_matrix.npy', T_cam2base)
        print("\nCalibration matrix saved to 'calib-results/hand_eye_calibration_matrix.npy'")
        
        # save rotation and translation separately
        T_rot = T_cam2base[0:3, 0:3]
        T_trans = T_cam2base[0:3, 3]
        np.save('calib-results/hand_eye_rotation.npy', T_rot)
        np.save('calib-results/hand_eye_translation.npy', T_trans)
        print("Rotation and translation matrices also saved separately.")

if __name__ == "__main__":
    main()