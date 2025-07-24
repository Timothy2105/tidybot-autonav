import numpy as np
import cv2
import re
from scipy.spatial.transform import Rotation as R

def parse_calibration_data(calibration_text):
    wheel_poses_raw = re.findall(r"WHEEL ODOMETRY END POSE:\s+\[x=([-\d.]+), y=([-\d.]+), theta=([-\d.]+)°\]", calibration_text)
    cam_positions_raw = re.findall(r"CAMERA ESTIMATED END POSITION \(SLAM\):\s+\[x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\]", calibration_text)
    cam_orientations_raw = re.findall(r"CAMERA ESTIMATED END ORIENTATION \(SLAM\):\s+\[roll=([-\d.]+)°, pitch=([-\d.]+)°, yaw=([-\d.]+)°\]", calibration_text)

    if not (len(wheel_poses_raw) == len(cam_positions_raw) == len(cam_orientations_raw)):
        raise ValueError("Inconsistent number of data points found in the calibration file.")
    
    if len(wheel_poses_raw) == 0:
        raise ValueError("No valid calibration data points were found in the provided text.")

    # process wheel odometry
    t_wheel2origin = []
    R_wheel2origin = []
    for x_str, y_str, theta_str in wheel_poses_raw:
        x, y, theta_deg = float(x_str), float(y_str), float(theta_str)
        # odometry translation
        t_wheel2origin.append(np.array([x, y, 0.0]))
        # convert theta to 3x3 rotation matrix
        R_wheel2origin.append(R.from_euler('z', theta_deg, degrees=True).as_matrix())

    # process slam poses
    t_cam2slam = []
    R_cam2slam = []
    for pos_raw, orient_raw in zip(cam_positions_raw, cam_orientations_raw):
        # position vector
        t_cam2slam.append(np.array([float(p) for p in pos_raw]))
        # convert rpy to 3x3 rotation matrix
        rpy_deg = [float(o) for o in orient_raw]
        R_cam2slam.append(R.from_euler('xyz', rpy_deg, degrees=True).as_matrix())

    return R_wheel2origin, t_wheel2origin, R_cam2slam, t_cam2slam

def calibrate_camera_to_wheel(calibration_file_path):
    try:
        with open(calibration_file_path, 'r') as f:
            calibration_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{calibration_file_path}' was not found.")
        return None

    # parse data
    try:
        R_wheel2origin, t_wheel2origin, R_cam2slam, t_cam2slam = parse_calibration_data(calibration_text)
    except ValueError as e:
        print(f"Error parsing data: {e}")
        return None

    print(f"Successfully parsed {len(R_wheel2origin)} data points for calibration.")

    # hand-eye calib
    R_cam2wheel, t_cam2wheel = cv2.calibrateHandEye(
        R_gripper2base=R_wheel2origin,
        t_gripper2base=t_wheel2origin,
        R_target2cam=R_cam2slam,
        t_target2cam=t_cam2slam,
        method=4
    )

    pos = t_cam2wheel.flatten()
    # rotation matrix -> euler
    angles = R.from_matrix(R_cam2wheel).as_euler('xyz', degrees=True)

    pose = np.concatenate([pos, angles])

    return pose

if __name__ == "__main__":
    # NOTE: change file to wherever calib results from robot_calibration.py are stored
    calibration_file = "calib-results/calib copy.txt"
    
    final_pose = calibrate_camera_to_wheel(calibration_file)

    if final_pose is not None:
        print("\n============================================================")
        print("CAMERA-TO-WHEEL CALIBRATION POSE")
        print("============================================================")
        print(f"Position (x, y, z):      [{final_pose[0]:.4f}, {final_pose[1]:.4f}, {final_pose[2]:.4f}] meters")
        print(f"Orientation (r, p, y):   [{final_pose[3]:.4f}, {final_pose[4]:.4f}, {final_pose[5]:.4f}] degrees")
        print("============================================================")
        print("\nThis pose represents the transformation FROM the camera's coordinate system TO the wheel's coordinate system.")