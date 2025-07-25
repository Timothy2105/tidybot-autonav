import numpy as np
import cv2
import re
from scipy.spatial.transform import Rotation as R

# create 4x4 homogeneous transformation matrix
def to_4x4_matrix(rotation_matrix, translation_vector):
    matrix = np.identity(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation_vector.flatten()
    return matrix

def parse_calibration_data(calibration_text):
    wheel_poses_raw = re.findall(r"WHEEL ODOMETRY END POSE:\s+\[x=([-\d.]+), y=([-\d.]+), theta=([-\d.]+)°\]", calibration_text)
    cam_positions_raw = re.findall(r"CAMERA ESTIMATED END POSITION \(SLAM\):\s+\[x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\]", calibration_text)
    cam_orientations_raw = re.findall(r"CAMERA ESTIMATED END ORIENTATION \(SLAM\) \(quaternion\):\s+\[w=([-\d.]+), x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\]", calibration_text)

    if not (len(wheel_poses_raw) == len(cam_positions_raw) == len(cam_orientations_raw)):
        raise ValueError("Inconsistent number of data points found in the calibration file.")
    if len(wheel_poses_raw) < 2:
        raise ValueError("Not enough data points to compute relative motion.")

    # process wheel odometry
    t_wheel2origin = []
    R_wheel2origin = []
    for x_str, y_str, theta_str in wheel_poses_raw:
        x, y, theta_deg = float(x_str), float(y_str), float(theta_str)
        t_wheel2origin.append(np.array([x, y, 0.0]))
        
        # Original rotation about z-axis from theta_deg - x is written here because quaternion is [w, z, y, x]
        R1 = R.from_euler('x', theta_deg, degrees=True).as_matrix()
        
        # Additional rotation transformations
        # R2: 90 degrees positive about y-axis
        R2 = R.from_euler('y', 90, degrees=True).as_matrix()
        
        # R3: 90 degrees positive about z-axis - x written here because quaternion is [w, x, y, z]
        R3 = R.from_euler('x', 90, degrees=True).as_matrix()
        
        # Apply transformations: R_final = R3 * R2 * R1
        R_final = R3 @ R2 @ R1
        
        R_wheel2origin.append(R_final)

    # proces slam poses
    t_cam2slam = []
    R_cam2slam = []
    for pos_raw, orient_raw in zip(cam_positions_raw, cam_orientations_raw):
        # position vector
        t_cam2slam.append(np.array([float(p) for p in pos_raw]))
        
        # quat to rot matrix
        quat_wxyz = [float(q) for q in orient_raw]
        
        # [w, x, y, z] -> [x, y, z, w]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        
        R_cam2slam.append(R.from_quat(quat_xyzw).as_matrix())

    return R_wheel2origin, t_wheel2origin, R_cam2slam, t_cam2slam

def calibrate_camera_to_wheel(R_wheel, t_wheel, R_cam, t_cam):
    R_cam2wheel, t_cam2wheel = cv2.calibrateHandEye(
        R_gripper2base=R_wheel,
        t_gripper2base=t_wheel,
        R_target2cam=R_cam,
        t_target2cam=t_cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS  # alt: method 4
    )
    return R_cam2wheel, t_cam2wheel

# uses train-test split
def verify_calibration_accuracy(R_cam2wheel, t_cam2wheel, R_wheel_test, t_wheel_test, R_cam_test, t_cam_test):
    print(f"\nVerifying calibration against {len(R_wheel_test)} unseen test points...")
    
    # conv calibration to 4x4 matrix
    T_cam_to_wheel = to_4x4_matrix(R_cam2wheel, t_cam2wheel)

    linear_errors = []
    rotation_errors_deg = []
    pred_positions = []
    pred_quaternions = []
    gt_positions = []  # Ground truth positions
    gt_quaternions = []  # Ground truth quaternions
    gt_wheel_positions = []  # Ground truth wheel positions
    gt_wheel_quaternions = []  # Ground truth wheel quaternions

    for i in range(len(R_wheel_test)):
        # get ground truth
        T_actual_wheel = to_4x4_matrix(R_wheel_test[i], t_wheel_test[i])
        T_actual_cam = to_4x4_matrix(R_cam_test[i], t_cam_test[i])

        # Pose(Wheel) = Pose(Camera) * Transform(Camera -> Wheel)
        T_predicted_wheel = T_actual_cam @ T_cam_to_wheel

        # calculate error btwn predicted and actual wheel poses
        # linear error
        pos_actual = T_actual_wheel[:3, 3]
        pos_predicted = T_predicted_wheel[:3, 3]
        lin_error = np.linalg.norm(pos_actual - pos_predicted)
        linear_errors.append(lin_error)

        # rotational error
        R_actual = T_actual_wheel[:3, :3]
        R_predicted = T_predicted_wheel[:3, :3]
        
        # Find the rotation that maps the predicted orientation to the actual one
        R_error = R_actual @ R_predicted.T
        
        # get angle of rotation in deg
        rot_error_rad = np.linalg.norm(R.from_matrix(R_error).as_rotvec())
        rotation_errors_deg.append(np.rad2deg(rot_error_rad))

        # print ground truth + preds
        quat_actual = R.from_matrix(R_actual).as_quat()  # xyzw
        quat_actual_wxyz = [quat_actual[3], quat_actual[0], quat_actual[1], quat_actual[2]]
        quat_pred = R.from_matrix(R_predicted).as_quat()  # xyzw
        quat_pred_wxyz = [quat_pred[3], quat_pred[0], quat_pred[1], quat_pred[2]]
        print(f"Test {i+1}:")
        print(f"  Ground Truth Wheel Pose:")
        print(f"    Position:   [{pos_actual[0]:.4f}, {pos_actual[1]:.4f}, {pos_actual[2]:.4f}]")
        print(f"    Quaternion: [w={quat_actual_wxyz[0]:.6f}, x={quat_actual_wxyz[1]:.6f}, y={quat_actual_wxyz[2]:.6f}, z={quat_actual_wxyz[3]:.6f}]")
        print(f"  Predicted Wheel Pose:")
        print(f"    Position:   [{pos_predicted[0]:.4f}, {pos_predicted[1]:.4f}, {pos_predicted[2]:.4f}]")
        print(f"    Quaternion: [w={quat_pred_wxyz[0]:.6f}, x={quat_pred_wxyz[1]:.6f}, y={quat_pred_wxyz[2]:.6f}, z={quat_pred_wxyz[3]:.6f}]")
        print(f"  Linear error: {lin_error:.6f} m, Rotational error: {np.rad2deg(rot_error_rad):.6f} deg\n")

        # Save for output
        pred_positions.append(pos_predicted)
        pred_quaternions.append(quat_pred_wxyz)
        
        # Save camera poses as ground truth (not wheel poses)
        cam_pos_actual = T_actual_cam[:3, 3]
        cam_quat_actual = R.from_matrix(T_actual_cam[:3, :3]).as_quat()  # xyzw
        cam_quat_actual_wxyz = [cam_quat_actual[3], cam_quat_actual[0], cam_quat_actual[1], cam_quat_actual[2]]
        gt_positions.append(cam_pos_actual)  # Save camera ground truth position
        gt_quaternions.append(cam_quat_actual_wxyz)  # Save camera ground truth quaternion
        
        # Also save wheel poses as ground truth for visualization
        wheel_pos_actual = pos_actual
        wheel_quat_actual_wxyz = quat_actual_wxyz
        gt_wheel_positions.append(wheel_pos_actual)
        gt_wheel_quaternions.append(wheel_quat_actual_wxyz)

    # Save predictions and ground truth to files
    import os
    os.makedirs('calib-results/kf-preds', exist_ok=True)
    
    # Save predicted wheel poses
    np.savez('calib-results/kf-preds/predicted_wheel_poses.npz', positions=np.array(pred_positions), quaternions=np.array(pred_quaternions))
    
    # Save ground truth camera poses (from SLAM) - ALL poses
    np.savez('calib-results/kf-preds/ground_truth_camera_poses.npz', positions=np.array(gt_positions), quaternions=np.array(gt_quaternions))
    with open('calib-results/kf-preds/ground_truth_camera_poses.txt', 'w') as f:
        for i, (pos, quat) in enumerate(zip(gt_positions, gt_quaternions)):
            f.write(f"Pose {i+1}:\n")
            f.write(f"  Position:   [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]\n")
            f.write(f"  Quaternion: [w={quat[0]:.6f}, x={quat[1]:.6f}, y={quat[2]:.6f}, z={quat[3]:.6f}]\n\n")

    # Save ground truth wheel poses - ALL poses
    np.savez('calib-results/kf-preds/ground_truth_wheel_poses.npz', positions=np.array(gt_wheel_positions), quaternions=np.array(gt_wheel_quaternions))
    with open('calib-results/kf-preds/ground_truth_wheel_poses.txt', 'w') as f:
        for i, (pos, quat) in enumerate(zip(gt_wheel_positions, gt_wheel_quaternions)):
            f.write(f"Pose {i+1}:\n")
            f.write(f"  Position:   [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]\n")
            f.write(f"  Quaternion: [w={quat[0]:.6f}, x={quat[1]:.6f}, y={quat[2]:.6f}, z={quat[3]:.6f}]\n\n")
    
    # Save all transposed ground truth wheel poses - using proper rotation matrix transformations
    with open('calib-results/kf-preds/ground_truth_transposed_wheel_poses.txt', 'w') as f:
        for i, (pos, quat) in enumerate(zip(gt_wheel_positions, gt_wheel_quaternions)):
            # Convert quaternion back to rotation matrix
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]  # wxyz -> xyzw for scipy
            wheel_rot_matrix = R.from_quat(quat_xyzw).as_matrix()
            
            # Create 4x4 transformation matrix
            wheel_pose = np.eye(4)
            wheel_pose[0:3, 0:3] = wheel_rot_matrix
            wheel_pose[0:3, 3] = pos
            
            # Apply coordinate transformation (similar to vis.py approach)
            # This represents the transformation from wheel frame to transposed frame
            transform_matrix = np.array([
                [1, 0, 0, 0],   # x -> x
                [0, 0, -1, 0],  # y -> -z  
                [0, 1, 0, 0],   # z -> y
                [0, 0, 0, 1]
            ])
            
            # Apply transformation
            transposed_wheel_pose = transform_matrix @ wheel_pose
            
            # Extract position and rotation
            transposed_pos = transposed_wheel_pose[0:3, 3]
            transposed_rot_matrix = transposed_wheel_pose[0:3, 0:3]
            
            # Convert back to quaternion (wxyz format)
            transposed_quat_xyzw = R.from_matrix(transposed_rot_matrix).as_quat()
            transposed_quat_wxyz = [transposed_quat_xyzw[3], transposed_quat_xyzw[0], transposed_quat_xyzw[1], transposed_quat_xyzw[2]]
            
            f.write(f"Pose {i+1}:\n")
            f.write(f"  Position:   [{transposed_pos[0]:.6f}, {transposed_pos[1]:.6f}, {transposed_pos[2]:.6f}]\n")
            f.write(f"  Quaternion: [w={transposed_quat_wxyz[0]:.6f}, x={transposed_quat_wxyz[1]:.6f}, y={transposed_quat_wxyz[2]:.6f}, z={transposed_quat_wxyz[3]:.6f}]\n\n")
    
    # Save comparison file with both predicted wheel poses and ground truth camera poses
    with open('calib-results/kf-preds/pose_comparison.txt', 'w') as f:
        for i, (pred_pos, pred_quat, gt_pos, gt_quat) in enumerate(zip(pred_positions, pred_quaternions, gt_positions, gt_quaternions)):
            lin_err = np.linalg.norm(np.array(gt_pos) - np.array(pred_pos))
            f.write(f"Test {i+1}:\n")
            f.write(f"  Ground Truth (Camera from SLAM):\n")
            f.write(f"    Position:   [{gt_pos[0]:.6f}, {gt_pos[1]:.6f}, {gt_pos[2]:.6f}]\n")
            f.write(f"    Quaternion: [w={gt_quat[0]:.6f}, x={gt_quat[1]:.6f}, y={gt_quat[2]:.6f}, z={gt_quat[3]:.6f}]\n")
            f.write(f"  Predicted (Wheel from Calibration):\n")
            f.write(f"    Position:   [{pred_pos[0]:.6f}, {pred_pos[1]:.6f}, {pred_pos[2]:.6f}]\n")
            f.write(f"    Quaternion: [w={pred_quat[0]:.6f}, x={pred_quat[1]:.6f}, y={pred_quat[2]:.6f}, z={pred_quat[3]:.6f}]\n")
            f.write(f"  Linear Error: {lin_err:.6f} m\n\n")

    # calculate RMSE score
    rmse_linear = np.sqrt(np.mean(np.square(linear_errors)))
    rmse_rotation = np.sqrt(np.mean(np.square(rotation_errors_deg)))
    
    return rmse_linear, rmse_rotation

def save_pose_as_npy(rotation_matrix, translation_vector, filename="calib-results/cam_to_wheel_transform.npy"):
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector.flatten()
    np.save(filename, transform_matrix)
    print(f"\nSaved 4x4 transformation matrix to {filename}")
    print("Matrix content:\n", transform_matrix)

if __name__ == "__main__":
    # load + parse data
    calibration_file = "calib-results/calib.txt" 
    try:
        with open(calibration_file, 'r') as f:
            calibration_text = f.read()
        R_wheel, t_wheel, R_cam, t_cam = parse_calibration_data(calibration_text)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit()

    num_samples = len(R_wheel)
    print(f"Successfully parsed {num_samples} total data points.")

    # Use all data for calibration
    print(f"Using all {num_samples} samples for calibration and evaluation.")

    # Calibrate with all data
    print("\nPerforming calibration on the full dataset...")
    R_final, t_final = calibrate_camera_to_wheel(R_wheel, t_wheel, R_cam, t_cam)

    # Evaluate on all data
    if R_final is not None:
        lin_err, rot_err = verify_calibration_accuracy(R_final, t_final, R_wheel, t_wheel, R_cam, t_cam)

        angles_final = R.from_matrix(R_final).as_euler('xyz', degrees=True)
        pos_final = t_final.flatten()

        print("\n============================================================")
        print("CAMERA-TO-WHEEL CALIBRATION RESULTS")
        print("============================================================")
        print(f"Position (x, y, z):      [{pos_final[0]:.4f}, {pos_final[1]:.4f}, {pos_final[2]:.4f}] meters")
        print(f"Orientation (r, p, y):   [{angles_final[0]:.4f}, {angles_final[1]:.4f}, {angles_final[2]:.4f}] degrees")
        print("------------------------------------------------------------")
        print("VERIFICATION ACCURACY (RMSE on All Data)")
        print(f"Linear Error:            {lin_err:.4f} meters ({lin_err*1000:.2f} mm)")
        print(f"Rotational Error:        {rot_err:.4f} degrees")
        print("============================================================")

        # save transformation matrix
        save_pose_as_npy(R_final, t_final)