import random
import time
import numpy as np
import viser
import math
import cv2
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData
import pathlib


def load_ply_file(ply_path):
    """
    Load a .ply file and return points and colors.
    
    Args:
        ply_path (str): Path to the .ply file
        
    Returns:
        tuple: (points, colors) where points is Nx3 array and colors is Nx3 array
    """
    print(f"Loading PLY file: {ply_path}")
    
    # Load the PLY file
    plydata = PlyData.read(ply_path)
    
    # Extract vertex data
    vertices = plydata['vertex']
    
    # Extract x, y, z coordinates
    points = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
    
    # Extract RGB colors (normalize to [0, 1])
    colors = np.column_stack([vertices['red'], vertices['green'], vertices['blue']]) / 255.0
    
    print(f"Loaded {len(points)} points from PLY file")
    print(f"Point cloud bounds: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
          f"Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
          f"Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    return points, colors


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


def quaternion_wxyz_to_rotation_matrix_scipy(q_wxyz):
    """
    Converts a quaternion in (w, x, y, z) format to a 3x3 rotation matrix using SciPy.
    """
    # SciPy's Rotation class expects quaternions in (x, y, z, w) format.
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    
    # Create a Rotation object from the quaternion
    rotation = R.from_quat(q_xyzw)
    
    # Convert the Rotation object to a rotation matrix
    rotation_matrix = rotation.as_matrix()
    
    return rotation_matrix


def apply_transformation_to_points(points, transformation_matrix):
    """
    Apply a 4x4 transformation matrix to a set of 3D points.
    
    Args:
        points (np.ndarray): Nx3 array of points
        transformation_matrix (np.ndarray): 4x4 transformation matrix
        
    Returns:
        np.ndarray: Nx3 array of transformed points
    """
    # Convert points to homogeneous coordinates (Nx4)
    points_homogeneous = np.column_stack([points, np.ones(len(points))])
    
    # Apply transformation
    transformed_points_homogeneous = (transformation_matrix @ points_homogeneous.T).T
    
    # Convert back to 3D coordinates
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points


if __name__ == "__main__":
    server = viser.ViserServer()
    server.set_up_direction((0.0, -1.0, 0.0)) 

    # Original data from vis.py
    wheel_start_pose_0 = [0.000, 0.000, 0.0]
    wheel_end_pose_0 = [0.295, 0.000, 22.5]
    cam_start_pose_0 = [1.613, -0.025, 0.658, 0.042132, 0.269313, 0.053504, 0.960642]
    cam_end_pose_0 = [1.371, 0.063, 0.204, 0.060310, 0.135636, 0.039643, 0.988127]

    wheel_start_pose_1 = [0.295, -0.000, 22.5]
    wheel_end_pose_1 = [0.567, 0.113, 45.1]
    cam_start_pose_1 = [1.484, 0.106, 0.254, 0.071914, 0.057991, 0.032786, 0.995184]
    cam_end_pose_1 = [1.569, 0.095, 0.337, 0.076506, 0.054139, 0.027955, 0.995206]

    wheel_start_pose_2 = [0.566, 0.112, 45.0]
    wheel_end_pose_2 = [0.775, 0.320, 67.4]
    cam_start_pose_2 = [1.644, 0.137, 0.510, 0.087999, -0.141576, 0.018991, 0.985825]
    cam_end_pose_2 = [1.635, 0.117, 0.686, 0.092474, -0.306174, 0.001773, 0.947472]

    wheel_start_pose_3 = [0.775, 0.320, 67.5]
    wheel_end_pose_3 = [0.887, 0.592, 90.0]
    cam_start_pose_3 = [1.604, 0.114, 0.750, 0.095951, -0.332223, 0.000158, 0.938308]
    cam_end_pose_3 = [1.621, 0.089, 0.864, 0.094145, -0.507060, -0.017871, 0.856567]

    wheel_start_pose_4 = [0.888, 0.592, 90.0]
    wheel_end_pose_4 = [0.887, 0.886, 112.5]
    cam_start_pose_4 = [1.604, 0.094, 0.894, 0.095964, -0.517236, -0.018057, 0.850254]
    cam_end_pose_4 = [1.463, 0.084, 1.003, 0.101290, -0.616556, -0.023326, 0.780419]

    wheel_start_pose_5 = [0.887, 0.887, 112.5]
    wheel_end_pose_5 = [0.778, 1.158, 134.8]
    cam_start_pose_5 = [1.397, 0.053, 1.066, 0.099586, -0.672314, -0.032764, 0.732805]
    cam_end_pose_5 = [1.176, 0.044, 1.084, 0.101732, -0.752173, -0.042902, 0.649650]

    wheel_start_pose_6 = [0.775, 1.158, 134.9]
    wheel_end_pose_6 = [0.568, 1.367, 157.4]
    cam_start_pose_6 = [1.131, 0.033, 1.101, 0.099513, -0.802825, -0.051637, 0.585580]
    cam_end_pose_6 = [1.093, -0.003, 1.142, 0.083582, -0.889283, -0.069003, 0.444328]

    wheel_start_pose_7 = [0.567, 1.367, 157.3]
    wheel_end_pose_7 = [0.293, 1.478, 179.6]
    cam_start_pose_7 = [1.086, -0.089, 1.365, 0.081452, -0.875221, -0.068192, 0.471914]
    cam_end_pose_7 = [0.737, -0.073, 1.167, 0.068101, -0.940973, -0.076547, 0.322602]

    wheel_start_pose_8 = [0.295, 1.480, 179.9]
    wheel_end_pose_8 = [0.001, 1.481, 202.4]
    cam_start_pose_8 = [0.758, -0.085, 1.237, 0.066722, -0.949187, -0.077885, 0.297534]
    cam_end_pose_8 = [0.463, -0.050, 0.869, 0.052936, -0.982472, -0.089128, 0.154930]

    wheel_start_pose_9 = [0.001, 1.481, 202.4]
    wheel_end_pose_9 = [-0.272, 1.369, 224.8]
    cam_start_pose_9 = [0.398, -0.032, 0.841, 0.052105, -0.990033, -0.087928, 0.096889]
    cam_end_pose_9 = [0.320, -0.028, 0.830, 0.042324, -0.994438, -0.092590, -0.026983]

    wheel_start_pose_10 = [-0.271, 1.368, 224.9]
    wheel_end_pose_10 = [-0.480, 1.161, 247.4]
    cam_start_pose_10 = [0.305, 0.002, 0.728, 0.034676, -0.989441, -0.091803, -0.106664]
    cam_end_pose_10 = [0.266, 0.017, 0.528, 0.011012, -0.953261, -0.093633, -0.287062]

    wheel_start_pose_11 = [-0.480, 1.160, 247.5]
    wheel_end_pose_11 = [-0.592, 0.888, 270.0]
    cam_start_pose_11 = [0.249, 0.018, 0.528, 0.008689, -0.946013, -0.092764, -0.310450]
    cam_end_pose_11 = [0.374, 0.073, 0.314, -0.005825, -0.894434, -0.090784, -0.437850]

    wheel_start_pose_12 = [-0.592, 0.888, 270.2]
    wheel_end_pose_12 = [-0.592, 0.593, 292.5]
    cam_start_pose_12 = [0.389, 0.096, 0.234, -0.009723, -0.867485, -0.089460, -0.489257]
    cam_end_pose_12 = [0.616, 0.158, 0.049, -0.026668, -0.784594, -0.085875, -0.613455]

    wheel_start_pose_13 = [-0.592, 0.594, 292.7]
    wheel_end_pose_13 = [-0.480, 0.321, 314.9]
    cam_start_pose_13 = [0.736, 0.193, 0.013, -0.037160, -0.745390, -0.078751, -0.660917]
    cam_end_pose_13 = [0.741, 0.197, -0.064, -0.051600, -0.628347, -0.072967, -0.772783]

    wheel_start_pose_14 = [-0.479, 0.322, 315.2]
    wheel_end_pose_14 = [-0.270, 0.115, 337.7]
    cam_start_pose_14 = [0.762, 0.196, -0.073, -0.054850, -0.596592, -0.072352, -0.797393]
    cam_end_pose_14 = [0.962, 0.206, -0.075, -0.072397, -0.435906, -0.057509, -0.895230]

    wheel_start_pose_15 = [-0.270, 0.115, 337.7]
    wheel_end_pose_15 = [0.003, 0.004, 360.2]
    cam_start_pose_15 = [0.961, 0.212, -0.092, -0.074099, -0.416891, -0.054416, -0.904295]
    cam_end_pose_15 = [1.065, 0.200, -0.031, -0.080559, -0.316492, -0.048290, -0.943934]

    wheel_poses = [
        wheel_start_pose_0, wheel_end_pose_0,
        wheel_start_pose_1, wheel_end_pose_1,
        wheel_start_pose_2, wheel_end_pose_2,
        wheel_start_pose_3, wheel_end_pose_3,
        wheel_start_pose_4, wheel_end_pose_4,
        wheel_start_pose_5, wheel_end_pose_5,
        wheel_start_pose_6, wheel_end_pose_6,
        wheel_start_pose_7, wheel_end_pose_7,
        wheel_start_pose_8, wheel_end_pose_8,
        wheel_start_pose_9, wheel_end_pose_9,
        wheel_start_pose_10, wheel_end_pose_10,
        wheel_start_pose_11, wheel_end_pose_11,
        wheel_start_pose_12, wheel_end_pose_12,
        wheel_start_pose_13, wheel_end_pose_13,
        wheel_start_pose_14, wheel_end_pose_14,
        wheel_start_pose_15, wheel_end_pose_15
    ]

    # Generate synthetic camera data - tilted circle
    def generate_tilted_circle_cameras():
        # Circle parameters
        radius = 0.3
        num_points = len(wheel_poses)  # Match the number of wheel poses (32)
        slant_angle = 45 * np.pi / 180  # 45 degrees in radians

        # Calculate quaternion for 45-degree rotation around X-axis
        # For rotation around X-axis: q = [cos(θ/2), sin(θ/2), 0, 0]
        half_angle = slant_angle / 2
        qw = np.cos(half_angle)
        qx = np.sin(half_angle)
        qy = 0
        qz = 0

        # Generate points
        points = []
        for i in range(num_points):
            # Angle for each point (evenly spaced around circle)
            angle = 2 * np.pi * i / num_points
            
            # Initial circle points in XY plane
            x_initial = radius * np.cos(angle)
            y_initial = radius * np.sin(angle)
            z_initial = 0
            
            # Apply 45-degree rotation around X-axis
            # Rotation matrix for X-axis rotation:
            x = x_initial
            y = y_initial * np.cos(slant_angle) - z_initial * np.sin(slant_angle)
            z = y_initial * np.sin(slant_angle) + z_initial * np.cos(slant_angle)
            
            # Format: [x, y, z, qw, qx, qy, qz]
            point = [x, y, z, qw, qx, qy, qz]
            points.append(point)
            
            # Print each point
            print(f"Camera Point {i}: [{x:.4f}, {y:.4f}, {z:.4f}, {qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f}]")
        
        return points

    # Generate synthetic tilted circle camera poses
    cam_poses = generate_tilted_circle_cameras()

    # Prepare data for hand-eye calibration (same as vis.py)
    wheel_translations = [np.array([pose[0], pose[1], 0.0]) for pose in wheel_poses]
    wheel_rotations = [rotation_matrix_z(pose[2]) for pose in wheel_poses]

    cam_translations = [np.array(pose[0:3]) for pose in cam_poses]
    cam_rotations = [quaternion_wxyz_to_rotation_matrix_scipy(pose[3:]) for pose in cam_poses]

    # Perform hand-eye calibration
    print("Performing hand-eye calibration...")
    T_rot, T_trans = cv2.calibrateHandEye(
        R_gripper2base=wheel_rotations,
        t_gripper2base=wheel_translations,
        R_target2cam=cam_rotations,
        t_target2cam=cam_translations,
        method=4,
    )
    T_cam2base = np.eye(4)
    T_cam2base[0:3, 0:3] = T_rot
    T_cam2base[0:3, 3:4] = T_trans

    print(f"Hand-eye calibration transformation matrix:")
    print(T_cam2base)

    # Load the PLY file
    ply_file_path = "logs/live_session_20250729_154408_replay.ply"
    original_points, original_colors = load_ply_file(ply_file_path)

    # Apply hand-eye calibration transformation to all points
    print("Applying hand-eye calibration transformation to point cloud...")
    transformed_points = apply_transformation_to_points(original_points, np.linalg.inv(T_cam2base))

    # Flip everything on Y-axis (multiply Y coordinates by -1)
    print("Flipping everything on Y-axis...")
    transformed_points[:, 1] = -transformed_points[:, 1]  # Flip Y coordinates

    print(f"Transformed point cloud bounds: X[{transformed_points[:, 0].min():.3f}, {transformed_points[:, 0].max():.3f}], "
          f"Y[{transformed_points[:, 1].min():.3f}, {transformed_points[:, 1].max():.3f}], "
          f"Z[{transformed_points[:, 2].min():.3f}, {transformed_points[:, 2].max():.3f}]")

    # Collect camera positions for point cloud
    cam_positions = []
    cam_colors = []
    for i, cam_pose in enumerate(cam_poses):
        cam_position = np.array(cam_pose[0:3])
        cam_positions.append(cam_position)
        # Use blue color for camera points
        cam_colors.append([0.0, 0.0, 1.0])  # RGB blue

    # Convert to numpy arrays
    cam_positions = np.array(cam_positions)
    cam_colors = np.array(cam_colors)

    # Flip camera and wheel positions on Y-axis to match point cloud
    print("Flipping camera and wheel positions on Y-axis...")
    cam_positions[:, 1] = -cam_positions[:, 1]

    # Add point clouds to the scene
    print("Adding point clouds to visualization...")
    
    # Only show transformed point cloud (post-calibration)
    print(f"Adding transformed point cloud with {len(transformed_points)} points")
    server.scene.add_point_cloud(
        "/transformed_pointcloud",
        points=transformed_points,
        colors=original_colors,  # Keep original colors
        point_size=0.02,  # Smaller for better detail
    )

    # Camera positions (blue)
    print(f"Adding camera positions with {len(cam_positions)} points")
    server.scene.add_point_cloud(
        "/camera_points",
        points=cam_positions,
        colors=cam_colors,
        point_size=0.05,  # Smaller for visibility
    )

    # Add coordinate frames to help with orientation
    print("Adding coordinate frames...")
    server.scene.add_frame("/origin", wxyz=np.array([1, 0, 0, 0]), position=np.array([0, 0, 0]), axes_length=1.0, axes_radius=0.02)
    
    # Add a frame at the center of the transformed point cloud
    transformed_center = np.mean(transformed_points, axis=0)
    print(f"Transformed point cloud center: {transformed_center}")
    server.scene.add_frame("/transformed_center", wxyz=np.array([1, 0, 0, 0]), position=transformed_center, axes_length=0.5, axes_radius=0.01)

    print("Visualization started!")
    print("- Main point cloud: PLY points after hand-eye calibration transformation (point_size=0.02)")
    print("- Blue points: Synthetic tilted circle camera positions (45° around X-axis)")
    print("- Coordinate frames: Origin (large), transformed center (medium)")
    print(f"Point sizes: Main point cloud=0.02, camera points=0.05")
    print(f"Try zooming out or moving the camera to see all points!")

    while True:
        time.sleep(0.5)
