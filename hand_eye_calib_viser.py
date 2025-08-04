import time
import numpy as np
import viser
import cv2
import argparse
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData
import os


def load_ply_file(ply_path):
    print(f"Loading PLY file: {ply_path}")
    
    plydata = PlyData.read(ply_path)
    
    vertices = plydata['vertex']
    
    # extract x, y, z coordinates
    points = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
    
    # extract RGB colors (normalize to [0, 1])
    colors = np.column_stack([vertices['red'], vertices['green'], vertices['blue']]) / 255.0
    
    print(f"Loaded {len(points)} points from PLY file")
    
    return points, colors


# convert degrees to rotation matrix for z-axis
def rotation_matrix_z(degrees):
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
def quaternion_wxyz_to_rotation_matrix_scipy(q_wxyz):
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    
    # create a Rotation object from quaternion
    rotation = R.from_quat(q_xyzw)
    
    # convert the Rotation object to rotation matrix
    rotation_matrix = rotation.as_matrix()
    
    return rotation_matrix

# apply transformation to points
def apply_transformation_to_points(points, transformation_matrix):
    # convert points to homogeneous coordinates (Nx4)
    points_homogeneous = np.column_stack([points, np.ones(len(points))])
    
    # apply transformation
    transformed_points_homogeneous = (transformation_matrix @ points_homogeneous.T).T
    
    # convert back to 3D coordinates
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points

# some hardcoded eyeballed rotations to align transformed point cloud with original axes
def create_final_transformation_matrix(hand_eye_transform):
    # y-axis rotation
    y_angle_rad = np.radians(-25)
    cos_y = np.cos(y_angle_rad)
    sin_y = np.sin(y_angle_rad)
    
    y_rotation_matrix = np.array([
        [cos_y,  0, sin_y, 0],
        [0,      1, 0,     0],
        [-sin_y, 0, cos_y, 0],
        [0,      0, 0,     1]
    ])
    
    # z-axis rotation
    z_angle_rad = np.radians(-5)
    cos_z = np.cos(z_angle_rad)
    sin_z = np.sin(z_angle_rad)
    
    z_rotation_matrix = np.array([
        [cos_z, -sin_z, 0, 0],
        [sin_z,  cos_z, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ])
    
    # x-axis rotation
    x_angle_rad = np.radians(-12)
    cos_x = np.cos(x_angle_rad)
    sin_x = np.sin(x_angle_rad)
    
    x_rotation_matrix = np.array([
        [1, 0,      0,      0],
        [0, cos_x, -sin_x,  0],
        [0, sin_x,  cos_x,  0],
        [0, 0,      0,      1]
    ])
    
    # up on y-axis
    translation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.4],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # combine transformations: hand_eye_transform @ y_rotation @ z_rotation @ x_rotation @ translation
    final_transform = hand_eye_transform @ y_rotation_matrix @ z_rotation_matrix @ x_rotation_matrix @ translation_matrix
    
    return final_transform


def create_inverse_transformation_matrix(hand_eye_transform):
    # get the forward transformation matrix
    forward_transform = create_final_transformation_matrix(hand_eye_transform)
    
    # return inverse
    return np.linalg.inv(forward_transform)


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Hand-eye calibration visualization tool")
    parser.add_argument("--save", action="store_true", 
                       help="Save transformation matrices to calib-results directory")
    args = parser.parse_args()

    server = viser.ViserServer()
    server.set_up_direction((0.0, 1.0, 0.0)) 

    # original data from vis.py
    # NOTE: currently using synthetic data for hand-eye calibration
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

    # generate synthetic camera data - tilted circle
    def generate_tilted_circle_cameras():
        # circle parameters
        radius = 0.3
        num_points = len(wheel_poses)
        slant_angle = 45 * np.pi / 180

        # calculate quaternion for 45-degree rotation around x-axis
        # for rotation around x-axis: q = [cos(θ/2), sin(θ/2), 0, 0]
        half_angle = slant_angle / 2
        qw = np.cos(half_angle)
        qx = np.sin(half_angle)
        qy = 0
        qz = 0

        # generate points
        points = []
        for i in range(num_points):
            # angle for each point (evenly spaced around circle)
            angle = 2 * np.pi * i / num_points
            
            # initial circle points in xy plane
            x_initial = radius * np.cos(angle)
            y_initial = radius * np.sin(angle)
            z_initial = 0
            
            # apply 45-degree rotation around x-axis
            # rotation matrix for x-axis rotation:
            x = x_initial
            y = y_initial * np.cos(slant_angle) - z_initial * np.sin(slant_angle)
            z = y_initial * np.sin(slant_angle) + z_initial * np.cos(slant_angle)
            
            # format: [x, y, z, qw, qx, qy, qz]
            point = [x, y, z, qw, qx, qy, qz]
            points.append(point)
            
            print(f"Camera Point {i}: [{x:.4f}, {y:.4f}, {z:.4f}, {qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f}]")
        
        return points

    # get synthetic camera poses
    cam_poses = generate_tilted_circle_cameras()

    # prepare data for hand-eye calibration (same as vis.py)
    wheel_translations = [np.array([pose[0], pose[1], 0.0]) for pose in wheel_poses]
    wheel_rotations = [rotation_matrix_z(pose[2]) for pose in wheel_poses]

    cam_translations = [np.array(pose[0:3]) for pose in cam_poses]
    cam_rotations = [quaternion_wxyz_to_rotation_matrix_scipy(pose[3:]) for pose in cam_poses]

    # perform hand-eye calibration
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

    # load the ply file
    ply_file_path = "saved-states/test-recalib/point_cloud.ply"
    original_points, original_colors = load_ply_file(ply_file_path)

    # keep original points (no rotation)
    print("Using original PLY points (no rotation)...")
    original_points_display = original_points

    # calculate transformed coordinate frame for reference
    print("Calculating transformed coordinate frame...")
    
    # apply hand-eye calibration + -30° y-axis rotation to coordinate frame
    hand_eye_transform = np.linalg.inv(T_cam2base)
    transformed_axes = create_final_transformation_matrix(hand_eye_transform)

    # create transformed point cloud by applying transformation to every point
    print("Creating transformed point cloud...")
    transformed_points = apply_transformation_to_points(original_points, transformed_axes)
    
    # save transformation matrices to calib-results directory
    if args.save:
        print("Saving transformation matrices...")
        calib_results_dir = "calib-results"
        os.makedirs(calib_results_dir, exist_ok=True)
        
        # save final transformation matrix
        final_transform_path = os.path.join(calib_results_dir, "final_transformation_matrix.npy")
        np.save(final_transform_path, transformed_axes)
        print(f"Saved final transformation matrix to: {final_transform_path}")
        
        # save inverse transformation matrix
        inverse_transform = create_inverse_transformation_matrix(hand_eye_transform)
        inverse_transform_path = os.path.join(calib_results_dir, "inverse_transformation_matrix.npy")
        np.save(inverse_transform_path, inverse_transform)
        print(f"Saved inverse transformation matrix to: {inverse_transform_path}")
        
        # save as text file
        final_transform_txt_path = os.path.join(calib_results_dir, "final_transformation_matrix.txt")
        np.savetxt(final_transform_txt_path, transformed_axes, fmt='%.6f', 
                   header='Final transformation matrix (4x4)\nFormat: [R R R T]\n        [R R R T]\n        [R R R T]\n        [0 0 0 1]')
        print(f"Saved final transformation matrix (text) to: {final_transform_txt_path}")
        
        inverse_transform_txt_path = os.path.join(calib_results_dir, "inverse_transformation_matrix.txt")
        np.savetxt(inverse_transform_txt_path, inverse_transform, fmt='%.6f',
                   header='Inverse transformation matrix (4x4)\nFormat: [R R R T]\n        [R R R T]\n        [R R R T]\n        [0 0 0 1]')
        print(f"Saved inverse transformation matrix (text) to: {inverse_transform_txt_path}")

    print(f"Transformed point cloud bounds: X[{transformed_points[:, 0].min():.3f}, {transformed_points[:, 0].max():.3f}], "
           f"Y[{transformed_points[:, 1].min():.3f}, {transformed_points[:, 1].max():.3f}], "
           f"Z[{transformed_points[:, 2].min():.3f}, {transformed_points[:, 2].max():.3f}]")

    # collect camera positions for point cloud
    cam_positions = []
    cam_colors = []
    for i, cam_pose in enumerate(cam_poses):
        cam_position = np.array(cam_pose[0:3])
        cam_positions.append(cam_position)
        cam_colors.append([0.0, 0.0, 1.0]) # blue

    # convert to numpy arrays
    cam_positions = np.array(cam_positions)
    cam_colors = np.array(cam_colors)

    # add point clouds to the scene
    print("Adding point clouds to visualization...")
    
    # show original point cloud
    print(f"Adding original point cloud with {len(original_points_display)} points")
    server.scene.add_point_cloud(
        "/original_pointcloud",
        points=original_points_display,
        colors=original_colors, 
        point_size=0.003,
    )
    
    # show transformed point cloud
    print(f"Adding transformed point cloud with {len(transformed_points)} points")
    server.scene.add_point_cloud(
        "/transformed_pointcloud",
        points=transformed_points,
        colors=original_colors,
        point_size=0.003,
    )

    # add coordinate frames to help with orientation
    print("Adding coordinate frames...")
    # origin frame - no rotation
    origin_rotation = np.array([1, 0, 0, 0]) # identity quaternion (no rotation)
    server.scene.add_frame("/origin", wxyz=origin_rotation, position=np.array([0, 0, 0]), axes_length=0.8, axes_radius=0.015)
    
    # transformed coordinate frame (hand-eye calibration applied)
    # convert rotation matrix to quaternion for viser
    transformed_rotation = R.from_matrix(transformed_axes[:3, :3]).as_quat()
    transformed_quaternion = np.array([transformed_rotation[3], transformed_rotation[0], transformed_rotation[1], transformed_rotation[2]])  # wxyz format
    
    # apply the translation part of the transformation
    transformed_position = transformed_axes[:3, 3] # extract translation from 4x4 matrix
    server.scene.add_frame("/transformed_axes", wxyz=transformed_quaternion, position=transformed_position, axes_length=0.8, axes_radius=0.015)
    
    # # add a green point at the origin (original)
    # print("Adding green point at PLY file origin...")
    # origin_point = np.array([[0, 0, 0]]) # origin coordinates
    # origin_color = np.array([[0, 255, 0]])
    # server.scene.add_point_cloud(
    #     "/origin_point",
    #     points=origin_point,
    #     colors=origin_color,
    #     point_size=0.1,  # Larger for visibility
    # )
    
    # # add a blue point at the transformed origin
    # print("Adding blue point at transformed origin...")
    # transformed_origin_point = apply_transformation_to_points(origin_point, transformed_axes)
    # transformed_origin_color = np.array([[255, 0, 0]]) 
    # server.scene.add_point_cloud(
    #     "/transformed_origin_point",
    #     points=transformed_origin_point,
    #     colors=transformed_origin_color,
    #     point_size=0.1,
    # )
    
    # # add a purple dot at (1, 0, 1) in original coordinate system
    # print("Adding purple dot at (1, 0, 1) in original coordinates...")
    # original_test_point = np.array([[1, 0, 1]]) # point in original coordinate system
    # original_test_color = np.array([[255, 0, 255]]) 
    # server.scene.add_point_cloud(
    #     "/original_test_point",
    #     points=original_test_point,
    #     colors=original_test_color,
    #     point_size=0.1,
    # )
    
    # # show the same point in transformed coordinate system
    # print("Showing same point in transformed coordinate system...")
    # transformed_test_point = apply_transformation_to_points(original_test_point, transformed_axes)
    # transformed_test_color = np.array([[255, 255, 0]])
    # server.scene.add_point_cloud(
    #     "/transformed_test_point",
    #     points=transformed_test_point,
    #     colors=transformed_test_color,
    #     point_size=0.1,
    # )

    print("Visualization started!")

    while True:
        time.sleep(0.5)
