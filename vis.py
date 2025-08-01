import random
import time
import numpy as np
import viser
import math
import cv2
# target_movement = [y=0.294, x=0.000, theta=22.5°]
from scipy.spatial.transform import Rotation as R


def generate_tilted_circle_cameras():
    # Circle parameters
    radius = 2.0
    num_points = 16  # Match the number of wheel end poses

    # Generate points
    points = []
    for i in range(num_points):
        # Angle for each point (evenly spaced around circle)
        angle = 2 * np.pi * i / num_points
        
        # Circle points in XZ plane (Y=0)
        x = radius * np.cos(angle)
        y = 0.0
        z = radius * np.sin(angle)
        
        # Create rotation for camera to face inward toward origin
        # Camera needs to look from its position toward (0,0,0)
        
        # Direction from camera position to center
        look_dir = np.array([-x, -y, -z])
        look_dir = look_dir / np.linalg.norm(look_dir)
        
        # Forward vector (where camera looks)
        forward = look_dir
        
        # Up vector is world Y axis
        up = np.array([0, 1, 0])
        
        # Right vector = up × forward
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        # Recalculate up to ensure orthogonality
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        
        # Build rotation matrix (camera looks along +Z in its local frame)
        R_matrix = np.column_stack([right, up, forward])
        
        # Convert rotation matrix to quaternion
        rotation = R.from_matrix(R_matrix)
        q = rotation.as_quat(scalar_first=True)  # [w, x, y, z]
        
        # Format: [x, y, z, qw, qx, qy, qz]
        point = [x, y, z, q[0], q[1], q[2], q[3]]
        points.append(point)
        
        # Print each point
        # print(f"Camera Point {i}: [{x:.4f}, {y:.4f}, {z:.4f}, {q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    
    return points


wheel_start_pose_0 = [0.000, 0.000, 0.0]  # x, y, theta (degrees)
wheel_end_pose_0 = [0.295, 0.000, 22.5]  # x, y, theta (degrees)

wheel_start_pose_1 = [0.295, -0.000, 22.5]  # x, y, theta (degrees)
wheel_end_pose_1 = [0.567, 0.113, 45.1]  # x, y, theta (degrees)

wheel_start_pose_2 = [0.566, 0.112, 45.0]  # x, y, theta (degrees)
wheel_end_pose_2 = [0.775, 0.320, 67.4]  # x, y, theta (degrees)

wheel_start_pose_3 = [0.775, 0.320, 67.5]  # x, y, theta (degrees)
wheel_end_pose_3 = [0.887, 0.592, 90.0]  # x, y, theta (degrees)

wheel_start_pose_4 = [0.888, 0.592, 90.0]  # x, y, theta (degrees)
wheel_end_pose_4 = [0.887, 0.886, 112.5]  # x, y, theta (degrees)

wheel_start_pose_5 = [0.887, 0.887, 112.5]  # x, y, theta (degrees)
wheel_end_pose_5 = [0.778, 1.158, 134.8]  # x, y, theta (degrees)

wheel_start_pose_6 = [0.775, 1.158, 134.9]  # x, y, theta (degrees)
wheel_end_pose_6 = [0.568, 1.367, 157.4]  # x, y, theta (degrees)

wheel_start_pose_7 = [0.567, 1.367, 157.3]  # x, y, theta (degrees)
wheel_end_pose_7 = [0.293, 1.478, 179.6]  # x, y, theta (degrees)

wheel_start_pose_8 = [0.295, 1.480, 179.9]  # x, y, theta (degrees)
wheel_end_pose_8 = [0.001, 1.481, 202.4]  # x, y, theta (degrees)

wheel_start_pose_9 = [0.001, 1.481, 202.4]  # x, y, theta (degrees)
wheel_end_pose_9 = [-0.272, 1.369, 224.8]  # x, y, theta (degrees)

wheel_start_pose_10 = [-0.271, 1.368, 224.9]  # x, y, theta (degrees)
wheel_end_pose_10 = [-0.480, 1.161, 247.4]  # x, y, theta (degrees)

wheel_start_pose_11 = [-0.480, 1.160, 247.5]  # x, y, theta (degrees)
wheel_end_pose_11 = [-0.592, 0.888, 270.0]  # x, y, theta (degrees)

wheel_start_pose_12 = [-0.592, 0.888, 270.2]  # x, y, theta (degrees)
wheel_end_pose_12 = [-0.592, 0.593, 292.5]  # x, y, theta (degrees)

wheel_start_pose_13 = [-0.592, 0.594, 292.7]  # x, y, theta (degrees)
wheel_end_pose_13 = [-0.480, 0.321, 314.9]  # x, y, theta (degrees)

wheel_start_pose_14 = [-0.479, 0.322, 315.2]  # x, y, theta (degrees)
wheel_end_pose_14 = [-0.270, 0.115, 337.7]  # x, y, theta (degrees)

wheel_start_pose_15 = [-0.270, 0.115, 337.7]  # x, y, theta (degrees)
wheel_end_pose_15 = [0.003, 0.004, 360.2]  # x, y, theta (degrees)

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

def z_rotation_to_quaternion(theta_degrees):
    theta_radians = math.radians(theta_degrees)
    half_angle = theta_radians / 2
    w = math.cos(half_angle)
    x = 0.0
    y = 0.0
    z = math.sin(half_angle)
    return np.array([w, x, y, z])  # Format: [w, x, y, z]

def rotation_matrix_z(degrees):
    """
    Converts an angle in degrees around the Z-axis to a 3x3 rotation matrix.

    Args:
        degrees (float): The angle of rotation in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
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

def quaternion_to_rotation_matrix(q_wxyz):
    """
    Converts a quaternion in (w, x, y, z) format to a 3x3 rotation matrix.

    Args:
        q_wxyz (array_like): A 4-element array or list representing the quaternion
                             in (w, x, y, z) order (scalar part first).

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    w, x, y, z = q_wxyz

    # Normalize the quaternion to ensure it's a unit quaternion
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    if norm == 0:
        # Handle the case of a zero quaternion (no rotation)
        return np.eye(3)
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    # Calculate the terms for the rotation matrix
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    # Construct the rotation matrix
    R = np.array([
        [1 - 2*y2 - 2*z2, 2*xy - 2*zw, 2*xz + 2*yw],
        [2*xy + 2*zw, 1 - 2*x2 - 2*z2, 2*yz - 2*xw],
        [2*xz - 2*yw, 2*yz + 2*xw, 1 - 2*x2 - 2*y2]
    ])
    return R

def quaternion_wxyz_to_rotation_matrix_scipy(q_wxyz):
    """
    Converts a quaternion in (w, x, y, z) format to a 3x3 rotation matrix using SciPy.

    Args:
        q_wxyz (array_like): A 4-element array or list representing the quaternion
                             in (w, x, y, z) order (scalar part 'w' first).

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    # SciPy's Rotation class expects quaternions in (x, y, z, w) format.
    # So, we need to reorder the input.
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

    # Create a Rotation object from the quaternion
    # SciPy automatically handles normalization if the input is not a unit quaternion.
    rotation = R.from_quat(q_xyzw)

    # Convert the Rotation object to a rotation matrix
    rotation_matrix = rotation.as_matrix()

    return rotation_matrix

if __name__ == "__main__":
  server = viser.ViserServer()
  server.scene.set_up_direction((0.0, -1.0, 0.0)) 

  angle = 20

  while True:
    angle+=3
    server.scene.reset()

    # Generate synthetic camera poses using the circular arrangement
    synthetic_cam_poses = generate_tilted_circle_cameras()

    # Apply 20 degree rotation about x-axis to all synthetic poses
    x_rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(angle % 360)), -np.sin(np.radians(angle % 360))],
        [0, np.sin(np.radians(angle % 360)), np.cos(np.radians(angle % 360))]
    ])

    rotated_cam_poses = []
    for pose in synthetic_cam_poses:
        # Extract position and quaternion
        position = np.array(pose[:3])
        quat_wxyz = np.array(pose[3:])
        
        # Rotate position
        rotated_position = x_rotation_matrix @ position
        
        # Rotate orientation: convert quat to rotation matrix, apply rotation, convert back
        original_rotation = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # Convert to xyzw format
        original_rot_matrix = original_rotation.as_matrix()
        rotated_rot_matrix = x_rotation_matrix @ original_rot_matrix
        rotated_rotation = R.from_matrix(rotated_rot_matrix)
        rotated_quat_xyzw = rotated_rotation.as_quat()
        rotated_quat_wxyz = [rotated_quat_xyzw[3], rotated_quat_xyzw[0], rotated_quat_xyzw[1], rotated_quat_xyzw[2]]  # Convert back to wxyz
        
        # Create new pose
        rotated_pose = [rotated_position[0], rotated_position[1], rotated_position[2], 
                      rotated_quat_wxyz[0], rotated_quat_wxyz[1], rotated_quat_wxyz[2], rotated_quat_wxyz[3]]
        rotated_cam_poses.append(rotated_pose)

    # Create cam_poses list with rotated synthetic end poses
    cam_poses = []
    for i in range(16):
        cam_poses.append(rotated_cam_poses[i])  # Start pose
        cam_poses.append(rotated_cam_poses[i])  # End pose (same as start for synthetic data)



    wheel_translations = [np.array([pose[0], pose[1], 0.0]) for pose in wheel_poses]
    wheel_rotations = [rotation_matrix_z(pose[2]) for pose in wheel_poses]

    cam_translations = [np.array(pose[0:3]) for pose in cam_poses]
    cam_rotations = [quaternion_wxyz_to_rotation_matrix_scipy(pose[3:]) for pose in cam_poses]

    try:
      T_rot, T_trans = cv2.calibrateHandEye(
          R_gripper2base=wheel_rotations,
          t_gripper2base=wheel_translations,
          R_target2cam=cam_rotations,
          t_target2cam=cam_translations,
          method=4,
      )
    except:
      pass
    T_cam2base = np.eye(4)
    T_cam2base[0:3, 0:3] = T_rot
    T_cam2base[0:3, 3:4] = T_trans

    for i, cam_pose in enumerate(cam_poses):
      cam_wxyz = np.array(cam_pose[3:])
      cam_position = np.array(cam_pose[0:3])
      server.scene.add_frame(f"/cam/{i}", wxyz=cam_wxyz, position=cam_position,
                            axes_length=0.294, axes_radius=0.01)  
      # fov = 2 * np.arctan2(800 / 2, 90)
      # aspect = 800.0 / 600.0
      # server.scene.add_camera_frustum(f"/cam_frustum/{i}", fov, aspect, cam_wxyz, cam_position)  

    for i, wheel_pose in enumerate(wheel_poses):
      wheel_wxyz = z_rotation_to_quaternion(wheel_pose[2])
      wheel_position = np.array([wheel_pose[0], wheel_pose[1], 0.0])
      server.scene.add_frame(f"/wheel/{i}", wxyz=wheel_wxyz, position=wheel_position,
                            axes_length=0.294, axes_radius=0.01)    
  
    for i, wheel_pose in enumerate(wheel_poses):

      wheel_rot = rotation_matrix_z(wheel_pose[2])
      wheel_position = np.array([wheel_pose[0], wheel_pose[1], 0.0])
      wheel_pose = np.eye(4)
      wheel_pose[0:3, 0:3] = wheel_rot
      wheel_pose[0:3, 3] = wheel_position

      # Apply the hand-eye calibration transformation
      try:
        wheel_pose = np.linalg.inv(T_cam2base) @ wheel_pose
        wheel_position = wheel_pose[0:3, 3]
        wheel_wxyz = R.from_matrix(wheel_pose[0:3, 0:3]).as_quat(scalar_first=True)

        server.scene.add_frame(f"/wheel_C/{i}", wxyz=wheel_wxyz, position=wheel_position,
                              axes_length=0.294, axes_radius=0.01)
      except:
        pass
    time.sleep(0.05)