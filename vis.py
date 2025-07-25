import random
import time
import numpy as np
import viser
import math
import cv2
# target_movement = [y=0.294, x=0.000, theta=22.5°]
from scipy.spatial.transform import Rotation as R


wheel_start_pose_0 = [0.000, 0.000, 0.0]  # x, y, theta (degrees)
wheel_end_pose_0 = [0.295, 0.000, 22.5]  # x, y, theta (degrees)
cam_start_pose_0 = [1.613, -0.025, 0.658, 0.042132, 0.269313, 0.053504, 0.960642]  # xyz, wxyz
cam_end_pose_0 = [1.371, 0.063, 0.204, 0.060310, 0.135636, 0.039643, 0.988127]  # xyz, wxyz

wheel_start_pose_1 = [0.295, -0.000, 22.5]  # x, y, theta (degrees)
wheel_end_pose_1 = [0.567, 0.113, 45.1]  # x, y, theta (degrees)
cam_start_pose_1 = [1.484, 0.106, 0.254, 0.071914, 0.057991, 0.032786, 0.995184]  # xyz, wxyz
cam_end_pose_1 = [1.569, 0.095, 0.337, 0.076506, 0.054139, 0.027955, 0.995206]  # xyz, wxyz

wheel_start_pose_2 = [0.566, 0.112, 45.0]  # x, y, theta (degrees)
wheel_end_pose_2 = [0.775, 0.320, 67.4]  # x, y, theta (degrees)
cam_start_pose_2 = [1.644, 0.137, 0.510, 0.087999, -0.141576, 0.018991, 0.985825]  # xyz, wxyz
cam_end_pose_2 = [1.635, 0.117, 0.686, 0.092474, -0.306174, 0.001773, 0.947472]  # xyz, wxyz

wheel_start_pose_3 = [0.775, 0.320, 67.5]  # x, y, theta (degrees)
wheel_end_pose_3 = [0.887, 0.592, 90.0]  # x, y, theta (degrees)
cam_start_pose_3 = [1.604, 0.114, 0.750, 0.095951, -0.332223, 0.000158, 0.938308]  # xyz, wxyz
cam_end_pose_3 = [1.621, 0.089, 0.864, 0.094145, -0.507060, -0.017871, 0.856567]  # xyz, wxyz

wheel_start_pose_4 = [0.888, 0.592, 90.0]  # x, y, theta (degrees)
wheel_end_pose_4 = [0.887, 0.886, 112.5]  # x, y, theta (degrees)
cam_start_pose_4 = [1.604, 0.094, 0.894, 0.095964, -0.517236, -0.018057, 0.850254]  # xyz, wxyz
cam_end_pose_4 = [1.463, 0.084, 1.003, 0.101290, -0.616556, -0.023326, 0.780419]  # xyz, wxyz

wheel_start_pose_5 = [0.887, 0.887, 112.5]  # x, y, theta (degrees)
wheel_end_pose_5 = [0.778, 1.158, 134.8]  # x, y, theta (degrees)
cam_start_pose_5 = [1.397, 0.053, 1.066, 0.099586, -0.672314, -0.032764, 0.732805]  # xyz, wxyz
cam_end_pose_5 = [1.176, 0.044, 1.084, 0.101732, -0.752173, -0.042902, 0.649650]  # xyz, wxyz

wheel_start_pose_6 = [0.775, 1.158, 134.9]  # x, y, theta (degrees)
wheel_end_pose_6 = [0.568, 1.367, 157.4]  # x, y, theta (degrees)
cam_start_pose_6 = [1.131, 0.033, 1.101, 0.099513, -0.802825, -0.051637, 0.585580]  # xyz, wxyz
cam_end_pose_6 = [1.093, -0.003, 1.142, 0.083582, -0.889283, -0.069003, 0.444328]  # xyz, wxyz

wheel_start_pose_7 = [0.567, 1.367, 157.3]  # x, y, theta (degrees)
wheel_end_pose_7 = [0.293, 1.478, 179.6]  # x, y, theta (degrees)
cam_start_pose_7 = [1.086, -0.089, 1.365, 0.081452, -0.875221, -0.068192, 0.471914]  # xyz, wxyz
cam_end_pose_7 = [0.737, -0.073, 1.167, 0.068101, -0.940973, -0.076547, 0.322602]  # xyz, wxyz

wheel_start_pose_8 = [0.295, 1.480, 179.9]  # x, y, theta (degrees)
wheel_end_pose_8 = [0.001, 1.481, 202.4]  # x, y, theta (degrees)
cam_start_pose_8 = [0.758, -0.085, 1.237, 0.066722, -0.949187, -0.077885, 0.297534]  # xyz, wxyz
cam_end_pose_8 = [0.463, -0.050, 0.869, 0.052936, -0.982472, -0.089128, 0.154930]  # xyz, wxyz

wheel_start_pose_9 = [0.001, 1.481, 202.4]  # x, y, theta (degrees)
wheel_end_pose_9 = [-0.272, 1.369, 224.8]  # x, y, theta (degrees)
cam_start_pose_9 = [0.398, -0.032, 0.841, 0.052105, -0.990033, -0.087928, 0.096889]  # xyz, wxyz
cam_end_pose_9 = [0.320, -0.028, 0.830, 0.042324, -0.994438, -0.092590, -0.026983]  # xyz, wxyz

wheel_start_pose_10 = [-0.271, 1.368, 224.9]  # x, y, theta (degrees)
wheel_end_pose_10 = [-0.480, 1.161, 247.4]  # x, y, theta (degrees)
cam_start_pose_10 = [0.305, 0.002, 0.728, 0.034676, -0.989441, -0.091803, -0.106664]  # xyz, wxyz
cam_end_pose_10 = [0.266, 0.017, 0.528, 0.011012, -0.953261, -0.093633, -0.287062]  # xyz, wxyz

wheel_start_pose_11 = [-0.480, 1.160, 247.5]  # x, y, theta (degrees)
wheel_end_pose_11 = [-0.592, 0.888, 270.0]  # x, y, theta (degrees)
cam_start_pose_11 = [0.249, 0.018, 0.528, 0.008689, -0.946013, -0.092764, -0.310450]  # xyz, wxyz
cam_end_pose_11 = [0.374, 0.073, 0.314, -0.005825, -0.894434, -0.090784, -0.437850]  # xyz, wxyz

wheel_start_pose_12 = [-0.592, 0.888, 270.2]  # x, y, theta (degrees)
wheel_end_pose_12 = [-0.592, 0.593, 292.5]  # x, y, theta (degrees)
cam_start_pose_12 = [0.389, 0.096, 0.234, -0.009723, -0.867485, -0.089460, -0.489257]  # xyz, wxyz
cam_end_pose_12 = [0.616, 0.158, 0.049, -0.026668, -0.784594, -0.085875, -0.613455]  # xyz, wxyz

wheel_start_pose_13 = [-0.592, 0.594, 292.7]  # x, y, theta (degrees)
wheel_end_pose_13 = [-0.480, 0.321, 314.9]  # x, y, theta (degrees)
cam_start_pose_13 = [0.736, 0.193, 0.013, -0.037160, -0.745390, -0.078751, -0.660917]  # xyz, wxyz
cam_end_pose_13 = [0.741, 0.197, -0.064, -0.051600, -0.628347, -0.072967, -0.772783]  # xyz, wxyz

wheel_start_pose_14 = [-0.479, 0.322, 315.2]  # x, y, theta (degrees)
wheel_end_pose_14 = [-0.270, 0.115, 337.7]  # x, y, theta (degrees)
cam_start_pose_14 = [0.762, 0.196, -0.073, -0.054850, -0.596592, -0.072352, -0.797393]  # xyz, wxyz
cam_end_pose_14 = [0.962, 0.206, -0.075, -0.072397, -0.435906, -0.057509, -0.895230]  # xyz, wxyz

wheel_start_pose_15 = [-0.270, 0.115, 337.7]  # x, y, theta (degrees)
wheel_end_pose_15 = [0.003, 0.004, 360.2]  # x, y, theta (degrees)
cam_start_pose_15 = [0.961, 0.212, -0.092, -0.074099, -0.416891, -0.054416, -0.904295]  # xyz, wxyz
cam_end_pose_15 = [1.065, 0.200, -0.031, -0.080559, -0.316492, -0.048290, -0.943934]  # xyz, wxyz

# WHEEL ODOMETRY START POSE: [x=0.775, y=0.320, theta=67.5°]
# WHEEL ODOMETRY END POSE:   [x=0.887, y=0.592, theta=90.0°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=1.604, y=0.114, z=0.750][w=0.095951, x=-0.332223, y=0.000158, z=0.938308]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=1.621, y=0.089, z=0.864][w=0.094145, x=-0.507060, y=-0.017871, z=0.856567]

# WHEEL ODOMETRY START POSE: [x=0.888, y=0.592, theta=90.0°]
# WHEEL ODOMETRY END POSE:   [x=0.887, y=0.886, theta=112.5°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=1.604, y=0.094, z=0.894][w=0.095964, x=-0.517236, y=-0.018057, z=0.850254]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=1.463, y=0.084, z=1.003][w=0.101290, x=-0.616556, y=-0.023326, z=0.780419]

# WHEEL ODOMETRY START POSE: [x=0.887, y=0.887, theta=112.5°]
# WHEEL ODOMETRY END POSE:   [x=0.778, y=1.158, theta=134.8°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=1.397, y=0.053, z=1.066][w=0.099586, x=-0.672314, y=-0.032764, z=0.732805]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=1.176, y=0.044, z=1.084][w=0.101732, x=-0.752173, y=-0.042902, z=0.649650]

# WHEEL ODOMETRY START POSE: [x=0.775, y=1.158, theta=134.9°]
# WHEEL ODOMETRY END POSE:   [x=0.568, y=1.367, theta=157.4°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=1.131, y=0.033, z=1.101][w=0.099513, x=-0.802825, y=-0.051637, z=0.585580]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=1.093, y=-0.003, z=1.142][w=0.083582, x=-0.889283, y=-0.069003, z=0.444328]

# WHEEL ODOMETRY START POSE: [x=0.567, y=1.367, theta=157.3°]
# WHEEL ODOMETRY END POSE:   [x=0.293, y=1.478, theta=179.6°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=1.086, y=-0.089, z=1.365][w=0.081452, x=-0.875221, y=-0.068192, z=0.471914]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=0.737, y=-0.073, z=1.167][w=0.068101, x=-0.940973, y=-0.076547, z=0.322602]

# WHEEL ODOMETRY START POSE: [x=0.295, y=1.480, theta=179.9°]
# WHEEL ODOMETRY END POSE:   [x=0.001, y=1.481, theta=202.4°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=0.758, y=-0.085, z=1.237][w=0.066722, x=-0.949187, y=-0.077885, z=0.297534]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=0.463, y=-0.050, z=0.869][w=0.052936, x=-0.982472, y=-0.089128, z=0.154930]

# WHEEL ODOMETRY START POSE: [x=0.001, y=1.481, theta=202.4°]
# WHEEL ODOMETRY END POSE:   [x=-0.272, y=1.369, theta=224.8°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=0.398, y=-0.032, z=0.841][w=0.052105, x=-0.990033, y=-0.087928, z=0.096889]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=0.320, y=-0.028, z=0.830][w=0.042324, x=-0.994438, y=-0.092590, z=-0.026983]

# WHEEL ODOMETRY START POSE: [x=-0.271, y=1.368, theta=224.9°]
# WHEEL ODOMETRY END POSE:   [x=-0.480, y=1.161, theta=247.4°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=0.305, y=0.002, z=0.728][w=0.034676, x=-0.989441, y=-0.091803, z=-0.106664]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=0.266, y=0.017, z=0.528][w=0.011012, x=-0.953261, y=-0.093633, z=-0.287062]

# WHEEL ODOMETRY START POSE: [x=-0.480, y=1.160, theta=247.5°]
# WHEEL ODOMETRY END POSE:   [x=-0.592, y=0.888, theta=270.0°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=0.249, y=0.018, z=0.528][w=0.008689, x=-0.946013, y=-0.092764, z=-0.310450]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=0.374, y=0.073, z=0.314][w=-0.005825, x=-0.894434, y=-0.090784, z=-0.437850]

# WHEEL ODOMETRY START POSE: [x=-0.592, y=0.888, theta=270.2°]
# WHEEL ODOMETRY END POSE:   [x=-0.592, y=0.593, theta=292.5°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=0.389, y=0.096, z=0.234][w=-0.009723, x=-0.867485, y=-0.089460, z=-0.489257]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=0.616, y=0.158, z=0.049][w=-0.026668, x=-0.784594, y=-0.085875, z=-0.613455]

# WHEEL ODOMETRY START POSE: [x=-0.592, y=0.594, theta=292.7°]
# WHEEL ODOMETRY END POSE:   [x=-0.480, y=0.321, theta=314.9°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=0.736, y=0.193, z=0.013][w=-0.037160, x=-0.745390, y=-0.078751, z=-0.660917]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=0.741, y=0.197, z=-0.064][w=-0.051600, x=-0.628347, y=-0.072967, z=-0.772783]

# WHEEL ODOMETRY START POSE: [x=-0.479, y=0.322, theta=315.2°]
# WHEEL ODOMETRY END POSE:   [x=-0.270, y=0.115, theta=337.7°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=0.762, y=0.196, z=-0.073][w=-0.054850, x=-0.596592, y=-0.072352, z=-0.797393]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=0.962, y=0.206, z=-0.075][w=-0.072397, x=-0.435906, y=-0.057509, z=-0.895230]

# WHEEL ODOMETRY START POSE: [x=-0.270, y=0.115, theta=337.7°]
# WHEEL ODOMETRY END POSE:   [x=0.003, y=0.004, theta=360.2°]
# CAMERA ESTIMATED START POSITION (SLAM): [x=0.961, y=0.212, z=-0.092][w=-0.074099, x=-0.416891, y=-0.054416, z=-0.904295]
# CAMERA ESTIMATED END POSITION (SLAM):   [x=1.065, y=0.200, z=-0.031][w=-0.080559, x=-0.316492, y=-0.048290, z=-0.943934]

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

cam_poses = [
    cam_start_pose_0, cam_end_pose_0,
    cam_start_pose_1, cam_end_pose_1,
    cam_start_pose_2, cam_end_pose_2,
    cam_start_pose_3, cam_end_pose_3,
    cam_start_pose_4, cam_end_pose_4,
    cam_start_pose_5, cam_end_pose_5,
    cam_start_pose_6, cam_end_pose_6,
    cam_start_pose_7, cam_end_pose_7,
    cam_start_pose_8, cam_end_pose_8,
    cam_start_pose_9, cam_end_pose_9,
    cam_start_pose_10, cam_end_pose_10,
    cam_start_pose_11, cam_end_pose_11,
    cam_start_pose_12, cam_end_pose_12,
    cam_start_pose_13, cam_end_pose_13,
    cam_start_pose_14, cam_end_pose_14,
    cam_start_pose_15, cam_end_pose_15
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
  server.set_up_direction((0.0, -1.0, 0.0)) 

  wheel_translations = [np.array([pose[0], pose[1], 10.0]) for pose in wheel_poses]
  wheel_rotations = [rotation_matrix_z(pose[2]) for pose in wheel_poses]

  cam_translations = [np.array(pose[0:3]) for pose in cam_poses]
  cam_rotations = [quaternion_wxyz_to_rotation_matrix_scipy(pose[3:]) for pose in cam_poses]

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

    T_base2cam = np.linalg.inv(T_cam2base) @ wheel_pose
    # wheel_pose = T_cam2base @ wheel_pose
    wheel_pose = T_base2cam @ wheel_pose
    wheel_position = wheel_pose[0:3, 3]
    wheel_wxyz = R.from_matrix(wheel_pose[0:3, 0:3]).as_quat(scalar_first=True)

    server.scene.add_frame(f"/wheel_C/{i}", wxyz=wheel_wxyz, position=wheel_position,
                           axes_length=0.294, axes_radius=0.01)

  while True:
      # Add some coordinate frames to the scene. These will be visualized in the viewer.
      time.sleep(0.5)