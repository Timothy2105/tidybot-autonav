import time
import numpy as np
import viser
import cv2
import argparse
import threading
import json
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import os

# baseline yaw file path
INITIAL_YAW_FILE = "calib-results/runtime/initial_yaw.txt"


def save_initial_yaw_to_file(yaw_deg, path = INITIAL_YAW_FILE):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(f"{yaw_deg:.6f}\n")
        print(f"Saved initial yaw {yaw_deg:.2f} deg to {path}")
        return True
    except Exception as e:
        print(f"Failed to save initial yaw: {e}")
        return False


def load_initial_yaw_from_file(path = INITIAL_YAW_FILE):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                return None
            return float(content)
    except Exception:
        return None

# return existing value, otherwise save and return it
def ensure_initial_yaw_saved(yaw_candidate_deg):
    existing = load_initial_yaw_from_file()
    if existing is None:
        save_initial_yaw_to_file(yaw_candidate_deg)
        return yaw_candidate_deg
    return existing


def load_transformation_matrix():
    transformation_matrix_file = "calib-results/final_transformation_matrix.npy"
    
    if os.path.exists(transformation_matrix_file):
        transformation_matrix = np.load(transformation_matrix_file)
        print(f"Loaded transformation matrix from {transformation_matrix_file}")
        return transformation_matrix
    else:
        print(f"Transformation matrix not found at {transformation_matrix_file}")
        print("Run this file with --save first to generate the matrix")


def read_camera_position():
    camera_position_file = "calib-results/runtime/camera_position.txt"
    max_retries = 100 
    retry_count = 0
    
    while retry_count < max_retries:
        if os.path.exists(camera_position_file):
            try:
                with open(camera_position_file, 'r') as f:
                    content = f.read().strip()
                    if not content: 
                        retry_count += 1
                        time.sleep(0.01)  
                        continue
                    
                    data = json.loads(content)
                    position = np.array([data['x'], data['y'], data['z']])
                    quaternion = data.get('quaternion', [1, 0, 0, 0])
                    timestamp = data.get('timestamp', time.time())
                    
                    return position, quaternion, timestamp
            except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                retry_count += 1
                time.sleep(0.01)  
                continue
            except Exception as e:
                retry_count += 1
                time.sleep(0.01)  
                continue
        else:
            retry_count += 1
            time.sleep(0.01)  
            continue
    
    return None, None, None


def apply_transformation(point, transformation_matrix):
    if transformation_matrix is None:
        print("No transformation matrix available, using original coordinates")
        return point
    
    # convert point to homogeneous coordinates
    point_homogeneous = np.append(point, 1.0)
    
    # apply transformation
    transformed_homogeneous = transformation_matrix @ point_homogeneous
    
    # convert back to 3D coordinates
    transformed_point = transformed_homogeneous[:3]
    
    return transformed_point


def flatten_point(point):
    return np.array([point[0], 0.0, point[2]])


def calculate_movement(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    
    # flatten both points
    flat_a = flatten_point(point_a)
    flat_b = flatten_point(point_b)
    
    # calculate net movement
    movement = flat_b - flat_a
    
    return {
        'point_a': point_a.tolist(),
        'point_b': point_b.tolist(),
        'flat_a': flat_a.tolist(),
        'flat_b': flat_b.tolist(),
        'movement': movement.tolist(),
        'distance': np.linalg.norm([movement[0], movement[2]]),
        'x_movement': movement[0],
        'y_movement': movement[1],
        'z_movement': movement[2]
    }


def save_movement_result(result, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Movement result saved to {filename}")


def save_planning_maps_json(heightmap, planning_map, eroded_map, x_edges, z_edges, output_dir="calib-results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # prepare grid info
    grid_info = {
        'height': heightmap.shape[0],
        'width': heightmap.shape[1],
        'x_min': float(x_edges[0]),
        'x_max': float(x_edges[-1]),
        'z_min': float(z_edges[0]),
        'z_max': float(z_edges[-1]),
        'cell_size': float(x_edges[1] - x_edges[0]),
        'coordinate_system': 'world coordinates (meters)',
        'map_values': {
            'free': 0,
            'obstacle': 1,
            'unknown': -1
        }
    }
    
    # save heightmap
    heightmap_data = heightmap.copy()
    heightmap_data[np.isnan(heightmap_data)] = -999.0
    
    planning_data = {
        'grid_info': grid_info,
        'heightmap': heightmap_data.tolist(),
        'planning_map': planning_map.tolist(),
        'eroded_map': eroded_map.tolist()
    }
    
    # save as JSON
    json_path = os.path.join(output_dir, "planning_maps.json")
    with open(json_path, 'w') as f:
        json.dump(planning_data, f, indent=2)
    print(f"Planning maps saved to: {json_path}")
    
    return json_path


def generate_planning_maps():
    try:
        ply_path = "calib-results/transformed_pointcloud.ply"
        if not os.path.exists(ply_path):
            print(f"Error: Transformed point cloud not found at {ply_path}")
            return False
        
        print("Generating planning maps from transformed point cloud...")
        print(f"Using PLY file: {ply_path}")
        
        import detect_obstacles
        
        # arguments for detect_obstacles.py
        class Args:
            def __init__(self):
                self.input = ply_path
                self.output = None
                self.cell_size = 0.10
                self.height_threshold = [0.3, 1.3]
                self.viz_clip = [-0.1, 2.0]
                self.xlim = None
                self.zlim = None
                self.height_axis = 1
                self.visualize = False
                self.erode = 3
                self.dilate = 2
                self.use_open3d = False
        
        args = Args()
        
        if args.use_open3d:
            points = detect_obstacles.load_xyz_from_ply_open3d(args.input)
        else:
            points = detect_obstacles.load_xyz_from_ply(args.input)
        
        print(f"Loaded {len(points)} points")
        
        # heightmap and planning map
        print("Creating heightmap and planning map...")
        heightmap, plan_map, x_edges, z_edges = detect_obstacles.heightmap_and_clearance(
            points,
            cell_size=args.cell_size,
            height_axis=args.height_axis,
            viz_clip=args.viz_clip,
            clear_low=args.height_threshold[0],
            clear_high=args.height_threshold[1]
        )
        
        print(f"Created {heightmap.shape[0]}x{heightmap.shape[1]} grid")
        print(f"Free cells: {np.sum(plan_map == 0)}")
        print(f"Obstacle cells: {np.sum(plan_map == 1)}")
        print(f"Unknown cells: {np.sum(plan_map == -1)}")
        
        # erode and dilate
        print("Applying morphological operations...")
        eroded_map = detect_obstacles.apply_morphological_operations(plan_map, erode_size=args.erode, dilate_size=args.dilate)
        
        # save planning maps
        print("Saving planning maps...")
        save_planning_maps_json(heightmap, plan_map, eroded_map, x_edges, z_edges)
        
        print("Planning maps generated successfully!")
        return True
        
    except Exception as e:
        import traceback
        print(f"Error generating planning maps: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False


def load_astar_planner():
    try:
        json_path = "calib-results/planning_maps.json"
        if not os.path.exists(json_path):
            print(f"Error: Planning maps not found at {json_path}")
            return None
        
        print(f"Loading A* planner from: {json_path}")
        
        # import A* planner
        from astar_planner import AStarPlanner
        
        planner = AStarPlanner(json_path)
        print("A* planner loaded successfully!")
        
        map_info = planner.get_map_info()
        print("Map info:")
        for key, value in map_info.items():
            print(f"  {key}: {value}")
        
        return planner
        
    except Exception as e:
        import traceback
        print(f"Error loading A* planner: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None


def visualize_astar_path(server, path_waypoints):
    try:
        try:
            server.scene["/astar_waypoints"].remove()
        except:
            pass
        
        if len(path_waypoints) < 1:
            return
        
        # convert path to 3D points
        path_points_3d = []
        for x, z in path_waypoints:
            path_points_3d.append([x, 0.3, z])
        
        path_points_3d = np.array(path_points_3d)
        
        # create colors for waypoints
        waypoint_colors = []
        for i in range(len(path_points_3d)):
            if i == 0:
                # start point - bright green
                waypoint_colors.append([0.0, 1.0, 0.0])
            elif i == len(path_points_3d) - 1:
                # end point - bright red  
                waypoint_colors.append([1.0, 0.0, 0.0])
            else:
                # intermediate points - bright blue
                waypoint_colors.append([0.0, 0.5, 1.0])
        
        # add waypoint markers
        server.scene.add_point_cloud(
            "/astar_waypoints",
            points=path_points_3d,
            colors=np.array(waypoint_colors),
            point_size=0.05,
        )
        
        print(f"Visualized A* path with {len(path_waypoints)} waypoints:")
        print(f"  Start: ({path_waypoints[0][0]:.2f}, {path_waypoints[0][1]:.2f})")
        print(f"  Goal:  ({path_waypoints[-1][0]:.2f}, {path_waypoints[-1][1]:.2f})")
        print(f"  {len(path_waypoints)-2} intermediate waypoints")
        
    except Exception as e:
        print(f"Error visualizing A* waypoints: {e}")


def send_tidybot_command(tidybot_command):
    command_file = "calib-results/runtime/robot_commands.txt"
    
    # {'type': 'tidybot_command', 'command': [y, x, theta]}
    command = {
        'type': 'tidybot_command',
        'command': tidybot_command
    }
    
    try:
        os.makedirs(os.path.dirname(command_file), exist_ok=True)
        with open(command_file, 'w') as f:
            json.dump(command, f, indent=2)
        print(f"Command sent to send_cmd.py: {command}")
        return True
    except Exception as e:
        print(f"Error sending command: {e}")
        return False


def wait_for_robot_completion(timeout=30.0):
    """Wait for robot command to complete by monitoring result file."""
    result_file = "calib-results/runtime/robot_results.txt"
    start_time = time.time()
    
    print("  Waiting for robot completion...")
    
    while time.time() - start_time < timeout:
        try:
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        result = json.loads(content)
                        if result.get('success', False):
                            # clear result file for next command
                            with open(result_file, 'w') as f:
                                f.write("")
                            return True
                        elif 'success' in result and not result['success']:
                            print(f"  Robot command failed: {result.get('message', 'Unknown error')}")
                            return False
        except Exception as e:
            pass
        
        time.sleep(0.2)
    
    print(f"  Timeout waiting for robot completion ({timeout}s)")
    return False


# send with dynamic re-planning after each waypoint
def send_astar_waypoint_sequence(path_waypoints, current_position, baseline_yaw, server):
    global astar_planner, transformation_matrix
    
    if not path_waypoints or len(path_waypoints) < 2:
        print("No valid A* path to execute")
        return False
    
    # store the final destination
    final_destination = path_waypoints[-1]
    print(f"Starting dynamic A* navigation to final destination: ({final_destination[0]:.2f}, {final_destination[1]:.2f})")
    
    waypoint_count = 0
    max_waypoints = 20
    
    while waypoint_count < max_waypoints:
        # get current camera position and transform
        try:
            current_camera_position, current_quaternion, _ = read_camera_position()
            if transformation_matrix is not None:
                current_homogeneous = np.append(current_camera_position, 1)
                transformed_current = transformation_matrix @ current_homogeneous
                current_world_pos = (transformed_current[0], transformed_current[2])
            else:
                current_world_pos = (current_camera_position[0], current_camera_position[2])
            
            print(f"\nStep {waypoint_count + 1}: Current position: ({current_world_pos[0]:.2f}, {current_world_pos[1]:.2f})")
            
            distance_to_goal = np.sqrt((final_destination[0] - current_world_pos[0])**2 + 
                                     (final_destination[1] - current_world_pos[1])**2)
            
            if distance_to_goal < 0.3:  # within 30cm of destination
                print(f"Reached destination! Distance to goal: {distance_to_goal:.3f}m")
                break
            
            # recalculate safe path from current position to destination
            print(f"Recalculating safe path from current position to destination...")
            current_path = astar_planner.plan_safe_path(
                current_world_pos[0], current_world_pos[1],
                final_destination[0], final_destination[1],
                use_eroded=True, simplify="string_pull"
            )
            
            if not current_path or len(current_path) < 2:
                print("Failed to find safe path from current position to destination!")
                return False
            
            print(f"Found safe path with {len(current_path)} waypoints")
            
            # choose first waypoint at least MIN_HOP m. away
            MIN_HOP = 0.20
            next_waypoint = current_path[1]  # default next step
            for j in range(len(current_path) - 1, 0, -1):
                dx = current_path[j][0] - current_world_pos[0]
                dz = current_path[j][1] - current_world_pos[1]
                if np.hypot(dx, dz) >= MIN_HOP:
                    next_waypoint = current_path[j]
                    print(f"Runtime guard: Selected waypoint {j} instead of 1 (distance: {np.hypot(dx, dz):.3f}m)")
                    break
            
            # update visualization with current path
            try:
                visualize_astar_path(server, current_path)
                print(f"Updated visualization: safe path (two-stage if needed)")
            except Exception as e:
                print(f"Warning: Could not update path visualization: {e}")
            
            print(f"Selected next waypoint: ({next_waypoint[0]:.2f}, {next_waypoint[1]:.2f})")
            
            # calculate direction to next waypoint
            world_x_movement = next_waypoint[0] - current_world_pos[0]
            world_z_movement = next_waypoint[1] - current_world_pos[1]
            
            # get current robot yaw orientation
            camera_transform = create_camera_transform_matrix(current_camera_position, current_quaternion)
            transformed_camera_matrix = apply_transformation_to_4x4_matrix(camera_transform, transformation_matrix)
            transformed_rotation, _ = extract_rotation_and_translation(transformed_camera_matrix)
            current_yaw_deg, _ = get_camera_yaw_angle(transformed_rotation, forward_is_neg_z=False)
            
            # calculate target yaw angle
            target_yaw_rad = np.arctan2(world_x_movement, world_z_movement)
            target_yaw_deg = np.degrees(target_yaw_rad)
            
            # calculate rotation difference
            yaw_difference = target_yaw_deg - current_yaw_deg
            
            # normalize to [-180, 180] range
            while yaw_difference > 180:
                yaw_difference -= 360
            while yaw_difference < -180:
                yaw_difference += 360
            
            # calculate distance to move forward
            forward_distance = np.sqrt(world_x_movement**2 + world_z_movement**2)
            
            print(f"  Next waypoint: ({next_waypoint[0]:.2f}, {next_waypoint[1]:.2f})")
            print(f"  Direction: ({world_x_movement:.3f}, {world_z_movement:.3f})")
            print(f"  Current yaw: {current_yaw_deg:.1f}°")
            print(f"  Target yaw: {target_yaw_deg:.1f}°")
            print(f"  Yaw difference: {yaw_difference:.1f}°")
            print(f"  Forward distance: {forward_distance:.3f}m")
            
            # rotate to face target direction
            rotation_command = [0.0, 0.0, yaw_difference]
            print(f"  Rotation command: [0.0, 0.0, {yaw_difference:.1f}]")
            
            if send_tidybot_command(rotation_command):
                print(f"  Sent rotation command successfully")
                
                if wait_for_robot_completion():
                    print(f"  Rotation completed!")
                else:
                    print(f"  Warning: Rotation may not have completed properly")
                
                print(f"  Waiting 1.0s for SLAM position update after rotation...")
                time.sleep(1.0)
                    
                # move forward in the now-aligned direction
                robot_current_yaw = target_yaw_deg
                yaw_difference_from_initial = robot_current_yaw - baseline_yaw
                
                # robot wants to move forward in curr dir
                local_forward = forward_distance
                local_right = 0.0
                
                # convert local movement to initial yaw's global frame
                yaw_diff_rad = np.radians(yaw_difference_from_initial)
                global_x_in_initial_frame =  local_forward * np.sin(yaw_diff_rad) + local_right * np.cos(yaw_diff_rad)
                global_y_in_initial_frame =  local_forward * np.cos(yaw_diff_rad) - local_right * np.sin(yaw_diff_rad)
                
                # TidyBot command in initial yaw's global frame
                forward_command = [global_y_in_initial_frame, global_x_in_initial_frame, 0.0]
                
                print(f"  Robot current yaw: {robot_current_yaw:.1f}°")
                print(f"  Initial baseline yaw: {baseline_yaw:.1f}°")
                print(f"  Yaw difference from initial: {yaw_difference_from_initial:.1f}°")
                print(f"  Local movement: forward={local_forward:.3f}m, right={local_right:.3f}m")
                print(f"  Global movement (initial yaw frame): x={global_x_in_initial_frame:.3f}m, y={global_y_in_initial_frame:.3f}m")
                print(f"  Forward command (initial yaw frame): [{forward_command[0]:.3f}, {forward_command[1]:.3f}, {forward_command[2]:.1f}]")
                
                if send_tidybot_command(forward_command):
                    print(f"  Sent forward command successfully")
                    
                    if wait_for_robot_completion():
                        print(f"  Forward movement completed!")
                    else:
                        print(f"  Warning: Forward movement may not have completed properly")
                    
                    print(f"  Waiting 1.0s for robot to settle and SLAM position update...")
                    time.sleep(1.0)
                    
                    print(f"  Refreshing camera position for next iteration...")
                    try:
                        for i in range(3):
                            test_pos, test_quat, _ = read_camera_position()
                            time.sleep(0.1)
                        print(f"  Position refresh completed")
                    except Exception as e:
                        print(f"  Warning: Could not refresh position: {e}")
                    
                    waypoint_count += 1
                else:
                    print(f"  Failed to send forward command")
                    return False
            else:
                print(f"  Failed to send rotation command")
                return False
                
        except Exception as e:
            print(f"Error during dynamic re-planning: {e}")
            return False
    
    if waypoint_count >= max_waypoints:
        print(f"Reached maximum waypoint limit ({max_waypoints}), stopping navigation")
        return False
    
    print(f"\nDynamic A* navigation completed! Executed {waypoint_count} waypoints with re-planning.")
    return True


def get_next_waypoint_command():
    sequence_file = "calib-results/runtime/astar_waypoint_sequence.json"
    
    try:
        if not os.path.exists(sequence_file):
            return None
        
        with open(sequence_file, 'r') as f:
            sequence_data = json.load(f)
        
        if sequence_data['status'] != 'ready':
            return None
        
        current_waypoint = sequence_data['current_waypoint']
        total_waypoints = sequence_data['total_waypoints']
        
        if current_waypoint >= total_waypoints - 1:
            # sequence completed
            sequence_data['status'] = 'completed'
            with open(sequence_file, 'w') as f:
                json.dump(sequence_data, f, indent=2)
            print("A* waypoint sequence completed!")
            return None
        
        # get next waypoint command
        next_waypoint_idx = current_waypoint + 1
        next_command_data = sequence_data['commands'][next_waypoint_idx]
        
        # update sequence progress
        sequence_data['current_waypoint'] = next_waypoint_idx
        with open(sequence_file, 'w') as f:
            json.dump(sequence_data, f, indent=2)
        
        print(f"Next waypoint ({next_waypoint_idx + 1}/{total_waypoints}): {next_command_data['command']}")
        return next_command_data['command']
        
    except Exception as e:
        print(f"Error getting next waypoint command: {e}")
        return None


def check_waypoint_sequence_status():
    sequence_file = "calib-results/runtime/astar_waypoint_sequence.json"
    
    try:
        if not os.path.exists(sequence_file):
            return {'status': 'none', 'progress': None}
        
        with open(sequence_file, 'r') as f:
            sequence_data = json.load(f)
        
        return {
            'status': sequence_data['status'],
            'current_waypoint': sequence_data['current_waypoint'],
            'total_waypoints': sequence_data['total_waypoints'],
            'progress': f"{sequence_data['current_waypoint'] + 1}/{sequence_data['total_waypoints']}"
        }
        
    except Exception as e:
        print(f"Error checking waypoint sequence status: {e}")
        return {'status': 'error', 'progress': None}


def add_camera_position_dot(server, transformation_matrix):    
    # read current camera position
    position, quaternion, _ = read_camera_position()
    
    if position is not None and quaternion is not None:
        # print original camera data
        print(f"\nOriginal Camera Position: {position}")
        print(f"Original Camera Quaternion [w,x,y,z]: {quaternion}")
        
        # get original rotation matrix (in camera local coordinates)
        original_rotation = quaternion_wxyz_to_rotation_matrix_scipy(quaternion)
        print(f"Original Rotation Matrix (3x3) - Camera Local Coordinates:")
        print(original_rotation)
        
        # create 4x4 camera transformation matrix
        camera_transform = create_camera_transform_matrix(position, quaternion)
        
        # apply hand-eye calibration transformation to get world coordinates
        transformed_camera_matrix = apply_transformation_to_4x4_matrix(camera_transform, transformation_matrix)
        
        # extract rotation matrix in world coordinates
        transformed_rotation, transformed_translation = extract_rotation_and_translation(transformed_camera_matrix)
        
        # calculate Euler angles with respect to world coordinate system
        world_euler = rotation_to_euler(transformed_rotation)
        print(f"World Euler Angles (around x, y, z) in degrees: {world_euler}")
        
        # calculate camera tilt angle relative to world z-axis
        signed_yaw, _ = get_camera_yaw_angle(transformed_rotation)
        print(f"Camera yaw relative to world z-axis: {signed_yaw:.1f} degrees")
        
        # save the first yaw as baseline
        ensure_initial_yaw_saved(signed_yaw)
        
        transformed_position = apply_transformation(position, transformation_matrix)
        

        
        # add red dot 
        server.scene.add_point_cloud(
            "/camera_position",
            points=np.array([transformed_position]),
            colors=np.array([[1.0, 0.0, 0.0]]),
            point_size=0.05,
        )
        
        # test tilt angle
        yaw_rad = np.radians(signed_yaw)
        cos_tilt = np.cos(yaw_rad)
        sin_tilt = np.sin(yaw_rad)
        
        # rotation matrix for tilt around y-axis
        tilt_rotation_matrix = np.array([
            [cos_tilt,  0, sin_tilt],
            [0,         1, 0],
            [-sin_tilt, 0, cos_tilt]
        ])
        
        # convert to quaternion
        tilt_quaternion = R.from_matrix(tilt_rotation_matrix).as_quat()
        tilt_quaternion_wxyz = np.array([tilt_quaternion[3], tilt_quaternion[0], tilt_quaternion[1], tilt_quaternion[2]])
        
        # add test frame at origin
        server.scene.add_frame(
            "/yaw_axis",
            wxyz=tilt_quaternion_wxyz,
            position=np.array([0, 0, 0]),  
            axes_length=0.3,
            axes_radius=0.015,
        )
        
        print(f"Added red dot and tilt test frame at origin with {yaw_rad:.1f}° tilt")
        return transformed_position
    else:
        print("Could not read camera position")
        return None


def process_clicked_point(clicked_point, transformation_matrix, astar_planner=None, server=None, status_display=None):
    # get current camera position
    current_camera_position, current_quaternion, _ = read_camera_position()
    
    if current_camera_position is None:
        print("Could not read current camera position from calib-results/runtime/camera_position.txt")
        print("Make sure main.py is running and writing camera position")
        return None
    
    destination_point = clicked_point
    
    print(f"Current Position (Point A): {current_camera_position}")
    print(f"Destination (Point B): {destination_point}")
    
    # apply transformation to current camera position
    transformed_current = apply_transformation(current_camera_position, transformation_matrix)
    
    print(f"Transformed Current Position: {transformed_current}")
    print(f"Destination (already transformed): {destination_point}")
    
    # get current camera Euler angles and world Euler angles
    if current_quaternion is not None:
        current_rotation = quaternion_wxyz_to_rotation_matrix_scipy(current_quaternion)
        current_euler = rotation_to_euler(current_rotation)
        print(f"Current Camera Euler Angles (around x, y, z): {current_euler}")
        
        # get world Euler angles by applying transformation
        camera_transform = create_camera_transform_matrix(current_camera_position, current_quaternion)
        transformed_camera_matrix = apply_transformation_to_4x4_matrix(camera_transform, transformation_matrix)
        transformed_rotation, _ = extract_rotation_and_translation(transformed_camera_matrix)
        world_euler = rotation_to_euler(transformed_rotation)
    
    movement_result = calculate_movement(transformed_current, destination_point)
    
    if movement_result:
        print(f"Movement calculation:")
        print(f"  Current Position: {movement_result['point_a']}")
        print(f"  Destination: {movement_result['point_b']}")
        print(f"  World X movement: {movement_result['x_movement']:.3f}m")
        print(f"  World Y movement: {movement_result['y_movement']:.3f}m")
        print(f"  World Z movement: {movement_result['z_movement']:.3f}m")
        
        # transform world movement to robot global coordinates w/ baseline yaw
        world_x = movement_result['x_movement']
        world_z = movement_result['z_movement']  # using z as the second coordinate
        world_y = movement_result['y_movement']  # y stays the same
        
        # use initial yaw as baseline 
        initial_yaw = load_initial_yaw_from_file()
        if initial_yaw is None:
            camera_transform = create_camera_transform_matrix(current_camera_position, current_quaternion)
            transformed_camera_matrix = apply_transformation_to_4x4_matrix(camera_transform, transformation_matrix)
            transformed_rotation, _ = extract_rotation_and_translation(transformed_camera_matrix)
            current_signed_yaw, _ = get_camera_yaw_angle(transformed_rotation, forward_is_neg_z=False)
            baseline_yaw = ensure_initial_yaw_saved(current_signed_yaw)
            print(f"  Baseline yaw not found; using current yaw {baseline_yaw:.1f}° and saving it.")
        else:
            baseline_yaw = initial_yaw
            print(f"  Using baseline yaw: {baseline_yaw:.1f}°")
        yaw_rad = np.radians(baseline_yaw)
        
        # apply 2D rotation transformation with baseline yaw
        global_x =  world_x * np.cos(-yaw_rad) + world_z * np.sin(-yaw_rad)
        global_z = -world_x * np.sin(-yaw_rad) + world_z * np.cos(-yaw_rad)
        
        print(f"  Global X movement: {global_x:.3f}m")
        print(f"  Global Y movement: {world_y:.3f}m")
        print(f"  Global Z movement: {global_z:.3f}m")
        print(f"  Using baseline yaw: {baseline_yaw:.1f} degrees")
        
        # A* path planning
        path_waypoints = None
        status_message = "Unknown status"
        
        if astar_planner is not None:
            try:
                print("A* path planning:")
                start_x, start_z = transformed_current[0], transformed_current[2]
                goal_x, goal_z = destination_point[0], destination_point[2]
                
                # check if destination is in a free cell
                goal_row, goal_col = astar_planner.world_to_grid(goal_x, goal_z)
                start_row, start_col = astar_planner.world_to_grid(start_x, start_z)
                
                print(f"Start grid position: ({start_row}, {start_col})")
                print(f"Goal grid position: ({goal_row}, {goal_col})")
                
                # check if goal is valid and free
                if not astar_planner.is_valid(goal_row, goal_col):
                    status_message = "LOCATION OUT OF BOUNDS - Click within the mapped area"
                    print("Goal position is outside the mapped area!")
                    if status_display:
                        status_display.value = status_message
                    return None
                elif not astar_planner.is_free(goal_row, goal_col, use_eroded=True):
                    # check what type of obstacle
                    planning_val = astar_planner.planning_map[goal_row, goal_col]
                    eroded_val = astar_planner.eroded_map[goal_row, goal_col]
                    
                    if planning_val == 1:
                        status_message = "LOCATION OBSTRUCTED - Obstacle detected at destination"
                    elif eroded_val == 1:
                        status_message = "LOCATION OBSTRUCTED - Too close to obstacles"
                    else:
                        status_message = "LOCATION OBSTRUCTED - Unknown obstacle type"
                    
                    print(f"Goal position is not free! Planning: {planning_val}, Eroded: {eroded_val}")
                    if status_display:
                        status_display.value = status_message
                    return None

                # use two-stage planning
                path_waypoints = astar_planner.plan_safe_path(
                    start_x=start_x, 
                    start_z=start_z,
                    goal_x=goal_x, 
                    goal_z=goal_z,
                    use_eroded=True,
                    simplify="string_pull"
                )
                
                if path_waypoints:
                    status_message = f"SAFE PATH - {len(path_waypoints)} waypoints found (two-stage if needed)"
                    print(f"Safe A* path found with {len(path_waypoints)} waypoints:")
                    
                    for i, (x, z) in enumerate(path_waypoints):
                        print(f"  Waypoint {i}: ({x:.2f}, {z:.2f})")
                    
                    if status_display:
                        status_display.value = status_message
                    
                    # visualize path in viser
                    if server is not None:
                        visualize_astar_path(server, path_waypoints)
                    
                    # send waypoint sequence to TidyBot
                    if len(path_waypoints) > 1:
                        print("Sending A* waypoint sequence to TidyBot...")
                        if send_astar_waypoint_sequence(path_waypoints, transformed_current, baseline_yaw, server):
                            # waypoint sequence sent successfully
                            save_movement_result(movement_result, "calib-results/runtime/movement_results.txt")
                            print("A* waypoint sequence initiated!")
                            
                            return {
                                'status': 'astar_sequence_started',
                                'movement_result': movement_result,
                                'waypoint_count': len(path_waypoints)
                            }
                        else:
                            print("Failed to send A* waypoint sequence")
                            return None
                    else:
                        # single waypoint - send direct TidyBot command
                        print("A* path has only start point - sending direct TidyBot command")
                        
                        # TidyBot commands
                        tidybot_x = global_x
                        tidybot_y = global_z 
                        tidybot_theta = 0.0  # degrees
                        
                        print(f"  TidyBot X movement (global): {tidybot_x:.3f}m")
                        print(f"  TidyBot Y movement (global): {tidybot_y:.3f}m")
                        print(f"  TidyBot rotation (theta): {tidybot_theta:.1f} degrees")
                        print(f"  TidyBot command: [{tidybot_y:.3f}, {tidybot_x:.3f}, {tidybot_theta:.1f}]")
                        
                        # send command to send_cmd.py
                        tidybot_command = [tidybot_y, tidybot_x, tidybot_theta]
                        if send_tidybot_command(tidybot_command):
                            print("TidyBot command sent successfully!")
                            save_movement_result(movement_result, "calib-results/runtime/movement_results.txt")
                            return {
                                'status': 'direct_movement_sent',
                                'movement_result': movement_result,
                                'tidybot_command': tidybot_command
                            }
                        else:
                            print("Failed to send TidyBot command")
                            return None
                else:
                    status_message = "PATH PLANNING FAILED - No safe route found"
                    print("A* safe path planning failed - cannot reach destination")
                    if status_display:
                        status_display.value = status_message
                    
            except Exception as e:
                status_message = f"PLANNING ERROR - {str(e)}"
                print(f"A* planning error: {e}")
                if status_display:
                    status_display.value = status_message
        else:
            status_message = "A* PLANNER NOT AVAILABLE"
            print("A* planner not available")
            if status_display:
                status_display.value = status_message
        
        # if we reach here, A* planning was not successful
        return None
    else:
        print("Failed to calculate movement")
        return None


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


def save_ply_file(points, colors, output_path):
    print(f"Saving transformed point cloud to: {output_path}")
    
    # ensure colors are in 0-255 range
    if colors.max() <= 1.0:
        colors_int = (colors * 255).astype(np.uint8)
    else:
        colors_int = colors.astype(np.uint8)
    
    # create vertex array for PLY format
    vertices = np.array([(points[i,0], points[i,1], points[i,2], 
                         colors_int[i,0], colors_int[i,1], colors_int[i,2]) 
                        for i in range(len(points))],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    # create PLY element
    vertex_element = PlyElement.describe(vertices, 'vertex')
    
    # ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    PlyData([vertex_element]).write(output_path)
    print(f"Saved {len(points)} points to PLY file: {output_path}")


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


def rotation_to_euler(rotation_matrix):
    euler_rad = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    euler_deg = np.degrees(euler_rad)
    return euler_deg


def create_camera_transform_matrix(position, quaternion):
    rotation_matrix = quaternion_wxyz_to_rotation_matrix_scipy(quaternion)
    
    # create 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    
    return transform_matrix


def apply_transformation_to_4x4_matrix(matrix_4x4, transformation_matrix):
    if transformation_matrix is None:
        return matrix_4x4
    
    # apply transformation: T_new = T_transformation @ T_camera
    transformed_matrix = transformation_matrix @ matrix_4x4
    return transformed_matrix


def extract_rotation_and_translation(matrix_4x4):
    rotation_matrix = matrix_4x4[:3, :3]
    translation = matrix_4x4[:3, 3]
    return rotation_matrix, translation


def get_camera_yaw_angle(R_cw, forward_is_neg_z=False):
    # get signed yaw between camera +z and world +z
    f = R_cw[:, 2]
    if forward_is_neg_z:
        f = -f

    # project onto world XZ plane
    f_proj = np.array([f[0], 0.0, f[2]])
    n = np.linalg.norm(f_proj)
    if n < 1e-8:
        # looking straight up/down -> yaw undefined
        return 0.0, 0.0
    f_proj /= n

    # components along +X and +Z
    fx, fz = f_proj[0], f_proj[2]

    signed_deg = np.degrees(np.arctan2(fx, fz))          # -180, 180
    full_deg   = signed_deg % 360.0                      # 0, 360

    if signed_deg > 180.0:
        signed_deg -= 360.0
    return signed_deg, full_deg


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
def create_final_transformation_matrix(hand_eye_transform, y_rot_deg=5.0, z_rot_deg=3.0, x_rot_deg=-9.0, y_trans=0.28):
    # y-axis rotation
    y_angle_rad = np.radians(y_rot_deg)
    cos_y = np.cos(y_angle_rad)
    sin_y = np.sin(y_angle_rad)
    
    y_rotation_matrix = np.array([
        [cos_y,  0, sin_y, 0],
        [0,      1, 0,     0],
        [-sin_y, 0, cos_y, 0],
        [0,      0, 0,     1]
    ])
    
    # z-axis rotation
    z_angle_rad = np.radians(z_rot_deg)
    cos_z = np.cos(z_angle_rad)
    sin_z = np.sin(z_angle_rad)
    
    z_rotation_matrix = np.array([
        [cos_z, -sin_z, 0, 0],
        [sin_z,  cos_z, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ])
    
    # x-axis rotation
    x_angle_rad = np.radians(x_rot_deg)
    cos_x = np.cos(x_angle_rad)
    sin_x = np.sin(x_angle_rad)
    
    x_rotation_matrix = np.array([
        [1, 0,      0,      0],
        [0, cos_x, -sin_x,  0],
        [0, sin_x,  cos_x,  0],
        [0, 0,      0,      1]
    ])
    
    # y-axis translation
    translation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, y_trans],
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
    parser = argparse.ArgumentParser(description="Hand-eye calibration visualization tool")
    parser.add_argument("--save", action="store_true", 
                       help="Save transformation matrices to calib-results directory")
    parser.add_argument("--save-transformed-ply", action="store_true",
                       help="Save the transformed point cloud as PLY file")
    parser.add_argument("--saved-folder", required=True,
                       help="Path to saved state folder containing point_cloud.ply (e.g., saved-states/test-recalib)")
    args = parser.parse_args()

    server = viser.ViserServer()
    server.set_up_direction((0.0, 1.0, 0.0))

    # reset yaw baseline
    try:
        os.makedirs("calib-results/runtime", exist_ok=True)
        if os.path.exists(INITIAL_YAW_FILE):
            os.remove(INITIAL_YAW_FILE)
            print(f"Cleared baseline yaw file: {INITIAL_YAW_FILE}")
    except Exception as e:
        print(f"Warning: could not clear baseline yaw file: {e}")
    
    # load transformation matrix for movement calculations
    print("Loading transformation matrix for movement calculations...")
    transformation_matrix = load_transformation_matrix()
    print("Movement calculation system ready!")
    
    # generate planning maps and initialize A* planner
    print("Initializing A* path planning...")
    
    # generate planning maps
    print("Generating planning maps...")
    planning_maps_generated = generate_planning_maps()
    
    astar_planner = None
    if planning_maps_generated:
        print("Planning maps generated successfully")
        
        # load A* planner
        print("Loading A* planner...")
        astar_planner = load_astar_planner()
        
        if astar_planner:
            print("A* planner loaded successfully!")
            print("A* path planning is ready!")
        else:
            print("Failed to load A* planner")
    else:
        print("Failed to generate planning maps")
    
    if astar_planner is None:
        print("WARNING: A* planner failed to initialize")
        print("Falling back to direct movement")
    
    print("Click any point to calculate movement from current camera position to that destination.")
    if astar_planner:
        print("A* path planning will be used to find safe routes and visualize trajectories.")
    print("Live camera position will be read from calib-results/runtime/camera_position.txt (make sure main.py is running)")
    
    # add red dot for current camera position
    print("Adding red dot for current camera position...")
    camera_position_dot = add_camera_position_dot(server, transformation_matrix) 

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

    # load the ply file from specified save folder
    ply_file_path = os.path.join(args.saved_folder, "point_cloud.ply")
    if not os.path.exists(ply_file_path):
        print(f"Error: point_cloud.ply not found in {args.saved_folder}")
        print(f"Expected file: {ply_file_path}")
        exit(1)
    
    print(f"Loading point cloud from: {ply_file_path}")
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
    
    # make these variables accessible to slider callbacks
    global transformed_points_global, transformed_axes_global
    transformed_points_global = transformed_points
    transformed_axes_global = transformed_axes
    
    # always save transformed PLY for A* planning (do this first)
    transformed_ply_path = os.path.join("calib-results", "transformed_pointcloud.ply")
    print(f"Saving transformed point cloud for A* planning to: {transformed_ply_path}")
    save_ply_file(transformed_points, original_colors, transformed_ply_path)
    
    # also save if user requested it explicitly (redundant but clear)
    if args.save_transformed_ply:
        print("User also requested saving transformed PLY (already saved above)")
    
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
    # print(f"Adding original point cloud with {len(original_points_display)} points")
    # server.scene.add_point_cloud(
    #     "/original_pointcloud",
    #     points=original_points_display,
    #     colors=original_colors, 
    #     point_size=0.003,
    # )
    
    # show transformed point cloud
    print(f"Adding transformed point cloud with {len(transformed_points)} points")
    server.scene.add_point_cloud(
        "/transformed_pointcloud",
        points=transformed_points,
        colors=original_colors,
        point_size=0.003,
    )

    # add large invisible sphere to capture clicks
    print("Adding click handler...")
    
    # calculate the bounding box of the transformed point cloud
    min_bounds = transformed_points.min(axis=0)
    max_bounds = transformed_points.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    diagonal = np.linalg.norm(max_bounds - min_bounds)
    
    # create an invisible sphere that encompasses the entire scene
    click_catcher = server.scene.add_icosphere(
        "/click_catcher",
        radius=diagonal,
        position=center,
        subdivisions=2,
        color=(255, 255, 255),
        opacity=0.0,
        visible=True
    )
    
    # create obstacle grid visualization
    def create_obstacle_grid_points(planner, height_y=0.0):
        if planner is None:
            return np.array([]), np.array([])
            
        points = []
        colors = []
        
        # create points for each cell
        for row in range(planner.height):
            for col in range(planner.width):
                # get world coordinates of cell center
                x, z = planner.grid_to_world(row, col)
                
                # use processed map
                processed_val = planner.eroded_map[row, col]
                
                if processed_val == 0:  # free
                    color = [0.0, 1.0, 0.0]
                elif processed_val == 1:  # obstacle
                    color = [1.0, 0.0, 0.0]
                else:  # unknown
                    color = [0.5, 0.5, 0.5]
                
                # add point at specified height
                points.append([x, height_y, z])
                colors.append(color)
        
        return np.array(points), np.array(colors)
    
    def update_obstacle_grid():
        if show_obstacle_grid.value and astar_planner is not None:
            # create grid points
            grid_points, grid_colors = create_obstacle_grid_points(astar_planner, grid_height.value)
            
            if len(grid_points) > 0:
                # add or update obstacle grid
                server.scene.add_point_cloud(
                    "/obstacle_grid",
                    points=grid_points,
                    colors=grid_colors,
                    point_size=0.01,
                )
            else:
                # remove grid if no points
                try:
                    server.scene["/obstacle_grid"].remove()
                except:
                    pass
        else:
            # remove grid when disabled
            try:
                server.scene["/obstacle_grid"].remove()
            except:
                pass

    # create GUI elements first
    with server.gui.add_folder("Coordinate Display"):
        # add a text display for the last clicked coordinate
        coord_display = server.gui.add_text(
            "Last clicked coordinate",
            initial_value="No point clicked yet",
            disabled=False
        )
        
        # add a status display for path planning
        path_status_display = server.gui.add_text(
            "Path Planning Status",
            initial_value="Click a point to begin",
            disabled=False
        )
    
    # add obstacle grid visualization controls
    with server.gui.add_folder("Obstacle Grid Visualization"):
        show_obstacle_grid = server.gui.add_checkbox(
            "Show Obstacle Grid",
            initial_value=False
        )
        
        grid_height = server.gui.add_slider(
            "Grid Height (Y)",
            min=-2.0,
            max=2.0,
            step=0.01,
            initial_value=0.5
        )
    
    # add transformation adjustment sliders
    with server.gui.add_folder("Alignment Adjustment"):
        
        y_rotation_slider = server.gui.add_slider(
            "Y Rotation (degrees)",
            min=-180.0,
            max=180.0,
            step=0.5,
            initial_value=5.0
        )
        
        z_rotation_slider = server.gui.add_slider(
            "Z Rotation (degrees)", 
            min=-180.0,
            max=180.0,
            step=0.5,
            initial_value=3.0
        )
        
        x_rotation_slider = server.gui.add_slider(
            "X Rotation (degrees)",
            min=-180.0, 
            max=180.0,
            step=0.5,
            initial_value=-9.0
        )
        
        y_translation_slider = server.gui.add_slider(
            "Y Translation (meters)",
            min=-5.0,
            max=5.0, 
            step=0.01,
            initial_value=0.28
        )
        
        reset_button = server.gui.add_button("Reset to Defaults")
        save_button = server.gui.add_button("Save Current Values")
    
    # function to update transformation based on slider values
    def update_transformation():
        global transformed_points_global, transformed_axes_global
        
        # get current slider values
        y_rot = y_rotation_slider.value
        z_rot = z_rotation_slider.value  
        x_rot = x_rotation_slider.value
        y_trans = y_translation_slider.value
        
        # recalculate transformation matrix
        hand_eye_transform = np.linalg.inv(T_cam2base)
        transformed_axes_global = create_final_transformation_matrix(
            hand_eye_transform, y_rot, z_rot, x_rot, y_trans
        )
        
        # recalculate transformed points
        transformed_points_global = apply_transformation_to_points(original_points, transformed_axes_global)
        
        # update point cloud visualization
        try:
            server.scene["/transformed_pointcloud"].remove()
        except:
            pass
        
        server.scene.add_point_cloud(
            "/transformed_pointcloud",
            points=transformed_points_global,
            colors=original_colors,
            point_size=0.003,
        )
        
        # update transformed axes
        try:
            server.scene["/transformed_axes"].remove()
        except:
            pass
        
        transformed_rotation = R.from_matrix(transformed_axes_global[:3, :3]).as_quat()
        transformed_quaternion = np.array([transformed_rotation[3], transformed_rotation[0], transformed_rotation[1], transformed_rotation[2]])
        transformed_position = transformed_axes_global[:3, 3]
        server.scene.add_frame("/transformed_axes", wxyz=transformed_quaternion, position=transformed_position, axes_length=0.8, axes_radius=0.015)
        
        print(f"Updated transformation: Y={y_rot:.1f}°, Z={z_rot:.1f}°, X={x_rot:.1f}°, Y_trans={y_trans:.3f}m")
    
    # add callbacks to sliders
    @y_rotation_slider.on_update
    def _(_):
        update_transformation()
    
    @z_rotation_slider.on_update  
    def _(_):
        update_transformation()
        
    @x_rotation_slider.on_update
    def _(_):
        update_transformation()
        
    @y_translation_slider.on_update
    def _(_):
        update_transformation()
    
    # obstacle grid callbacks
    @show_obstacle_grid.on_update
    def _(_):
        update_obstacle_grid()
        
    @grid_height.on_update
    def _(_):
        update_obstacle_grid()
    
    # reset button callback
    @reset_button.on_click
    def _(_):
        y_rotation_slider.value = 5.0
        z_rotation_slider.value = 3.0
        x_rotation_slider.value = -9.0
        y_translation_slider.value = 0.28
        update_transformation()
    
    # save button callback
    @save_button.on_click
    def _(_):
        y_rot = y_rotation_slider.value
        z_rot = z_rotation_slider.value
        x_rot = x_rotation_slider.value
        y_trans = y_translation_slider.value
        
        print("="*50)
        print("CURRENT TRANSFORMATION VALUES:")
        print(f"Y Rotation: {y_rot:.1f} degrees")
        print(f"Z Rotation: {z_rot:.1f} degrees") 
        print(f"X Rotation: {x_rot:.1f} degrees")
        print(f"Y Translation: {y_trans:.3f} meters")
        print("="*50)
        print("To use these values permanently, update the function:")
        print(f"create_final_transformation_matrix(hand_eye_transform, {y_rot}, {z_rot}, {x_rot}, {y_trans})")
        print("="*50)
    
    # keep track of active marker handles
    active_marker_handles = {}
    
    # initial obstacle grid update
    update_obstacle_grid()
    
    # state management for sequential movements
    robot_moving = False
    last_command_time = 0
    
    def check_robot_status():
        global robot_moving
        try:
            if os.path.exists("calib-results/runtime/robot_results.txt"):
                with open("calib-results/runtime/robot_results.txt", 'r') as f:
                    content = f.read().strip()
                    if content:
                        result = json.loads(content)
                        if result.get('success', False):
                            os.makedirs(os.path.dirname("calib-results/runtime/robot_results.txt"), exist_ok=True)
                            with open("calib-results/runtime/robot_results.txt", 'w') as f:
                                f.write("")
                            robot_moving = False
                            print("Robot movement completed!")
                            return True
        except Exception as e:
            pass
        return False
    
    # add click handler to the invisible sphere
    @click_catcher.on_click
    def handle_click(event):
        global robot_moving, last_command_time
        
        # check if robot is currently moving
        if robot_moving:
            # check if robot has finished moving
            if check_robot_status():
                print("Ready for next click!")
            else:
                print("Robot is still moving, please wait...")
                return
        
        # get the click ray in world coordinates
        click_origin = np.array(event.ray_origin)
        click_direction = np.array(event.ray_direction)
        
        # find intersection with transformed point cloud
        # project each point onto the ray and find the closest one
        if len(transformed_points) > 0:
            # vector from ray origin to each point
            point_vectors = transformed_points - click_origin
            
            # project each point onto the ray
            projections = np.dot(point_vectors, click_direction)
            
            # only consider points in front of the camera
            valid_mask = projections > 0
            
            if np.any(valid_mask):
                # calculate the closest point on the ray for each point
                ray_points = click_origin + projections[:, np.newaxis] * click_direction
                
                # calculate distances from each point to its closest point on the ray
                distances = np.linalg.norm(transformed_points - ray_points, axis=1)
                
                # apply the valid mask
                distances[~valid_mask] = np.inf
                
                # find the closest point
                closest_idx = np.argmin(distances)
                closest_distance = distances[closest_idx]
                
                # only register as a click if the point is close enough to the ray
                if closest_distance < 0.05: # 5cm threshold
                    closest_point = transformed_points[closest_idx]
                    coord_str = f"[{closest_point[0]:.3f}, {closest_point[1]:.3f}, {closest_point[2]:.3f}]"
                    print(f"Clicked point coordinates: {coord_str}")
                    
                    # update GUI display
                    coord_display.value = coord_str
                    
                    # Process clicked point for movement calculation
                    process_clicked_point(closest_point, transformation_matrix, astar_planner, server, path_status_display)
                    
                    # add a temporary marker at the clicked point
                    marker_name = f"/clicked_point_{time.time()}"
                    marker_handle = server.scene.add_point_cloud(
                        marker_name,
                        points=np.array([closest_point]),
                        colors=np.array([[1.0, 1.0, 0.0]]), # yellow marker
                        point_size=0.02,
                    )
                    
                    # store the marker handle
                    active_marker_handles[marker_name] = marker_handle
                    
                    # remove the marker after 3 seconds
                    def remove_marker(marker_name_to_remove, handle_to_remove):
                        time.sleep(15)
                        try:
                            # use the handle's remove method
                            handle_to_remove.remove()
                            # remove from tracking dictionary
                            if marker_name_to_remove in active_marker_handles:
                                del active_marker_handles[marker_name_to_remove]
                            print(f"Removed marker: {marker_name_to_remove}")
                        except Exception as e:
                            print(f"Error removing marker: {e}")
                    
                    # start a new thread to remove this marker
                    removal_thread = threading.Thread(
                        target=remove_marker, 
                        args=(marker_name, marker_handle),
                        daemon=True
                    )
                    removal_thread.start()
                else:
                    print(f"Click too far from any point (closest distance: {closest_distance:.3f}m)")
            else:
                print("Click is behind the camera view")
    
    print("Click handler ready! Click on points in the transformed point cloud to see their coordinates.")

    # add coordinate frames to help with orientation
    print("Adding coordinate frames...")
    # origin frame - no rotation
    origin_rotation = np.array([1, 0, 0, 0]) # identity quaternion (no rotation)
    server.scene.add_frame("/origin", wxyz=origin_rotation, position=np.array([0, 0, 0]), axes_length=0.8, axes_radius=0.015)
    
    # transformed coordinate frame (hand-eye calibration applied)
    # convert rotation matrix to quaternion for viser
    # transformed_rotation = R.from_matrix(transformed_axes[:3, :3]).as_quat()
    # transformed_quaternion = np.array([transformed_rotation[3], transformed_rotation[0], transformed_rotation[1], transformed_rotation[2]])  # wxyz format
    
    # apply the translation part of the transformation
    # transformed_position = transformed_axes[:3, 3] # extract translation from 4x4 matrix
    # server.scene.add_frame("/transformed_axes", wxyz=transformed_quaternion, position=transformed_position, axes_length=0.8, axes_radius=0.015)

    print("Visualization started!")
    print("Red dot shows current camera position (updates every 2 seconds)")

    # continuous camera position tracking
    last_camera_position = None
    camera_dot_handle = None
    camera_orientation_handle = None
    tilted_axis_handle = None
    
    def update_camera_position():
        global last_camera_position, camera_dot_handle, camera_orientation_handle, tilted_axis_handle
        
        try:
            # read current camera position
            position, quaternion, _ = read_camera_position()
            
            if position is not None and quaternion is not None:
                # apply transformation to get world coordinates
                transformed_position = apply_transformation(position, transformation_matrix)
                
                if (last_camera_position is None or 
                    np.linalg.norm(transformed_position - last_camera_position) > 0.01):
                    
                    if camera_dot_handle is not None:
                        try:
                            camera_dot_handle.remove()
                        except:
                            pass
                    
                    if camera_orientation_handle is not None:
                        try:
                            camera_orientation_handle.remove()
                        except:
                            pass
                    
                    if tilted_axis_handle is not None:
                        try:
                            tilted_axis_handle.remove()
                        except:
                            pass
                    
                    # add new red dot at updated position
                    camera_dot_handle = server.scene.add_point_cloud(
                        "/camera_position",
                        points=np.array([transformed_position]),
                        colors=np.array([[1.0, 0.0, 0.0]]),
                        point_size=0.05,
                    )
                    
                    # 4x4 camera transformation matrix
                    camera_transform = create_camera_transform_matrix(position, quaternion)
                    
                    # apply hand-eye calibration transformation to get world coordinates
                    transformed_camera_matrix = apply_transformation_to_4x4_matrix(camera_transform, transformation_matrix)
                    
                    # extract rotation matrix in world coordinates
                    transformed_rotation, _ = extract_rotation_and_translation(transformed_camera_matrix)
                    
                    # show camera orientation in world coordinates
                    world_quaternion = R.from_matrix(transformed_rotation).as_quat()
                    # [x,y,z,w] to [w,x,y,z] format
                    world_quaternion_wxyz = np.array([world_quaternion[3], world_quaternion[0], world_quaternion[1], world_quaternion[2]])
                    
                    camera_orientation_handle = server.scene.add_frame(
                        "/camera_orientation",
                        wxyz=world_quaternion_wxyz,
                        position=transformed_position,
                        axes_length=0.2,
                        axes_radius=0.01,
                    )
                    
                    # calculate and update tilt angle using signed yaw
                    signed_yaw, _ = get_camera_yaw_angle(transformed_rotation, forward_is_neg_z=False)
                    yaw_rad = np.radians(signed_yaw)

                    ensure_initial_yaw_saved(signed_yaw)

                    cos_tilt = np.cos(yaw_rad)
                    sin_tilt = np.sin(yaw_rad)
                    
                    # rotation matrix for tilt around y-axis
                    tilt_rotation_matrix = np.array([
                        [cos_tilt,  0, sin_tilt],
                        [0,         1, 0],
                        [-sin_tilt, 0, cos_tilt]
                    ])
                    
                    # convert to quaternion
                    tilt_quaternion = R.from_matrix(tilt_rotation_matrix).as_quat()
                    tilt_quaternion_wxyz = np.array([tilt_quaternion[3], tilt_quaternion[0], tilt_quaternion[1], tilt_quaternion[2]])
                    
                    tilted_axis_handle = server.scene.add_frame(
                        "/yaw_axis",
                        wxyz=tilt_quaternion_wxyz,
                        position=np.array([0, 0, 0]),  
                        axes_length=0.3,
                        axes_radius=0.015,
                    )
                    
                    last_camera_position = transformed_position
                    print(f"Camera position, orientation, and tilt angle ({signed_yaw:.1f}°) updated: {transformed_position}")
        except Exception as e:
            if hasattr(update_camera_position, 'last_error_time'):
                if time.time() - update_camera_position.last_error_time > 5.0: 
                    print(f"Error in update_camera_position (continuing...): {e}")
                    update_camera_position.last_error_time = time.time()
            else:
                update_camera_position.last_error_time = time.time()
                print(f"Error in update_camera_position (continuing...): {e}")

    while True:
        # update camera position every 2 seconds
        update_camera_position()
        
        # check robot status if it's moving
        if robot_moving:
            # check for timeout
            if time.time() - last_command_time > 30:
                print("Robot movement timeout - assuming completed")
                robot_moving = False
            else:
                check_robot_status()
        
        time.sleep(1.0)
