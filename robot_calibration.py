import argparse
import time
import numpy as np
import sys
import os
from pathlib import Path
import datetime
import json

# add mast3r_slam path
sys.path.insert(0, str(Path(__file__).parent))

from mast3r_slam.robot_interface import RobotInterface


def get_camera_position_from_slam():
    """
    Try to read the current camera position and orientation from SLAM system.
    This reads from a shared file that the main SLAM process writes to.
    
    Returns:
        tuple: (x, y, z, quaternion) camera position and orientation or None if not available
    """
    try:
        # Look for a file that contains the current camera position and orientation
        camera_pos_file = Path("calib-results/runtime/camera_position.txt")
        if camera_pos_file.exists():
            with open(camera_pos_file, 'r') as f:
                data = json.load(f)
                position = (data['x'], data['y'], data['z'])
                quaternion = data.get('quaternion', [1, 0, 0, 0])  # Default to identity quaternion
                return (position[0], position[1], position[2], quaternion)
    except Exception as e:
        print(f"Warning: Could not read camera position: {e}")
    
    return None


def log_test_result(log_file, test_name, target_movement, start_pose, end_pose, start_camera_data, end_camera_data, success):
    """
    Log test results to the calibration file.
    
    Args:
        log_file: File object to write to
        test_name: Name of the test
        target_movement: Target movement [y, x, theta]
        start_pose: Starting robot pose [x, y, theta]
        end_pose: Ending robot pose [x, y, theta]
        start_camera_data: Starting camera data (x, y, z, quaternion) from SLAM
        end_camera_data: Ending camera data (x, y, z, quaternion) from SLAM
        success: Whether the test was successful
    """
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"TEST: {test_name}\n")
    log_file.write(f"TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'='*60}\n")
    
    # Target movement
    log_file.write(f"TARGET MOVEMENT: [y={target_movement[0]:.3f}, x={target_movement[1]:.3f}, theta={np.degrees(target_movement[2]):.1f}°]\n")
    
    # Calculated wheel (robot odometry) start and end pose
    log_file.write(f"WHEEL ODOMETRY START POSE: [x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, theta={np.degrees(start_pose[2]):.1f}°]\n")
    log_file.write(f"WHEEL ODOMETRY END POSE:   [x={end_pose[0]:.3f}, y={end_pose[1]:.3f}, theta={np.degrees(end_pose[2]):.1f}°]\n")
    
    # Calculated wheel diff
    actual_movement = end_pose - start_pose
    actual_distance = np.linalg.norm(actual_movement[:2])
    actual_rotation = actual_movement[2]
    log_file.write(f"WHEEL ODOMETRY DIFF: [dx={actual_movement[0]:.3f}, dy={actual_movement[1]:.3f}, dtheta={np.degrees(actual_rotation):.1f}°]\n")
    log_file.write(f"WHEEL ODOMETRY DISTANCE: {actual_distance:.3f}m\n")
    log_file.write(f"WHEEL ODOMETRY ROTATION: {np.degrees(actual_rotation):.1f}°\n")
    
    log_file.write("\n")
    
    # Camera estimated start and end position + orientation (SLAM)
    if start_camera_data is not None:
        start_pos = start_camera_data[:3]
        start_quat = start_camera_data[3] if len(start_camera_data) > 3 else [1, 0, 0, 0]
        log_file.write(f"CAMERA ESTIMATED START POSITION (SLAM): [x={start_pos[0]:.3f}, y={start_pos[1]:.3f}, z={start_pos[2]:.3f}]\n")
        log_file.write(f"CAMERA ESTIMATED START ORIENTATION (SLAM) (quaternion): [w={start_quat[0]:.6f}, x={start_quat[1]:.6f}, y={start_quat[2]:.6f}, z={start_quat[3]:.6f}]\n")
    else:
        log_file.write(f"CAMERA ESTIMATED START POSITION (SLAM): Not available\n")
        log_file.write(f"CAMERA ESTIMATED START ORIENTATION (SLAM): Not available\n")
    
    if end_camera_data is not None:
        end_pos = end_camera_data[:3]
        end_quat = end_camera_data[3] if len(end_camera_data) > 3 else [1, 0, 0, 0]
        log_file.write(f"CAMERA ESTIMATED END POSITION (SLAM):   [x={end_pos[0]:.3f}, y={end_pos[1]:.3f}, z={end_pos[2]:.3f}]\n")
        log_file.write(f"CAMERA ESTIMATED END ORIENTATION (SLAM) (quaternion):   [w={end_quat[0]:.6f}, x={end_quat[1]:.6f}, y={end_quat[2]:.6f}, z={end_quat[3]:.6f}]\n")
    else:
        log_file.write(f"CAMERA ESTIMATED END POSITION (SLAM): Not available\n")
        log_file.write(f"CAMERA ESTIMATED END ORIENTATION (SLAM): Not available\n")
    
    # Calculated camera diffs
    if start_camera_data is not None and end_camera_data is not None:
        start_pos = np.array(start_camera_data[:3])
        end_pos = np.array(end_camera_data[:3])
        camera_movement = end_pos - start_pos
        camera_distance = np.linalg.norm(camera_movement[:2])
        start_quat = start_camera_data[3] if len(start_camera_data) > 3 else [1, 0, 0, 0]
        end_quat = end_camera_data[3] if len(end_camera_data) > 3 else [1, 0, 0, 0]
        log_file.write(f"CAMERA ESTIMATED DIFF: [dx={camera_movement[0]:.3f}, dy={camera_movement[1]:.3f}, dz={camera_movement[2]:.3f}]\n")
        log_file.write(f"CAMERA ESTIMATED DISTANCE: {camera_distance:.3f}m\n")
        quat_diff = np.array(end_quat) - np.array(start_quat)
        quat_diff_norm = np.linalg.norm(quat_diff)
        log_file.write(f"CAMERA ESTIMATED QUATERNION DIFF NORM: {quat_diff_norm:.6f}\n")
    else:
        log_file.write(f"CAMERA ESTIMATED DIFF: Not available\n")
        log_file.write(f"CAMERA ESTIMATED ROTATION DIFF: Not available\n")
    
    log_file.write("\n")
    
    # Test result
    log_file.write(f"TEST RESULT: {'PASSED' if success else 'FAILED'}\n")
    log_file.write(f"{'='*60}\n")
    log_file.flush()  # Ensure it's written immediately


def create_calib_results_folder():
    """Create the calib-results folder and return the path."""
    calib_dir = Path("calib-results")
    calib_dir.mkdir(exist_ok=True)
    return calib_dir


def run_calibration_tests(robot_interface, slam_interface=None):
    """
    Run the full calibration test suite.
    
    Args:
        robot_interface: RobotInterface instance
        slam_interface: Optional SLAM interface for communication
    """
    print("\n🤖 ROBOT CALIBRATION MODE")
    print("Testing basic movement commands...")
    
    # Create results folder and log file
    calib_dir = create_calib_results_folder()
    log_file_path = calib_dir / "calib.txt"
    
    with open(log_file_path, 'w') as log_file:
        log_file.write("ROBOT CALIBRATION RESULTS\n")
        log_file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Robot Interface: {'Simulation' if robot_interface.simulate else 'Real Robot'}\n")
        log_file.write(f"{'='*60}\n")
        
        # Get initial robot position
        initial_obs = robot_interface.get_obs()
        initial_pose = np.array(initial_obs["base_pose"])
        log_file.write(f"INITIAL ROBOT POSE: [x={initial_pose[0]:.3f}, y={initial_pose[1]:.3f}, theta={np.degrees(initial_pose[2]):.1f}°]\n")
        log_file.write(f"{'='*60}\n")
        
        # Helper to extract base pose from base.get_state()
        def get_base_pose(base):
            obs = base.get_state()
            if isinstance(obs, dict) and 'base_pose' in obs:
                return np.array(obs['base_pose'])
            raise RuntimeError(f"base.get_state() did not return a dict with 'base_pose': got {obs}")

        # Helper function for non-blocking movement/rotation
        def move_base_to(target_pose, base, description):
            print(f"\n➡️  Command: {description}")
            while True:
                base.execute_action({'base_pose': target_pose})  # <-- send every loop!
                state = get_base_pose(base)
                print(f"Current pose: x={state[0]:.3f}, y={state[1]:.3f}, theta={np.degrees(state[2]):.1f}°")
                pos_error = np.linalg.norm(state[:2] - target_pose[:2])
                theta_error = np.abs(np.arctan2(np.sin(target_pose[2] - state[2]), np.cos(target_pose[2] - state[2])))
                if pos_error < 0.01 and theta_error < 0.005:
                    print("Target reached!")
                    break
                time.sleep(0.1)

        base = robot_interface.base if hasattr(robot_interface, 'base') else robot_interface

        # 16-SIDED POLYGON TEST SEQUENCE
        num_sides = 16
        side_length = 4.71 / num_sides  # ≈ 0.294375m
        turn_angle = 2 * np.pi / num_sides  # 22.5° in radians
        for i in range(num_sides):
            print(f"\n[TEST {i+1}/{num_sides}] 16-gon Step {i+1}: Move forward and rotate {np.degrees(turn_angle):.1f}°")
            start_pose = get_base_pose(base)
            start_camera_pos = get_camera_position_from_slam()
            # Calculate target pose: move forward along current heading, rotate at the same time
            target_x = start_pose[0] + side_length * np.cos(start_pose[2])
            target_y = start_pose[1] + side_length * np.sin(start_pose[2])
            target_theta = start_pose[2] + turn_angle
            target_pose = np.array([target_x, target_y, target_theta])
            # Send as a single move+rotate command
            move_base_to(target_pose, base, f"16-gon Step {i+1}: Move {side_length:.3f}m and rotate {np.degrees(turn_angle):.1f}°")
            end_pose = get_base_pose(base)
            end_camera_pos = get_camera_position_from_slam()
            log_test_result(
                log_file,
                f"16-gon Step {i+1}: Move {side_length:.3f}m + Rotate {np.degrees(turn_angle):.1f}°",
                [side_length, 0.0, turn_angle],
                start_pose,
                end_pose,
                start_camera_pos,
                end_camera_pos,
                True
            )
            time.sleep(2)

        # Final summary
        final_obs = robot_interface.get_obs()
        final_pose = np.array(final_obs["base_pose"])
        total_movement = final_pose - initial_pose
        total_distance = np.linalg.norm(total_movement[:2])
        
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"CALIBRATION SUMMARY\n")
        log_file.write(f"{'='*60}\n")
        log_file.write(f"FINAL ROBOT POSE: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]\n")
        log_file.write(f"TOTAL MOVEMENT FROM START: [dx={total_movement[0]:.3f}, dy={total_movement[1]:.3f}, dtheta={np.degrees(total_movement[2]):.1f}°]\n")
        log_file.write(f"TOTAL DISTANCE TRAVELED: {total_distance:.3f}m\n")
        log_file.write(f"TOTAL ROTATION: {np.degrees(total_movement[2]):.1f}°\n")
        log_file.write(f"{'='*60}\n")
    
    print(f"\n🎉 ROBOT CALIBRATION COMPLETE!")
    print(f"Results saved to: {log_file_path}")
    print("All basic movement tests finished.")
    print("SLAM system continues running - you can close the visualization window when done.")


def main():
    parser = argparse.ArgumentParser(description="Standalone Robot Calibration")
    parser.add_argument("--simulate", action="store_true", 
                       help="Simulate robot commands instead of sending to real robot")
    parser.add_argument("--robot-host", default="192.168.1.4",
                       help="Robot RPC server host (default: 192.168.1.4)")
    parser.add_argument("--robot-port", type=int, default=50000,
                       help="Robot RPC server port (default: 50000)")
    
    args = parser.parse_args()
    
    print("🤖 Starting Standalone Robot Calibration")
    print(f"Robot server: {args.robot_host}:{args.robot_port}")
    print(f"Simulation mode: {args.simulate}")
    
    if not args.simulate:
        print("\n" + "="*60)
        print("IMPORTANT: Before running this script, you need to start the TidyBot server!")
        print("1. Open a new terminal")
        print("2. Navigate to the TidyBot directory:")
        print("   cd MASt3R-SLAM/thirdparty/tidybot2")
        print("3. Start the base server:")
        print("   python base_server.py")
        print("4. Wait for the server to start, then run this calibration script")
        print("="*60)
        
        response = input("Have you started the TidyBot server? (y/n): ")
        if response.lower() != 'y':
            print("Please start the TidyBot server first, then run this script again.")
            return 1
    
    # Initialize robot interface
    try:
        robot_interface = RobotInterface(simulate=args.simulate)
        print("✅ Robot interface initialized successfully")
        
        if not args.simulate:
            # Test connection
            print("Testing robot connection...")
            obs = robot_interface.get_obs()
            print(f"Current robot position: {obs['base_pose']}")
            print("✅ Robot connection successful!")
        
    except Exception as e:
        print(f"❌ Failed to initialize robot interface: {e}")
        if not args.simulate:
            print("\nTroubleshooting tips:")
            print("1. Make sure the TidyBot server is running (python base_server.py)")
            print("2. Check that the robot host/port are correct")
            print("3. Try running with --simulate first to test the script")
        return 1
    
    # Wait for user to confirm SLAM is running
    print("\n" + "="*60)
    print("IMPORTANT: Make sure the SLAM system is running in another terminal!")
    print("The SLAM system should be running with --load-state and --calib-robot")
    print("="*60)
    input("Press Enter when SLAM system is ready...")
    
    # Run calibration tests
    try:
        run_calibration_tests(robot_interface)
    except KeyboardInterrupt:
        print("\n⚠️ Calibration interrupted by user")
    except Exception as e:
        print(f"\n❌ Calibration failed with error: {e}")
        return 1
    finally:
        # Clean up
        robot_interface.close()
        print("Robot interface closed.")
    
    print("\n✅ Calibration script finished successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 