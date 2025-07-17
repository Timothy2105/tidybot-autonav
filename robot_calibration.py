import argparse
import time
import numpy as np
import sys
import os
from pathlib import Path
import datetime

# add mast3r_slam path
sys.path.insert(0, str(Path(__file__).parent))

from mast3r_slam.robot_interface import RobotInterface


def create_calib_results_folder():
    """Create the calib-results folder and return the path."""
    calib_dir = Path("calib-results")
    calib_dir.mkdir(exist_ok=True)
    return calib_dir


def log_test_result(log_file, test_name, target_movement, start_pose, end_pose, camera_positions, success):
    """
    Log test results to the calibration file.
    
    Args:
        log_file: File object to write to
        test_name: Name of the test
        target_movement: Target movement [y, x, theta]
        start_pose: Starting robot pose [x, y, theta]
        end_pose: Ending robot pose [x, y, theta]
        camera_positions: List of camera positions during test
        success: Whether the test was successful
    """
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"TEST: {test_name}\n")
    log_file.write(f"TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'='*60}\n")
    
    # Target movement
    log_file.write(f"TARGET MOVEMENT: [y={target_movement[0]:.3f}, x={target_movement[1]:.3f}, theta={np.degrees(target_movement[2]):.1f}°]\n")
    
    # Robot poses
    log_file.write(f"ROBOT START POSE: [x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, theta={np.degrees(start_pose[2]):.1f}°]\n")
    log_file.write(f"ROBOT END POSE: [x={end_pose[0]:.3f}, y={end_pose[1]:.3f}, theta={np.degrees(end_pose[2]):.1f}°]\n")
    
    # Actual movement (relative to start)
    actual_movement = end_pose - start_pose
    actual_distance = np.linalg.norm(actual_movement[:2])
    actual_rotation = actual_movement[2]
    log_file.write(f"ACTUAL MOVEMENT: [dx={actual_movement[0]:.3f}, dy={actual_movement[1]:.3f}, dtheta={np.degrees(actual_rotation):.1f}°]\n")
    log_file.write(f"ACTUAL DISTANCE: {actual_distance:.3f}m\n")
    log_file.write(f"ACTUAL ROTATION: {np.degrees(actual_rotation):.1f}°\n")
    
    # Camera positions during test
    log_file.write(f"CAMERA POSITIONS DURING TEST ({len(camera_positions)} samples):\n")
    for i, pos in enumerate(camera_positions):
        log_file.write(f"  Sample {i+1}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}\n")
    
    # Test result
    log_file.write(f"TEST RESULT: {'PASSED' if success else 'FAILED'}\n")
    log_file.write(f"{'='*60}\n")
    log_file.flush()  # Ensure it's written immediately


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
        
        # Test 1: Move forward 1 meter
        print("\n📋 Test 1: Moving FORWARD by 1.0 meter")
        
        # Get starting pose
        start_obs = robot_interface.get_obs()
        start_pose = np.array(start_obs["base_pose"])
        camera_positions = []  # Will be populated if we have SLAM interface
        
        print("Starting movement with wheel odometry monitoring...")
        success = robot_interface.move_forward(1.0)  # Use new move_forward method
        
        # Get ending pose
        end_obs = robot_interface.get_obs()
        end_pose = np.array(end_obs["base_pose"])
        
        if success:
            print("✅ Test 1 PASSED: Robot moved forward successfully")
            print(f"Final position: [x={end_pose[0]:.3f}, y={end_pose[1]:.3f}, theta={np.degrees(end_pose[2]):.1f}°]")
        else:
            print("❌ Test 1 FAILED: Robot failed to move forward")
        
        # Log test results
        log_test_result(log_file, "Test 1: Move Forward 1.0m", [0.0, 1.0, 0.0], start_pose, end_pose, camera_positions, success)
        
        time.sleep(2)  # Wait 2 seconds between tests
        
        # Test 2: Move backward 1 meter
        print("\n📋 Test 2: Moving BACKWARD by 1.0 meter")
        
        start_obs = robot_interface.get_obs()
        start_pose = np.array(start_obs["base_pose"])
        camera_positions = []
        
        print("Starting movement with wheel odometry monitoring...")
        success = robot_interface.move_forward(-1.0)  # Use negative distance for backward
        
        end_obs = robot_interface.get_obs()
        end_pose = np.array(end_obs["base_pose"])
        
        if success:
            print("✅ Test 2 PASSED: Robot moved backward successfully")
            print(f"Final position: x={end_pose[0]:.3f}, y={end_pose[1]:.3f}, theta={np.degrees(end_pose[2]):.1f}°]")
        else:
            print("❌ Test 2 FAILED: Robot failed to move backward")
        
        log_test_result(log_file, "Test 2: Move Backward 1.0m", [0.0, -1.0, 0.0], start_pose, end_pose, camera_positions, success)
        
        time.sleep(2)  # Wait 2 seconds between tests
        
        # Test 3: Rotate left 90 degrees
        print("\n📋 Test 3: Rotating LEFT by 90 degrees")
        
        start_obs = robot_interface.get_obs()
        start_pose = np.array(start_obs["base_pose"])
        camera_positions = []
        
        print("Starting rotation with wheel odometry monitoring...")
        success = robot_interface.rotate_in_place(np.radians(90))  # Use new rotate_in_place method
        
        end_obs = robot_interface.get_obs()
        end_pose = np.array(end_obs["base_pose"])
        
        if success:
            print("✅ Test 3 PASSED: Robot rotated left successfully")
            print(f"Final position: x={end_pose[0]:.3f}, y={end_pose[1]:.3f}, theta={np.degrees(end_pose[2]):.1f}°]")
        else:
            print("❌ Test 3 FAILED: Robot failed to rotate left")
        
        log_test_result(log_file, "Test 3: Rotate Left 90°", [0.0, 0.0, np.radians(90)], start_pose, end_pose, camera_positions, success)
        
        time.sleep(2)  # Wait 2 seconds between tests
        
        # Test 4: Rotate right 90 degrees
        print("\n📋 Test 4: Rotating RIGHT by 90 degrees")
        
        start_obs = robot_interface.get_obs()
        start_pose = np.array(start_obs["base_pose"])
        camera_positions = []
        
        print("Starting rotation with wheel odometry monitoring...")
        success = robot_interface.rotate_in_place(np.radians(-90))  # Use negative angle for right rotation
        
        end_obs = robot_interface.get_obs()
        end_pose = np.array(end_obs["base_pose"])
        
        if success:
            print("✅ Test 4 PASSED: Robot rotated right successfully")
            print(f"Final position: x={end_pose[0]:.3f}, y={end_pose[1]:.3f}, theta={np.degrees(end_pose[2]):.1f}°]")
        else:
            print("❌ Test 4 FAILED: Robot failed to rotate right")
        
        log_test_result(log_file, "Test 4: Rotate Right 90°", [0.0, 0.0, np.radians(-90)], start_pose, end_pose, camera_positions, success)
        
        time.sleep(2)  # Wait 2 seconds between tests
        
        # Test 5: Strafe left 1 meter
        print("\n📋 Test 5: Strafing LEFT by 1.0 meter")
        
        start_obs = robot_interface.get_obs()
        start_pose = np.array(start_obs["base_pose"])
        camera_positions = []
        
        print("Starting strafe with wheel odometry monitoring...")
        # For strafing, we need to use the original method but with proper target calculation
        # Strafe left means move perpendicular to current heading
        start_theta = start_pose[2]
        strafe_x = 1.0 * np.cos(start_theta + np.pi/2)  # Perpendicular to heading
        strafe_y = 1.0 * np.sin(start_theta + np.pi/2)
        target_pose = np.array([start_pose[0] + strafe_x, start_pose[1] + strafe_y, start_pose[2]])
        
        success = robot_interface.move_to_base_waypoint(
            [1.0, 0.0, 0.0],  # [y, x, theta] - strafe left
            threshold_pos=0.01,
            threshold_theta=0.01,
            max_steps=100
        )
        
        end_obs = robot_interface.get_obs()
        end_pose = np.array(end_obs["base_pose"])
        
        if success:
            print("✅ Test 5 PASSED: Robot strafed left successfully")
            print(f"Final position: x={end_pose[0]:.3f}, y={end_pose[1]:.3f}, theta={np.degrees(end_pose[2]):.1f}°]")
        else:
            print("❌ Test 5 FAILED: Robot failed to strafe left")
        
        log_test_result(log_file, "Test 5: Strafe Left 1.0m", [1.0, 0.0, 0.0], start_pose, end_pose, camera_positions, success)
        
        time.sleep(2)  # Wait 2 seconds between tests
        
        # Test 6: Strafe right 1 meter
        print("\n📋 Test 6: Strafing RIGHT by 1.0 meter")
        
        start_obs = robot_interface.get_obs()
        start_pose = np.array(start_obs["base_pose"])
        camera_positions = []
        
        print("Starting strafe with wheel odometry monitoring...")
        success = robot_interface.move_to_base_waypoint(
            [-1.0, 0.0, 0.0],  # [y, x, theta] - strafe right
            threshold_pos=0.01,
            threshold_theta=0.01,
            max_steps=100
        )
        
        end_obs = robot_interface.get_obs()
        end_pose = np.array(end_obs["base_pose"])
        
        if success:
            print("✅ Test 6 PASSED: Robot strafed right successfully")
            print(f"Final position: x={end_pose[0]:.3f}, y={end_pose[1]:.3f}, theta={np.degrees(end_pose[2]):.1f}°]")
        else:
            print("❌ Test 6 FAILED: Robot failed to strafe right")
        
        log_test_result(log_file, "Test 6: Strafe Right 1.0m", [-1.0, 0.0, 0.0], start_pose, end_pose, camera_positions, success)
        
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