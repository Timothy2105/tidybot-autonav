#!/usr/bin/env python3
"""
Standalone Robot Calibration Script

This script runs robot calibration tests while communicating with a running SLAM system.
Run this in a separate terminal while the SLAM system is running.

Usage:
    python robot_calibration.py --slam-port 50001 --robot-port 50000
"""

import argparse
import time
import numpy as np
import sys
import os
from pathlib import Path

# Add the mast3r_slam path
sys.path.insert(0, str(Path(__file__).parent))

from mast3r_slam.robot_interface import RobotInterface


def run_calibration_tests(robot_interface, slam_interface=None):
    """
    Run the full calibration test suite.
    
    Args:
        robot_interface: RobotInterface instance
        slam_interface: Optional SLAM interface for communication
    """
    print("\n🤖 ROBOT CALIBRATION MODE")
    print("Testing basic movement commands...")
    
    # Test 1: Move forward 1 meter
    print("\n📋 Test 1: Moving FORWARD by 1.0 meter")
    target_pose = np.array([0.0, 1.0, 0.0])  # [y, x, theta] - x=1.0 means forward
    print("Starting movement with wheel odometry monitoring...")
    success = robot_interface.move_to_base_waypoint(
        target_pose, 
        threshold_pos=0.01,
        threshold_theta=0.01,
        max_steps=100
    )
    if success:
        print("✅ Test 1 PASSED: Robot moved forward successfully")
        final_obs = robot_interface.get_obs()
        final_pose = final_obs["base_pose"]
        print(f"Final position: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]")
    else:
        print("❌ Test 1 FAILED: Robot failed to move forward")
    
    time.sleep(2)  # Wait 2 seconds between tests
    
    # Test 2: Move backward 1 meter
    print("\n📋 Test 2: Moving BACKWARD by 1.0 meter")
    target_pose = np.array([0.0, -1.0, 0.0])  # [y, x, theta] - x=-1.0 means backward
    print("Starting movement with wheel odometry monitoring...")
    success = robot_interface.move_to_base_waypoint(
        target_pose, 
        threshold_pos=0.01,
        threshold_theta=0.01,
        max_steps=100
    )
    if success:
        print("✅ Test 2 PASSED: Robot moved backward successfully")
        final_obs = robot_interface.get_obs()
        final_pose = final_obs["base_pose"]
        print(f"Final position: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]")
    else:
        print("❌ Test 2 FAILED: Robot failed to move backward")
    
    time.sleep(2)  # Wait 2 seconds between tests
    
    # Test 3: Rotate left 90 degrees
    print("\n📋 Test 3: Rotating LEFT by 90 degrees")
    target_pose = np.array([0.0, 0.0, np.radians(90)])  # [y, x, theta] - 90 degrees left
    print("Starting rotation with wheel odometry monitoring...")
    success = robot_interface.move_to_base_waypoint(
        target_pose, 
        threshold_pos=0.01,
        threshold_theta=0.01,
        max_steps=100
    )
    if success:
        print("✅ Test 3 PASSED: Robot rotated left successfully")
        final_obs = robot_interface.get_obs()
        final_pose = final_obs["base_pose"]
        print(f"Final position: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]")
    else:
        print("❌ Test 3 FAILED: Robot failed to rotate left")
    
    time.sleep(2)  # Wait 2 seconds between tests
    
    # Test 4: Rotate right 90 degrees
    print("\n📋 Test 4: Rotating RIGHT by 90 degrees")
    target_pose = np.array([0.0, 0.0, np.radians(-90)])  # [y, x, rotation] - 90 degrees right
    print("Starting rotation with wheel odometry monitoring...")
    success = robot_interface.move_to_base_waypoint(
        target_pose, 
        threshold_pos=0.01,
        threshold_theta=0.01,
        max_steps=100
    )
    if success:
        print("✅ Test 4 PASSED: Robot rotated right successfully")
        final_obs = robot_interface.get_obs()
        final_pose = final_obs["base_pose"]
        print(f"Final position: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]")
    else:
        print("❌ Test 4 FAILED: Robot failed to rotate right")
    
    time.sleep(2)  # Wait 2 seconds between tests
    
    # Test 5: Strafe left 0.5 meters
    print("\n📋 Test 5: Strafing LEFT by 0.5 meters")
    target_pose = np.array([0.5, 0.0, 0.0])  # [y, x, rotation] - y=0.5 means strafe left
    print("Starting strafe with wheel odometry monitoring...")
    success = robot_interface.move_to_base_waypoint(
        target_pose, 
        threshold_pos=0.01,
        threshold_theta=0.01,
        max_steps=100
    )
    if success:
        print("✅ Test 5 PASSED: Robot strafed left successfully")
        final_obs = robot_interface.get_obs()
        final_pose = final_obs["base_pose"]
        print(f"Final position: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]")
    else:
        print("❌ Test 5 FAILED: Robot failed to strafe left")
    
    time.sleep(2)  # Wait 2 seconds between tests
    
    # Test 6: Strafe right 0.5 meters
    print("\n📋 Test 6: Strafing RIGHT by 0.5 meters")
    target_pose = np.array([-0.5, 0.0, 0.0])  # [y, x, rotation] - y=-0.5 means strafe right
    print("Starting strafe with wheel odometry monitoring...")
    success = robot_interface.move_to_base_waypoint(
        target_pose, 
        threshold_pos=0.01,
        threshold_theta=0.01,
        max_steps=100
    )
    if success:
        print("✅ Test 6 PASSED: Robot strafed right successfully")
        final_obs = robot_interface.get_obs()
        final_pose = final_obs["base_pose"]
        print(f"Final position: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]")
    else:
        print("❌ Test 6 FAILED: Robot failed to strafe right")
    
    print("\n🎉 ROBOT CALIBRATION COMPLETE!")
    print("All basic movement tests finished.")
    print("SLAM system continues running - you can close the visualization window when done.")


def main():
    parser = argparse.ArgumentParser(description="Standalone Robot Calibration")
    parser.add_argument("--simulate", action="store_true", 
                       help="Simulate robot commands instead of sending to real robot")
    parser.add_argument("--robot-host", default="localhost",
                       help="Robot RPC server host (default: localhost)")
    parser.add_argument("--robot-port", type=int, default=50000,
                       help="Robot RPC server port (default: 50000)")
    
    args = parser.parse_args()
    
    print("🤖 Starting Standalone Robot Calibration")
    print(f"Robot server: {args.robot_host}:{args.robot_port}")
    print(f"Simulation mode: {args.simulate}")
    
    # Initialize robot interface
    try:
        robot_interface = RobotInterface(simulate=args.simulate)
        print("✅ Robot interface initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize robot interface: {e}")
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