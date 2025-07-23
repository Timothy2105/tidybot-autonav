import sys
import os
import time
import numpy as np
from pathlib import Path

# add tidybot path 
tidybot2_path = Path(__file__).parent.parent / "thirdparty" / "tidybot2"
sys.path.insert(0, str(tidybot2_path))

try:
    from constants import BASE_RPC_HOST, BASE_RPC_PORT, RPC_AUTHKEY, POLICY_CONTROL_PERIOD
    from base_server import BaseManager
except ImportError as e:
    print(f"Warning: Could not import TidyBot modules: {e}")
    print("Robot commands will be simulated (printed but not sent)")
    BASE_RPC_HOST = 'localhost'
    BASE_RPC_PORT = 50000
    RPC_AUTHKEY = b'secret password'
    POLICY_CONTROL_PERIOD = 0.1


class RobotInterface:
    def __init__(self, simulate=False):
        """ Initialize robot interface for TidyBot. """
        self.simulate = simulate
        self.base = None
        self.connected = False
        
        if not simulate:
            try:
                # RPC server connection for base
                base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
                base_manager.connect()
                
                # RPC proxy object
                self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
                
                # Initialize the base
                try:
                    self.base.reset()
                    self.connected = True
                    print(f"Connected to TidyBot base at {BASE_RPC_HOST}:{BASE_RPC_PORT}")
                except Exception as e:
                    print(f"Warning: Could not initialize TidyBot base: {e}")
                    print("Robot commands will be simulated (printed but not sent)")
                    self.simulate = True
                    self.connected = False
                
            except Exception as e:
                print(f"Warning: Could not connect to TidyBot: {e}")
                print("Robot commands will be simulated (printed but not sent)")
                self.simulate = True
                self.connected = False
        else:
            print("Robot interface in simulation mode")
    
    def get_state_obs(self):
        """
        Get current robot state observations.
        Returns wheel odometry from base.get_state().
        """
        if self.simulate:
            # return simulated state
            return {'base_pose': np.array([0.0, 0.0, 0.0])}
        
        if self.connected and self.base:
            try:
                state = self.base.get_state()
                # TidyBot returns {'base_pose': vehicle.x} where vehicle.x is [x, y, theta]
                # Convert to our expected format
                if 'base_pose' in state:
                    base_pose = state['base_pose']
                    if hasattr(base_pose, '__len__') and len(base_pose) == 3:
                        return {'base_pose': np.array(base_pose)}
                    else:
                        # If it's a single value, assume it's the x position
                        return {'base_pose': np.array([base_pose, 0.0, 0.0])}
                return state
            except Exception as e:
                print(f"Error getting robot state: {e}")
                return {'base_pose': np.array([0.0, 0.0, 0.0])}
        
        return {'base_pose': np.array([0.0, 0.0, 0.0])}
    
    def get_obs(self):
        """
        Get complete observations (similar to common_real_env.py).
        Currently only includes state observations, but could be extended with camera data.
        """
        obs = {}
        obs.update(self.get_state_obs())
        return obs
    
    def reset(self):
        """Reset the robot base to initial position."""
        if self.simulate:
            print("SIMULATION: Resetting robot base")
            return
        
        if self.connected and self.base:
            try:
                print("Resetting robot base...")
                self.base.reset()
                print("Robot base reset complete")
            except Exception as e:
                print(f"Error resetting robot: {e}")
    
    def get_state(self):
        """Get current robot state (legacy method for compatibility)."""
        return self.get_state_obs()
    
    def execute_action(self, action):
        """
        Execute a robot action.
        
        Args:
            action (dict): Action dictionary with 'base_pose' key containing [y, x, rotation]
                          where y=up/down, x=left/right, rotation=turn left/right
        """
        if self.simulate:
            print(f"SIMULATION: Executing action {action}")
            return
        
        if self.connected and self.base:
            try:
                self.base.execute_action(action)
            except Exception as e:
                print(f"Error executing robot action: {e}")
    
    def rotate_in_place(self, target_angle_rad, threshold_theta=0.005, max_steps=100):
        """
        Rotate the robot in place without moving position.
        
        Args:
            target_angle_rad (float): Target angle in radians
            threshold_theta (float): Rotation error threshold (in radians) for stopping
            max_steps (int): Maximum number of steps before giving up
            
        Returns:
            bool: True if the target rotation is reached
        """
        if self.simulate:
            print(f"SIMULATION: Rotating in place by {np.degrees(target_angle_rad):.1f}°")
            return True
        
        if not self.connected:
            print("Not connected to robot")
            return False
        
        ALPHA = 0.1  # Interpolation factor
        step = 0
        
        # Get current pose
        obs = self.get_obs()
        curr_pose = np.array(obs["base_pose"])
        start_x, start_y = curr_pose[0], curr_pose[1]
        start_theta = curr_pose[2]
        
        # Target is current position with new angle
        target_theta = start_theta + target_angle_rad
        
        print(f"Rotating in place: {np.degrees(target_angle_rad):.1f}° (from {np.degrees(start_theta):.1f}° to {np.degrees(target_theta):.1f}°)")
        
        while step < max_steps:
            # Get current state
            obs = self.get_obs()
            curr_pose = np.array(obs["base_pose"])
            curr_x, curr_y, curr_theta = curr_pose[0], curr_pose[1], curr_pose[2]
            
            # Calculate rotation error
            theta_error = target_theta - curr_theta
            theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))  # Normalize to [-pi, pi]
            
            # Calculate position drift (should stay near start position)
            pos_drift = np.sqrt((curr_x - start_x)**2 + (curr_y - start_y)**2)
            
            print(f"[Step {step}] theta_err: {np.degrees(theta_error):.2f}°, pos_drift: {pos_drift:.4f}m")
            print(f"  Current: [x={curr_x:.3f}, y={curr_y:.3f}, theta={np.degrees(curr_theta):.1f}°]")
            
            if abs(theta_error) < threshold_theta:
                print("Target rotation reached!")
                return True
            
            # Only update rotation, keep position fixed
            next_theta = curr_theta + ALPHA * theta_error
            
            # Create target pose: keep current position, update only angle
            next_pose = np.array([curr_x, curr_y, next_theta])
            
            # Execute action
            self.execute_action({"base_pose": next_pose})
            
            time.sleep(POLICY_CONTROL_PERIOD)
            step += 1
        
        print(f"Failed to reach target rotation after {max_steps} steps")
        return False
    
    def move_forward(self, distance, threshold_pos=0.01, max_steps=100):
        """
        Move the robot forward by a specified distance.
        
        Args:
            distance (float): Distance to move forward (positive) or backward (negative)
            threshold_pos (float): Position error threshold for stopping
            max_steps (int): Maximum number of steps before giving up
            
        Returns:
            bool: True if the target distance is reached
        """
        if self.simulate:
            print(f"SIMULATION: Moving forward by {distance:.3f}m")
            return True
        
        if not self.connected:
            print("Not connected to robot")
            return False
        
        ALPHA = 0.1  # Interpolation factor
        step = 0
        
        # Get current pose
        obs = self.get_obs()
        curr_pose = np.array(obs["base_pose"])
        start_x, start_y, start_theta = curr_pose[0], curr_pose[1], curr_pose[2]
        
        # Calculate target position (move along current heading)
        target_x = start_x + distance * np.cos(start_theta)
        target_y = start_y + distance * np.sin(start_theta)
        
        print(f"Moving forward: {distance:.3f}m (from [{start_x:.3f}, {start_y:.3f}] to [{target_x:.3f}, {target_y:.3f}])")
        
        while step < max_steps:
            # Get current state
            obs = self.get_obs()
            curr_pose = np.array(obs["base_pose"])
            curr_x, curr_y, curr_theta = curr_pose[0], curr_pose[1], curr_pose[2]
            
            # Calculate position error
            pos_error = np.array([target_x - curr_x, target_y - curr_y])
            pos_error_norm = np.linalg.norm(pos_error)
            
            print(f"[Step {step}] pos_err: {pos_error_norm:.4f}m")
            print(f"  Current: [x={curr_x:.3f}, y={curr_y:.3f}, theta={np.degrees(curr_theta):.1f}°]")
            
            if pos_error_norm < threshold_pos:
                print("Target position reached!")
                return True
            
            # Interpolate toward target position
            next_x = curr_x + ALPHA * pos_error[0]
            next_y = curr_y + ALPHA * pos_error[1]
            
            # Keep current orientation
            next_pose = np.array([next_x, next_y, curr_theta])
            
            # Execute action
            self.execute_action({"base_pose": next_pose})
            
            time.sleep(POLICY_CONTROL_PERIOD)
            step += 1
        
        print(f"Failed to reach target position after {max_steps} steps")
        return False

    def move_to_base_waypoint(self, target_base_pose, threshold_pos=0.01, threshold_theta=0.005, max_steps=100):
        """
        Smoothly move the robot base to a target [y, x, theta] pose via interpolation.
        Uses get_obs() to monitor position during movement (similar to common_real_env.py).
        
        Args:
            target_base_pose (array-like): [y, x, theta] target for the base (theta in radians)
            threshold_pos (float): Position error threshold for stopping
            threshold_theta (float): Rotation error threshold (in radians) for stopping
            max_steps (int): Maximum number of steps before giving up
            
        Returns:
            bool: True if the target is reached
        """
        if self.simulate:
            print(f"SIMULATION: Moving to waypoint {target_base_pose}")
            return True
        
        if not self.connected:
            print("Not connected to robot")
            return False
        
        ALPHA = 0.1  # Reduced interpolation factor for smoother movement
        step = 0
        
        # Convert target from [y, x, theta] to TidyBot format [x, y, theta]
        # TidyBot uses global frame [x, y, theta] where x=forward, y=left, theta=rotation
        tidybot_target = np.array([target_base_pose[1], target_base_pose[0], target_base_pose[2]])
        
        print(f"Target in TidyBot format: [x={tidybot_target[0]:.3f}, y={tidybot_target[1]:.3f}, theta={np.degrees(tidybot_target[2]):.1f}°]")
        
        while step < max_steps:
            # Get current state using get_obs() (similar to common_real_env.py)
            obs = self.get_obs()
            curr_base_pose = np.array(obs["base_pose"])
            
            # Ensure we have a 3-element array [x, y, theta]
            if len(curr_base_pose) < 3:
                curr_base_pose = np.array([curr_base_pose[0] if len(curr_base_pose) > 0 else 0.0,
                                          curr_base_pose[1] if len(curr_base_pose) > 1 else 0.0,
                                          curr_base_pose[2] if len(curr_base_pose) > 2 else 0.0])
            
            # TidyBot returns [x, y, theta] in global frame
            curr_tidybot_pose = curr_base_pose
            
            # Compute errors in TidyBot coordinate system
            pos_error = tidybot_target[:2] - curr_tidybot_pose[:2]
            theta_error = tidybot_target[2] - curr_tidybot_pose[2]
            pos_error_norm = np.linalg.norm(pos_error)
            
            # Normalize theta error to [-pi, pi]
            theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
            
            print(f"[Step {step}] pos_err: {pos_error_norm:.4f}, theta_err: {np.degrees(theta_error):.2f}°")
            print(f"  Current: [x={curr_tidybot_pose[0]:.3f}, y={curr_tidybot_pose[1]:.3f}, theta={np.degrees(curr_tidybot_pose[2]):.1f}°]")
            
            if pos_error_norm < threshold_pos and abs(theta_error) < threshold_theta:
                print("Target reached!")
                return True
            
            # Interpolate linearly toward target
            next_pose = curr_tidybot_pose.copy()
            next_pose[:2] += ALPHA * pos_error
            next_pose[2] += ALPHA * theta_error
            
            # Execute base-only action (TidyBot expects [x, y, theta])
            self.execute_action({"base_pose": next_pose})
            
            time.sleep(POLICY_CONTROL_PERIOD)
            step += 1
        
        print(f"Failed to reach target after {max_steps} steps")
        return False
    
    def close(self):
        """Close the robot connection."""
        if self.connected and self.base:
            try:
                self.base.close()
                print("Robot connection closed")
            except Exception as e:
                print(f"Error closing robot connection: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 