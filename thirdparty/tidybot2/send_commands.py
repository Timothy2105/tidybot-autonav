# Author: Timothy Yu
# Date: July 2025

from cameras import KinovaCamera, LogitechCamera
from constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import BASE_CAMERA_SERIAL, POLICY_CONTROL_PERIOD
#from arm_server import ArmManager
from base_server import BaseManager
import time
import numpy as np

class RealEnv:
    def __init__(self):
        # RPC server connection for base
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to base RPC server, is base_server.py running?') from e
        
        # RPC server connection for arm
        # arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        # try:
        #     arm_manager.connect()
        # except ConnectionRefusedError as e:
        #     raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e
        
        # RPC proxy objects
        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        # self.arm = arm_manager.Arm()
        
        # Cameras
        # self.base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
        # self.wrist_camera = KinovaCamera()
    
    def get_obs(self):
        obs = {}
        obs.update(self.base.get_state())
        # obs.update(self.arm.get_state())
        # obs['base_image'] = self.base_camera.get_image()
        # obs['wrist_image'] = self.wrist_camera.get_image()
        return obs
    
    def reset(self):
        print('Resetting base...')
        self.base.reset()
        # print('Resetting arm...')
        # self.arm.reset()
        print('Robot has been reset')
    
    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        self.base.execute_action(action) # Non-blocking
        # self.arm.execute_action(action) # Non-blocking
    
    def step_base_only(self, action):
        self.base.execute_action(action)  # Non-blocking
    
    def close(self):
        self.base.close()
        # self.arm.close()
        # self.base_camera.close()
        # self.wrist_camera.close()

    def move_to_base_waypoint(self, target_base_pose, threshold_pos=0.01, threshold_theta=0.01):
        """
        Smoothly moves the robot base to a target [x, y, theta] pose via interpolation.
        Based on Priya's proven working implementation.
        
        Args:
            target_base_pose (array-like): [x, y, theta] target for the base.
            threshold_pos (float): Position error threshold for stopping.
            threshold_theta (float): Rotation error threshold (in radians) for stopping.
        
        Returns:
            bool: True if the target is reached.
        """
        obs = self.get_obs()
        curr_base_pose = np.array(obs["base_pose"])
        MAX_STEP = 100
        ALPHA = 0.1  # interpolation factor (0 < ALPHA <= 1)
        step = 0
    
        while True:
            obs = self.get_obs()
            curr_base_pose = np.array(obs["base_pose"])
    
            # Compute errors
            pos_error = target_base_pose[:2] - curr_base_pose[:2]
            theta_error = target_base_pose[2] - curr_base_pose[2]
            pos_error_norm = np.linalg.norm(pos_error)
    
            print(f"[Step {step}] pos_err: {pos_error_norm:.4f}, theta_err: {theta_error:.4f}")
    
            if pos_error_norm < threshold_pos and abs(theta_error) < threshold_theta:
                return True
            elif step > MAX_STEP:
                break
    
            # Interpolate linearly toward target
            next_pose = curr_base_pose.copy()
            next_pose[:2] += ALPHA * pos_error
            next_pose[2] += ALPHA * theta_error
    
            # Execute base-only action
            self.step_base_only({"base_pose": next_pose})
    
            time.sleep(POLICY_CONTROL_PERIOD)
            step += 1
    
        return False

def get_user_input():
    """Get movement commands from user input"""
    print("\n=== Robot Control ===")
    print("Enter movement commands (or 'quit' to quit, 'r' to reset):")
    print("Format: y,x,rotation (e.g., 0.1,0,0 to move up)")
    print("  y: up/down (pos=up, neg=down)")
    print("  x: left/right (pos=left, neg=right)")
    print("  rotation: turn in DEGREES (pos=left, neg=right)")
    print("    OR radians with 'r' suffix (e.g., 1.57r)")
    print("Or use shortcuts:")
    print("  w: up         s: down")
    print("  a: left       d: right")
    print("  q: rotate left    e: rotate right")
    print("  space: stop")
    print("  goto: move to absolute position (e.g., 'goto 1,2,90')")
    print("  current: show current position")
    
    while True:
        try:
            user_input = input("\nCommand: ").strip().lower()
            
            if user_input == 'quit':
                return None
            elif user_input == 'r' or user_input == 'reset':
                return 'reset'
            elif user_input == 'current' or user_input == 'pos':
                return 'current'
            elif user_input.startswith('goto '):
                # Parse goto command: "goto 1,2,90"
                try:
                    coords = user_input[5:].strip()  # Remove "goto "
                    parts = coords.split(',')
                    if len(parts) == 3:
                        y, x, rotation_str = parts
                        y = float(y)
                        x = float(x)
                        
                        rotation_str = rotation_str.strip()
                        if rotation_str.endswith('r'):
                            rotation = float(rotation_str[:-1])
                        else:
                            rotation = np.deg2rad(float(rotation_str))
                        
                        return ('goto', [y, x, rotation])
                    else:
                        print("Invalid goto format. Use: goto y,x,rotation")
                        continue
                except ValueError:
                    print("Invalid goto coordinates. Use numbers only.")
                    continue
            elif user_input == 'w':
                return [0.2, 0, 0]  # positive y = up
            elif user_input == 's':
                return [-0.2, 0, 0]  # negative y = down
            elif user_input == 'a':
                return [0, 0.2, 0]  # positive x = left
            elif user_input == 'd':
                return [0, -0.2, 0]  # negative x = right
            elif user_input == 'q':
                return [0, 0, 0.5]  # positive rotation = left
            elif user_input == 'e':
                return [0, 0, -0.5]  # negative rotation = right
            elif user_input == ' ' or user_input == 'space' or user_input == 'stop':
                return [0, 0, 0]
            else:
                # Try to parse as comma-separated values
                parts = user_input.split(',')
                if len(parts) == 3:
                    y, x, rotation_str = parts
                    y = float(y)
                    x = float(x)
                    
                    # Handle rotation: check if it ends with 'r' for radians
                    rotation_str = rotation_str.strip()
                    if rotation_str.endswith('r'):
                        # Radians
                        rotation = float(rotation_str[:-1])
                    else:
                        # Degrees - convert to radians
                        rotation_deg = float(rotation_str)
                        rotation = np.deg2rad(rotation_deg)
                    
                    # Clamp values to safe ranges
                    y = max(-0.5, min(0.5, y))
                    x = max(-0.5, min(0.5, x))
                    rotation = max(-np.pi, min(np.pi, rotation))  # -180° to 180°
                    
                    print(f"Parsed: y={y}, x={x}, rotation={rotation:.3f}rad ({np.rad2deg(rotation):.1f}°)")
                    return [y, x, rotation]
                else:
                    print("Invalid format. Use y,x,rotation or shortcuts (w/a/s/d/q/e)")
                    print("Examples: '0,0,90' (90°), '0,0,1.57r' (1.57 radians)")
        except ValueError:
            print("Invalid input. Please enter numbers or use shortcuts.")
        except KeyboardInterrupt:
            return None

if __name__ == '__main__':
    env = RealEnv()
    
    try:
        print("Manual Robot Control Started")
        print("Initializing...")
        env.reset()
        
        while True:
            # Get user input
            command = get_user_input()
            
            if command is None:
                print("Exiting...")
                break
            elif command == 'reset':
                env.reset()
                continue
            elif command == 'current':
                obs = env.get_obs()
                current_pose = np.array(obs["base_pose"])
                print(f"Current pose: y={current_pose[0]:.3f}, x={current_pose[1]:.3f}, rotation={current_pose[2]:.3f}rad ({np.rad2deg(current_pose[2]):.1f}°)")
                continue
            elif isinstance(command, tuple) and command[0] == 'goto':
                target_pose = np.array(command[1])
                obs = env.get_obs()
                current_pose = np.array(obs["base_pose"])
                print(f"Moving from {current_pose} to {target_pose}")
                print(f"Target: y={target_pose[0]:.3f}, x={target_pose[1]:.3f}, rotation={target_pose[2]:.3f}rad ({np.rad2deg(target_pose[2]):.1f}°)")
                
                # Use Priya's interpolated movement approach
                success = env.move_to_base_waypoint(target_pose)
                if success:
                    print("Successfully reached target!")
                else:
                    print("Failed to reach target")
                continue
            
            # Single movement command
            action = {
                'base_pose': np.array(command),
                'arm_pos': np.array([0.55, 0.0, 0.4]),  # Default arm position
                'arm_quat': np.array([0, 0, 0, 1]),     # Default arm orientation
                'gripper_pos': np.array([0.5]),         # Default gripper position
            }
            
            # Execute action
            env.step(action)
            
            # Get and display current state
            obs = env.get_obs()
            print(f"Command sent: {command}")
            print(f"Current state: {[(k, v.shape) if hasattr(v, 'ndim') and v.ndim > 0 else (k, v) for (k, v) in obs.items()]}")
            
            time.sleep(POLICY_CONTROL_PERIOD)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        print("Robot connection closed")