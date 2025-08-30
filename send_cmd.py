import argparse
import json
import time
import numpy as np
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "mast3r_slam"))
from robot_interface import RobotInterface


class RobotCommandServer:
    def __init__(self, simulate=False, robot_host="192.168.1.4", robot_port=50000):
        self.simulate = simulate
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.robot_interface = None
        self.running = True
        self.command_file = "calib-results/runtime/robot_commands.txt"
        self.result_file = "calib-results/runtime/robot_results.txt"
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print(" Robot Command Server Starting")
        print(f"Simulation mode: {simulate}")
        print(f"Robot server: {robot_host}:{robot_port}")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n Received signal {signum}, shutting down")
        self.running = False
        if self.robot_interface:
            self.robot_interface.close()
        sys.exit(0)
    
    def initialize_robot(self):
        try:
            self.robot_interface = RobotInterface(simulate=self.simulate)
            print(" Robot interface initialized")
            
            if not self.simulate:
                # Test connection
                print("Testing robot connection...")
                obs = self.robot_interface.get_obs()
                print(f"Current robot position: {obs['base_pose']}")
                print(" Robot connection successful")
            
            return True
            
        except Exception as e:
            print(f" Failed to initialize robot interface: {e}")
            if not self.simulate:
                print("\nTroubleshooting tips:")
                print("1. Make sure the TidyBot server is running (python base_server.py)")
                print("2. Check that the robot host/port are correct")
                print("3. Try running with --simulate first to test the script")
            return False
    
    def read_command(self):
        try:
            if os.path.exists(self.command_file):
                with open(self.command_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        os.makedirs(os.path.dirname(self.command_file), exist_ok=True)
                        with open(self.command_file, 'w') as f:
                            f.write("")
                        return json.loads(content)
        except Exception as e:
            print(f"Error reading command: {e}")
        return None
    
    def write_result(self, result):
        try:
            os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
            with open(self.result_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Error writing result: {e}")
    
    def execute_command(self, command):
        try:
            command_type = command.get('type', 'unknown')
            
            if command_type == 'tidybot_command':
                tidybot_cmd = command['command']  # [x, y, rotation]
                x_movement = tidybot_cmd[0]  # left/right
                y_movement = tidybot_cmd[1]  # forward/backward
                theta_rotation = tidybot_cmd[2]  # rotation in degrees
                
                print(f" Executing TidyBot command: [{x_movement:.3f}, {y_movement:.3f}, {theta_rotation:.1f}°]")
                print(f"   X (left/right): {x_movement:.3f}m")
                print(f"   Y (forward/back): {y_movement:.3f}m")
                print(f"   Theta (rotation): {theta_rotation:.1f}°")
                
                def get_base_pose(base):
                    obs = base.get_state()
                    if isinstance(obs, dict) and 'base_pose' in obs:
                        return np.array(obs['base_pose'])
                    raise RuntimeError(f"base.get_state() did not return a dict with 'base_pose': got {obs}")

                def move_base_to(target_pose, base, description):
                    print(f"\n Command: {description}")
                    while True:
                        base.execute_action({'base_pose': target_pose})  # <-- send every loop!
                        state = get_base_pose(base)
                        print(f"Current pose: x={state[0]:.3f}, y={state[1]:.3f}, theta={np.degrees(state[2]):.1f}°")
                        pos_error = np.linalg.norm(state[:2] - target_pose[:2])
                        theta_error = np.abs(np.arctan2(np.sin(target_pose[2] - state[2]), np.cos(target_pose[2] - state[2])))
                        if pos_error < 0.01 and theta_error < 0.015: 
                            print("Target reached!")
                            break
                        time.sleep(0.1)

                # get base object
                base = self.robot_interface.base if hasattr(self.robot_interface, 'base') else self.robot_interface
                
                # get current pose
                current_pose = get_base_pose(base)
                print(f"Current robot pose: x={current_pose[0]:.3f}, y={current_pose[1]:.3f}, theta={np.degrees(current_pose[2]):.1f}°")
                
                # calculate target pose: current + movement
                target_x = current_pose[0] + x_movement
                target_y = current_pose[1] + y_movement
                target_theta = current_pose[2] + np.radians(theta_rotation)
                target_pose = np.array([target_x, target_y, target_theta])
                
                print(f"Target pose: x={target_x:.3f}, y={target_y:.3f}, theta={np.degrees(target_theta):.1f}°")
                
                # execute the movement
                move_base_to(target_pose, base, f"TidyBot command: [{x_movement:.3f}, {y_movement:.3f}, {theta_rotation:.1f}°]")
                
                # get final position from wheel odometry
                final_pose = get_base_pose(base)
                
                return {
                    'success': True,
                    'final_pose': final_pose.tolist(),
                    'message': 'TidyBot command executed successfully'
                }
            
            elif command_type == 'move_to_waypoint':
                target_pose = np.array(command['target_pose'])
                threshold_pos = command.get('threshold_pos', 0.01)
                threshold_theta = command.get('threshold_theta', 0.01)
                max_steps = command.get('max_steps', 100)
                
                print(f" Executing: move_to_waypoint {target_pose}")
                
                def get_base_pose(base):
                    obs = base.get_state()
                    if isinstance(obs, dict) and 'base_pose' in obs:
                        return np.array(obs['base_pose'])
                    raise RuntimeError(f"base.get_state() did not return a dict with 'base_pose': got {obs}")

                def move_base_to(target_pose, base, description):
                    print(f"\n Command: {description}")
                    while True:
                        base.execute_action({'base_pose': target_pose})  # <-- send every loop!
                        state = get_base_pose(base)
                        print(f"Current pose: x={state[0]:.3f}, y={state[1]:.3f}, theta={np.degrees(state[2]):.1f}°")
                        pos_error = np.linalg.norm(state[:2] - target_pose[:2])
                        theta_error = np.abs(np.arctan2(np.sin(target_pose[2] - state[2]), np.cos(target_pose[2] - state[2])))
                        if pos_error < threshold_pos and theta_error < threshold_theta:
                            print("Target reached!")
                            break
                        time.sleep(0.1)

                # get the base object
                base = self.robot_interface.base if hasattr(self.robot_interface, 'base') else self.robot_interface
                
                # execute the movement
                move_base_to(target_pose, base, f"Move to waypoint {target_pose}")
                
                # get final position from wheel odometry
                final_pose = get_base_pose(base)
                
                return {
                    'success': True,
                    'final_pose': final_pose.tolist(),
                    'message': 'Target reached successfully'
                }
            
            elif command_type == 'get_pose':
                def get_base_pose(base):
                    obs = base.get_state()
                    if isinstance(obs, dict) and 'base_pose' in obs:
                        return np.array(obs['base_pose'])
                    raise RuntimeError(f"base.get_state() did not return a dict with 'base_pose': got {obs}")
                
                base = self.robot_interface.base if hasattr(self.robot_interface, 'base') else self.robot_interface
                pose = get_base_pose(base)
                
                return {
                    'success': True,
                    'pose': pose.tolist(),
                    'message': 'Current pose retrieved'
                }
            
            elif command_type == 'reset':
                print(" Executing: reset")
                self.robot_interface.reset()
                
                return {
                    'success': True,
                    'message': 'Robot reset complete'
                }
            
            elif command_type == 'execute_action':
                action = command['action']
                
                print(f" Executing: execute_action {action}")
                
                base = self.robot_interface.base if hasattr(self.robot_interface, 'base') else self.robot_interface
                base.execute_action(action)
                
                return {
                    'success': True,
                    'message': 'Action executed successfully'
                }
            
            else:
                return {
                    'success': False,
                    'message': f'Unknown command type: {command_type}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error executing command: {str(e)}'
            }
    
    def run(self):
        if not self.initialize_robot():
            return 1
        
        print(" Robot command server ready")
        print("Waiting for commands from main.py...")
        print("Commands will be read from 'calib-results/runtime/robot_commands.txt'")
        print("Results will be written to 'calib-results/runtime/robot_results.txt'")
        print("Press Ctrl+C to stop the server")
        
        # clear any existing command file
        if os.path.exists(self.command_file):
            os.remove(self.command_file)
        
        while self.running:
            try:
                # read command
                command = self.read_command()
                
                if command:
                    print(f"\n Received command: {command}")
                    
                    # execute command
                    result = self.execute_command(command)
                    
                    # write result
                    self.write_result(result)
                    
                    print(f" Result: {result}")
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n Interrupted by user")
                break
            except Exception as e:
                print(f" Error in main loop: {e}")
                time.sleep(1)
        
        # cleanup
        if self.robot_interface:
            self.robot_interface.close()
        
        print(" Robot command server stopped")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Robot Command Server")
    parser.add_argument("--simulate", action="store_true", 
                       help="Simulate robot commands instead of sending to real robot")
    parser.add_argument("--robot-host", default="192.168.1.4",
                       help="Robot RPC server host (default: 192.168.1.4)")
    parser.add_argument("--robot-port", type=int, default=50000,
                       help="Robot RPC server port (default: 50000)")
    
    args = parser.parse_args()
    
    if not args.simulate:
        print("\n" + "="*60)
        print("IMPORTANT: Before running this script, you need to start the TidyBot server!")
        print("1. Open a new terminal")
        print("2. Navigate to the TidyBot directory:")
        print("   cd MASt3R-SLAM/thirdparty/tidybot2")
        print("3. Start the base server:")
        print("   python controlled_base_server.py")
        print("4. Wait for the server to start, then run this command server")
        print("="*60)
        
        response = input("Have you started the TidyBot server? (y/n): ")
        if response.lower() != 'y':
            print("Please start the TidyBot server first, then run this script again.")
            return 1
    
    server = RobotCommandServer(
        simulate=args.simulate,
        robot_host=args.robot_host,
        robot_port=args.robot_port
    )
    
    return server.run()


if __name__ == "__main__":
    sys.exit(main()) 