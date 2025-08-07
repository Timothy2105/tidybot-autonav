import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
import json
import numpy as np
import os
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset, HybridDataset
import mast3r_slam.evaluate as eval
from mast3r_slam.evaluate import load_slam_state
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
from mast3r_slam.robot_interface import RobotInterface
import torch.multiprocessing as mp
import os


class RobotCommandSender:    
    def __init__(self):
        self.command_file = "robot_commands.txt"
        self.result_file = "robot_results.txt"
    
    def send_command(self, command):
        try:
            with open(self.command_file, 'w') as f:
                json.dump(command, f, indent=2)
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def read_result(self, timeout=10.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if os.path.exists(self.result_file):
                    with open(self.result_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            # clear the file after reading
                            with open(self.result_file, 'w') as f:
                                f.write("")
                            return json.loads(content)
            except Exception as e:
                print(f"Error reading result: {e}")
            time.sleep(0.1)
        return None
    
    def move_to_waypoint(self, target_pose, threshold_pos=0.01, threshold_theta=0.01, max_steps=100):
        command = {
            'type': 'move_to_waypoint',
            'target_pose': target_pose.tolist(),
            'threshold_pos': threshold_pos,
            'threshold_theta': threshold_theta,
            'max_steps': max_steps
        }
        
        if self.send_command(command):
            result = self.read_result()
            if result:
                return result.get('success', False)
        return False
    
    def get_pose(self):
        command = {'type': 'get_pose'}
        
        if self.send_command(command):
            result = self.read_result()
            if result and result.get('success', False):
                return np.array(result.get('pose', [0, 0, 0]))
        return np.array([0, 0, 0])
    
    def reset(self):
        command = {'type': 'reset'}
        
        if self.send_command(command):
            result = self.read_result()
            return result.get('success', False) if result else False
        return False


def find_nearest_keyframe(current_pose, trajectory_data, keyframes, original_kf_count=None):
    """
    Find the nearest keyframe in the loaded trajectory to the current camera pose.
    
    Args:
        current_pose: Current camera pose as Sim3 transformation
        trajectory_data: Loaded trajectory data containing poses
        keyframes: SharedKeyframes object containing loaded keyframes
        original_kf_count: Number of original keyframes loaded (if None, search all keyframes)
    
    Returns:
        tuple: (nearest_kf_idx, nearest_kf_pose, distance, movement_required, rotation_required)
    """
    if current_pose is None:
        return None, None, None, None, None
        
    current_translation = current_pose.data[0, :3].cpu().numpy()
    
    min_distance = float('inf')
    nearest_kf_idx = -1
    nearest_kf_pose = None
    
    # Determine which keyframes to search through
    if original_kf_count is not None:
        # Only search through the original loaded keyframes
        search_range = min(original_kf_count, len(keyframes))
        print(f"🔍 Searching through original {search_range} keyframes for trajectory following")
    else:
        # Search through all keyframes
        search_range = len(keyframes)
        print(f"🔍 Searching through all {search_range} keyframes")
    
    # Search through keyframes in the loaded trajectory
    for i in range(search_range):
        kf_pose = keyframes[i].T_WC
        if kf_pose is None:
            continue
        kf_translation = kf_pose.data[0, :3].cpu().numpy()
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(current_translation - kf_translation)
        
        if distance < min_distance:
            min_distance = distance
            nearest_kf_idx = i
            nearest_kf_pose = kf_pose
    
    if nearest_kf_idx == -1 or nearest_kf_pose is None:
        return None, None, None, None, None
    
    # Calculate movement required to reach the nearest keyframe
    # This is the difference in translation
    movement_required = nearest_kf_pose.data[0, :3].cpu().numpy() - current_translation
    
    # Calculate rotation difference
    # Sim3 data format: [quaternion (4), translation (3), scale (1)]
    # Extract quaternion and convert to rotation matrix
    current_quat = current_pose.data[0, 3:7].cpu().numpy()  # quaternion part
    target_quat = nearest_kf_pose.data[0, 3:7].cpu().numpy()  # quaternion part
    
    def quaternion_to_rotation_matrix(q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
    
    current_rotation = quaternion_to_rotation_matrix(current_quat)
    target_rotation = quaternion_to_rotation_matrix(target_quat)
    
    # Calculate relative rotation: R_rel = R_target * R_current^T
    relative_rotation = target_rotation @ current_rotation.T
    
    # Convert to Euler angles (roll, pitch, yaw) in degrees
    # Using the standard aerospace sequence (ZYX)
    def rotation_matrix_to_euler_angles(R):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
            pitch = np.arctan2(-R[2, 0], sy) * 180 / np.pi
            yaw = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1]) * 180 / np.pi
            pitch = np.arctan2(-R[2, 0], sy) * 180 / np.pi
            yaw = 0
        
        return np.array([roll, pitch, yaw])
    
    rotation_required = rotation_matrix_to_euler_angles(relative_rotation)
    
    return nearest_kf_idx, nearest_kf_pose, min_distance, movement_required, rotation_required


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
                
                # Print camera position after successful relocalization
                T_WC = keyframes.T_WC[n_kf - 1]
                # Extract translation (position) from Sim3 transformation
                translation = T_WC.data[0, :3].cpu().numpy()  # x, y, z coordinates
                print(f"🎯 RELOCALIZED CAMERA POSITION: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}")
                
                # Set relocalization flag for trajectory following
                keyframes.relocalized_flag.value = 1
                print("🔧 BACKEND: Set relocalized_flag to 1")
                
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K, graph_state=None, retrieval_state=None):
    set_global_config(cfg)
    device = keyframes.device

    # Create new objects in the backend process
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    # If state was passed, restore it
    if graph_state:
        print("Backend: Restoring Factor Graph state")
        factor_graph.ii = graph_state['ii'].to(device)
        factor_graph.jj = graph_state['jj'].to(device)
        factor_graph.idx_ii2jj = graph_state['idx_ii2jj'].to(device)
        factor_graph.idx_jj2ii = graph_state['idx_jj2ii'].to(device)
        factor_graph.valid_match_j = graph_state['valid_match_j'].to(device)
        factor_graph.valid_match_i = graph_state['valid_match_i'].to(device)
        factor_graph.Q_ii2jj = graph_state['Q_ii2jj'].to(device)
        factor_graph.Q_jj2ii = graph_state['Q_jj2ii'].to(device)

    if retrieval_state:
        print("Backend: Restoring Retrieval Database state")
        retrieval_database.kf_counter = retrieval_state['kf_counter']
        retrieval_database.kf_ids = retrieval_state['kf_ids']
        # Rebuild the search index with the loaded keyframes
        print("Backend: Rebuilding retrieval database index...")
        for i in range(len(keyframes)):
            frame = keyframes[i]
            feat = retrieval_database.prep_features(frame.feat)
            feat_np = feat[0].cpu().numpy()
            id_np = i * np.ones(feat_np.shape[0], dtype=np.int64)
            retrieval_database.add_to_database(feat_np, id_np, None)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        
        # Check if we should save state
        with states.lock:
            if states.save_info.get('should_save', False):
                save_dir = states.save_info['save_dir']
                timestamps = states.save_info['timestamps']
                
                print(f"Backend: Saving state to {save_dir}")
                eval.save_slam_state(save_dir, keyframes, retrieval_database, 
                                   factor_graph, states, timestamps)
                
                # Reset flag
                states.save_info['should_save'] = False
        
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
            
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
            
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument("--skip-frames", type=int, default=None, 
                       help="Number of frames to skip when processing video files. For example, --skip-frames 60 will process every 60th frame (skip 59 frames between each processed frame). Default is 1 (no skipping).")
    parser.add_argument("--hybrid", action="store_true",
                       help="Enable hybrid mode: process MP4 first, then switch to live Realsense feed")
    parser.add_argument("--load-state", default="", help="Path to saved SLAM state")
    parser.add_argument("--save-state", default="", help="Directory to save SLAM state")
    parser.add_argument("--follow-traj", action="store_true",
                       help="Enable trajectory following mode: find nearest keyframe and calculate movement required")
    parser.add_argument("--send-cmd", action="store_true",
                       help="Enable robot command sending: actually send commands to TidyBot")
    parser.add_argument("--simulate-robot", action="store_true",
                       help="Simulate robot commands instead of sending to real robot for testing")
    parser.add_argument("--calib-robot", action="store_true",
                       help="Enable robot calibration mode: test basic movement commands")
    parser.add_argument("--load-preds", action="store_true",
                       help="Load prediction data and enable prediction visualization")
    parser.add_argument("--enable-click", action="store_true",
                       help="Enable point cloud clicking functionality to get 3D coordinates")

    args = parser.parse_args()
    
    if args.load_state and not args.dataset:
        print("Error: --load-state requires a dataset to be specified")
        sys.exit(1)

    if args.follow_traj and not args.load_state:
        print("Error: --follow-traj requires --load-state to be specified")
        sys.exit(1)

    # Validation for hybrid mode
    if args.hybrid:
        if not args.dataset.endswith(('.mp4', '.avi', '.MOV', '.mov')):
            print("Error: Hybrid mode requires an MP4/video file as --dataset")
            sys.exit(1)
        print(f"Hybrid mode enabled: Will process {args.dataset} then switch to Realsense")

    if args.follow_traj:
        print("Trajectory following mode enabled - will find nearest keyframe and calculate movement")

    load_config(args.config)
    
    # Override subsample config if skip-frames is provided
    if args.skip_frames is not None:
        config["dataset"]["subsample"] = args.skip_frames
        print(f"Frame skipping set to: {args.skip_frames}")
    
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset, hybrid=args.hybrid)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main, args.load_preds, args.enable_click),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    graph_state = None
    retrieval_state = None

    # handle state loading or initial creation
    if args.load_state:
        print(f"Loading SLAM state from {args.load_state}")
        
        # Load state in the main process
        factor_graph, retrieval_database, saved_timestamps, K_loaded, state_info, trajectory_data = load_slam_state(
            args.load_state, model, keyframes, device
        )
        
        if K_loaded is not None:
            K = K_loaded

        # Extract pickle-able state data to pass to the backend
        graph_state = {
            'ii': factor_graph.ii.cpu(), 'jj': factor_graph.jj.cpu(),
            'idx_ii2jj': factor_graph.idx_ii2jj.cpu(), 'idx_jj2ii': factor_graph.idx_jj2ii.cpu(),
            'valid_match_j': factor_graph.valid_match_j.cpu(), 'valid_match_i': factor_graph.valid_match_i.cpu(),
            'Q_ii2jj': factor_graph.Q_ii2jj.cpu(), 'Q_jj2ii': factor_graph.Q_jj2ii.cpu(),
        }
        retrieval_state = {
            'kf_counter': retrieval_database.kf_counter,
            'kf_ids': retrieval_database.kf_ids,
        }

        if args.hybrid and isinstance(dataset, HybridDataset):
            dataset.timestamps = saved_timestamps
            dataset.switch_to_live()
            print("Switched to live mode after loaded map")

        states.set_mode(Mode.TRACKING)

    # If loading state, skip backend and tracker, just replay for visualization and saving
    if args.load_state:
        print("Replaying saved keyframes for visualization...")
        
        # Set up the edge information for trajectory visualization
        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()
        
        last_msg = WindowMsg()
        i = 0
        fps_timer = time.time()
        current_fps = 30.0 # default fps estimate
        frames = []
        N_keyframes = len(keyframes)
        
        # Use trajectory timestamps if available, otherwise use saved timestamps
        # Ensure we have enough timestamps for all keyframes
        if 'pose_timestamps' in trajectory_data and len(trajectory_data['pose_timestamps']) >= N_keyframes:
            all_timestamps = trajectory_data['pose_timestamps']
        else:
            # Create a mapping from frame_id to timestamp for keyframes
            keyframe_timestamps = {}
            for kf_idx in range(N_keyframes):
                kf = keyframes[kf_idx]
                if kf.frame_id < len(saved_timestamps):
                    keyframe_timestamps[kf.frame_id] = saved_timestamps[kf.frame_id]
                else:
                    # If frame_id is out of range, use a default timestamp
                    keyframe_timestamps[kf.frame_id] = kf_idx * 0.1  # 0.1 second intervals
            
            # Create a list of timestamps in keyframe order
            all_timestamps = [keyframe_timestamps[keyframes[i].frame_id] for i in range(N_keyframes)]
        
        # Visualization loop for replay
        while True:
            msg = try_get_msg(viz2main)
            last_msg = msg if msg is not None else last_msg
            if last_msg.is_terminated:
                states.set_mode(Mode.TERMINATED)
                break
            if last_msg.is_paused and not last_msg.next:
                states.pause()
                time.sleep(0.01)
                continue
            if not last_msg.is_paused:
                states.unpause()
            if i >= N_keyframes:
                print("Replay finished. Continuing with live Realsense feed...")
                break
            # Set the current frame for visualization
            with keyframes.lock:
                frame = keyframes[i]
                states.set_frame(frame)
            if save_frames:
                frames.append(frame.img.cpu().numpy())
            if i % 30 == 0:
                current_fps = i / (time.time() - fps_timer)
                print(f"[REPLAY] Frame: {i} (FPS: {current_fps})")
            i += 1
        
        # save results from replay
        if dataset.save_results:
            save_dir, seq_name = eval.prepare_savedir(args, dataset)
            eval.save_traj(save_dir, f"{seq_name}_replay.txt", all_timestamps, keyframes)
            eval.save_reconstruction(
                save_dir,
                f"{seq_name}_replay.ply",
                keyframes,
                last_msg.C_conf_threshold,
            )
            eval.save_keyframes(
                save_dir / "keyframes" / f"{seq_name}_replay", all_timestamps, keyframes
            )
        if save_frames:
            savedir = pathlib.Path(f"logs/frames/{datetime_now}_replay")
            savedir.mkdir(exist_ok=True, parents=True)
            for i, frame_img in tqdm.tqdm(enumerate(frames), total=len(frames)):
                frame = (frame_img * 255).clip(0, 255)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{savedir}/{i}.png", frame)
        print("Replay completed. Starting live processing...")

    # start backend process, passing only pickle-able data
    backend = mp.Process(
        target = run_backend,
        args = (config, model, states, keyframes, K, graph_state, retrieval_state)
    )
    backend.start()

    # remove the trajectory from previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    if args.load_state:
        i = 0
        print(f"Starting live processing from frame {i}")
    else:
        i = 0
    
    fps_timer = time.time()
    current_fps = 30.0 # default fps estimate

    frames = []
    
    # track when we switch
    switched_to_live = False
    
    # track if we've successfully relocalized
    has_relocalized = False
    
    # track original keyframe count
    original_kf_count = None
    if args.load_state:
        original_kf_count = len(keyframes)
        print(f"Original keyframe count: {original_kf_count}")
    
    # initialize robot interface
    robot_interface = None
    robot_command_sender = None
    if args.send_cmd:
        simulate = args.simulate_robot
        if args.use_command_server:
            robot_command_sender = RobotCommandSender()
            print(f"Robot command sender initialized (will send commands to send_cmd.py)")
        else:
            robot_interface = RobotInterface(simulate=simulate)
        print(f"Robot interface initialized (simulate={simulate})")

    # load prediction data
    if args.load_preds:
        print("Loading prediction data...")
        try:
            import numpy as np
            pred_file = "calib-results/kf-preds/predicted_wheel_poses.npz"
            gt_file = "calib-results/kf-preds/ground_truth_camera_poses.npz"
            
            if os.path.exists(pred_file) and os.path.exists(gt_file):
                # load prediction data
                pred_data = np.load(pred_file)
                gt_data = np.load(gt_file)
                
                print(f"Loaded {len(pred_data['positions'])} prediction poses")
                
                # assume the first 4 keyframes correspond to the predictions
                prediction_kf_indices = list(range(min(4, len(keyframes))))
                
                # set prediction keyframe indices
                with states.lock:
                    states.prediction_keyframe_indices[:] = prediction_kf_indices
                
                print(f"Set prediction keyframe indices: {prediction_kf_indices}")
            else:
                print(f"Warning: Prediction files not found: {pred_file}, {gt_file}")
        except Exception as e:
            print(f"Error loading prediction data: {e}")

    # track if calibration has already run
    calibration_ran = False

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        # check for clicked points from visualization
        if args.enable_click:
            try:
                with keyframes.lock:
                    # only process clicks if relocalized
                    is_relocalized = keyframes.relocalized_flag.value == 1
                    if is_relocalized and states.point_clicked.value == 1:
                        clicked_coords = np.array(states.clicked_point[:])
                        print(f"MAIN: Point clicked at coordinates: [{clicked_coords[0]:.3f}, {clicked_coords[1]:.3f}, {clicked_coords[2]:.3f}]")
                        
                        # reset flag
                        states.point_clicked.value = 0
                        
            except Exception as e:
                pass

        # check if we've finished MP4 and need to switch
        if args.hybrid and not switched_to_live and isinstance(dataset, HybridDataset):
            if i >= len(dataset.mp4_dataset):
                print(f"\nFinished processing MP4 ({i} frames). Switching to live mode...")
                dataset.switch_to_live()
                switched_to_live = True
                print("Continue processing with live Realsense feed...")

        # in hybrid live mode, we never reach the end
        if not (args.hybrid and switched_to_live) and i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # initialize via mono inference, and encoded features needed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)
            
            # check if we just successfully relocalized (tracking after reloc)
            if (args.follow_traj or args.calib_robot) and not has_relocalized:
                print(f" In TRACKING mode, checking for relocalization (has_relocalized={has_relocalized})")
                # check if the backend relocalized
                with keyframes.lock:
                    flag_value = keyframes.relocalized_flag.value
                    print(f" Checking relocalized_flag.value = {flag_value}")
                    if flag_value == 1:
                        with states.lock:
                            states.relocalized.value = 1
                        has_relocalized = True
                        print(" Relocalization detected - Starting post-relocalization action!")
                        # reset flag
                        keyframes.relocalized_flag.value = 0
                        print(" Reset relocalized_flag to 0")

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # in single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # in single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        
        # track camera position continuously
        if args.load_state:  # only track position when we have a loaded state
            current_frame = states.get_frame()
            if current_frame is not None:
                T_WC = current_frame.T_WC
                translation = T_WC.data[0, :3].cpu().numpy()
                if i % 15 == 0:
                    print(f"📷 LIVE CAMERA POSITION: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}")
                
                # write camera position and orientation to file for calibration script to read
                try:
                    # get camera orientation from current frame
                    current_frame = states.get_frame()
                    if current_frame is not None:
                        T_WC = current_frame.T_WC
                        # Extract quaternion from Sim3 transformation
                        quaternion = T_WC.data[0, 3:7].cpu().numpy()  # quaternion part
                        
                        camera_data = {
                            'x': float(translation[0]),
                            'y': float(translation[1]),
                            'z': float(translation[2]),
                            'quaternion': [float(q) for q in quaternion],  # [w, x, y, z]
                            'timestamp': time.time()
                        }
                        with open('camera_position.txt', 'w') as f:
                            json.dump(camera_data, f)
                except Exception as e:
                    pass
                
                # trajectory following mode - only after relocalization
                if args.follow_traj and trajectory_data is not None:
                    # check if we've successfully relocalized
                    with states.lock:
                        has_relocalized = states.relocalized.value == 1
                    
                    if has_relocalized:
                        nearest_kf_idx, nearest_kf_pose, distance, movement_required, rotation_required = find_nearest_keyframe(
                            T_WC, trajectory_data, keyframes, original_kf_count=original_kf_count
                        )
                        
                        if nearest_kf_idx is not None and nearest_kf_pose is not None and distance is not None and movement_required is not None and rotation_required is not None:
                            nearest_translation = nearest_kf_pose.data[0, :3].cpu().numpy()
                            
                            # get current camera orientation
                            # sim3 data format: [quaternion (4), translation (3), scale (1)]
                            # extract quaternion and convert to rotation matrix
                            current_quat = T_WC.data[0, 3:7].cpu().numpy()  # quaternion part
                            target_quat = nearest_kf_pose.data[0, 3:7].cpu().numpy()  # quaternion part
                            
                            def quaternion_to_rotation_matrix(q):
                                """Convert quaternion to rotation matrix"""
                                w, x, y, z = q
                                return np.array([
                                    [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                                    [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                                    [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
                                ])
                            
                            current_rotation = quaternion_to_rotation_matrix(current_quat)
                            target_rotation = quaternion_to_rotation_matrix(target_quat)
                            
                            def rotation_matrix_to_euler_angles(R):
                                """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees"""
                                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                                singular = sy < 1e-6
                                
                                if not singular:
                                    roll = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
                                    pitch = np.arctan2(-R[2, 0], sy) * 180 / np.pi
                                    yaw = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
                                else:
                                    roll = np.arctan2(-R[1, 2], R[1, 1]) * 180 / np.pi
                                    pitch = np.arctan2(-R[2, 0], sy) * 180 / np.pi
                                    yaw = 0
                                
                                return np.array([roll, pitch, yaw])
                            
                            current_euler = rotation_matrix_to_euler_angles(current_rotation)
                            target_euler = rotation_matrix_to_euler_angles(target_rotation)
                            
                            # Calculate movement and rotation in camera's local coordinate frame
                            
                            # Create rotation matrix from current camera orientation
                            def euler_to_rotation_matrix(roll, pitch, yaw):
                                """Convert Euler angles to rotation matrix (ZYX convention)"""
                                roll_rad = np.radians(roll)
                                pitch_rad = np.radians(pitch)
                                yaw_rad = np.radians(yaw)
                                
                                # Rotation matrices
                                Rx = np.array([[1, 0, 0], [0, np.cos(roll_rad), -np.sin(roll_rad)], [0, np.sin(roll_rad), np.cos(roll_rad)]])
                                Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)], [0, 1, 0], [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
                                Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
                                
                                return Rz @ Ry @ Rx
                            
                            # Get current camera rotation matrix
                            current_rot_matrix = euler_to_rotation_matrix(current_euler[0], current_euler[1], current_euler[2])
                            
                            # Transform movement from world coordinates to camera coordinates
                            # Camera coordinates: X=forward, Y=left, Z=up
                            movement_in_camera_frame = current_rot_matrix.T @ movement_required
                            
                            # Extract movement components in camera frame
                            forward_movement = movement_in_camera_frame[0]  # X-axis (forward/backward)
                            left_movement = movement_in_camera_frame[1]     # Y-axis (left/right)
                            up_movement = movement_in_camera_frame[2]       # Z-axis (up/down)
                            
                            # Calculate Y-axis rotation from quaternions (camera frame)
                            # Convert quaternions to Euler angles and extract Y-axis rotation (pitch)
                            def quaternion_to_euler_angles(q):
                                """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
                                w, x, y, z = q
                                
                                # Roll (x-axis rotation)
                                sinr_cosp = 2 * (w * x + y * z)
                                cosr_cosp = 1 - 2 * (x * x + y * y)
                                roll = np.arctan2(sinr_cosp, cosr_cosp)
                                
                                # Pitch (y-axis rotation)
                                sinp = 2 * (w * y - z * x)
                                if abs(sinp) >= 1:
                                    pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
                                else:
                                    pitch = np.arcsin(sinp)
                                
                                # Yaw (z-axis rotation)
                                siny_cosp = 2 * (w * z + x * y)
                                cosy_cosp = 1 - 2 * (y * y + z * z)
                                yaw = np.arctan2(siny_cosp, cosy_cosp)
                                
                                return np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])
                            
                            # Get current and target Euler angles from quaternions
                            current_euler_from_quat = quaternion_to_euler_angles(current_quat)
                            target_euler_from_quat = quaternion_to_euler_angles(target_quat)
                            
                            # Calculate Y-axis rotation difference (pitch)
                            y_rotation_deg = target_euler_from_quat[1] - current_euler_from_quat[1]
                            
                            # Normalize to [-180, 180] degrees
                            while y_rotation_deg > 180:
                                y_rotation_deg -= 360
                            while y_rotation_deg < -180:
                                y_rotation_deg += 360
                            
                            print(f"TRAJECTORY FOLLOWING (After Relocalization):")
                            print(f"   Current Frame: {i}")
                            print(f"   Current Position: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}")
                            print(f"   Current Orientation: roll={current_euler[0]:.1f}°, pitch={current_euler[1]:.1f}°, yaw={current_euler[2]:.1f}°")
                            print(f"   Target Keyframe: {nearest_kf_idx} (Frame ID: {keyframes[nearest_kf_idx].frame_id})")
                            print(f"   Target Position: x={nearest_translation[0]:.3f}, y={nearest_translation[1]:.3f}, z={nearest_translation[2]:.3f}")
                            print(f"   Target Orientation: roll={target_euler[0]:.1f}°, pitch={target_euler[1]:.1f}°, yaw={target_euler[2]:.1f}°")
                            print(f"   Distance to Target: {distance:.3f}")
                            print(f"   World Movement: dx={movement_required[0]:.3f}, dy={movement_required[1]:.3f}, dz={movement_required[2]:.3f}")
                            print(f"   Camera Movement: forward={forward_movement:+.3f}, left={left_movement:+.3f}, up={up_movement:+.3f}")
                            print(f"   Total Movement: {np.linalg.norm(movement_required):.3f}")
                            print(f"   World Rotation Required: roll={rotation_required[0]:.1f}°, pitch={rotation_required[1]:.1f}°, yaw={rotation_required[2]:.1f}°")
                            print(f"   Camera Y-Axis Rotation (from quaternion): {y_rotation_deg:+.1f}°")
                            print(f"   Total World Rotation: {np.linalg.norm(rotation_required):.1f}°")
                            
                            # Generate robot movement suggestions
                            print(f"ROBOT COMMANDS:")
                            if abs(y_rotation_deg) > 5.0:  # If camera Y-axis rotation > 5 degrees
                                print(f"   1. ROTATE: {y_rotation_deg:+.1f}° (camera Y-axis)")
                            if abs(forward_movement) > 0.05:  # If forward movement > 5cm
                                direction = "FORWARD" if forward_movement > 0 else "BACKWARD"
                                print(f"   2. MOVE {direction}: {abs(forward_movement):.3f}m")
                            if abs(left_movement) > 0.05:  # If lateral movement > 5cm
                                direction = "LEFT" if left_movement > 0 else "RIGHT"
                                print(f"   3. STRAFE {direction}: {abs(left_movement):.3f}m")
                            if abs(up_movement) > 0.05:  # If vertical movement > 5cm
                                direction = "UP" if up_movement > 0 else "DOWN"
                                print(f"   4. MOVE {direction}: {abs(up_movement):.3f}m")
                            print("   ---")
                            
                            # TidyBot Commands [y, x, rotation]
                            # y: pos=up, neg=down
                            # x: pos=left, neg=right  
                            # rotation: pos=rotate_left, neg=rotate_right
                            print(f"TIDYBOT COMMANDS:")
                            
                            # Convert movement to TidyBot format
                            tidybot_y = 0  # No up/down movement (camera is static)
                            tidybot_x = -left_movement  # Invert left movement for TidyBot coordinate system
                            tidybot_rotation_deg = -y_rotation_deg  # Use camera Y-axis rotation from quaternion
                            
                            # Apply thresholds to avoid tiny movements
                            if abs(tidybot_x) < 0.05:  # Less than 5cm
                                tidybot_x = 0
                            if abs(tidybot_rotation_deg) < 5.0:  # Less than 5 degrees
                                tidybot_rotation_deg = 0
                            
                            print(f"🤖 TIDYBOT COMMANDS:")
                            print(f"   TidyBot Command: [{tidybot_y:.3f}, {tidybot_x:.3f}, {tidybot_rotation_deg:.1f}°]")
                            print(f"   Explanation:")
                            if tidybot_y != 0:
                                print(f"     Y: {tidybot_y:+.3f} ({'UP' if tidybot_y > 0 else 'DOWN'})")
                            if tidybot_x != 0:
                                print(f"     X: {tidybot_x:+.3f} ({'LEFT' if tidybot_x > 0 else 'RIGHT'})")
                            if tidybot_rotation_deg != 0:
                                print(f"     Rotation: {tidybot_rotation_deg:+.1f}° ({'ROTATE LEFT' if tidybot_rotation_deg > 0 else 'ROTATE RIGHT'})")
                            if tidybot_y == 0 and tidybot_x == 0 and tidybot_rotation_deg == 0:
                                print(f"     ✅ No movement required (within thresholds)")
                            print("   ---")
                            
                            # Send command to robot if enabled
                            if args.send_cmd and robot_interface is not None:
                                print("SENDING COMMAND TO ROBOT...")
                                
                                # Convert to radians for robot
                                rotation_rad = np.radians(tidybot_rotation_deg)
                                
                                # Create target pose for robot
                                # TidyBot format: [y, x, rotation] where y=up/down, x=left/right, rotation=turn
                                target_pose = np.array([tidybot_y, tidybot_x, rotation_rad])
                                
                                # Use incremental movement to reach target with improved position monitoring
                                print("Starting robot movement with wheel odometry monitoring...")
                                success = robot_interface.move_to_base_waypoint(
                                    target_pose, 
                                    threshold_pos=0.01,  # 1cm position threshold
                                    threshold_theta=0.01,  # ~0.6 degrees rotation threshold
                                    max_steps=100
                                )
                                
                                if success:
                                    print("✅ Robot successfully reached target position using wheel odometry!")
                                    # Get final position for verification
                                    final_obs = robot_interface.get_obs()
                                    final_pose = final_obs["base_pose"]
                                    print(f"Final robot position: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]")
                                else:
                                    print("❌ Robot failed to reach target position")
                                
                                # Close robot interface
                                robot_interface.close()
                            
                            # send command to robot command server if enabled
                            if args.send_cmd and robot_command_sender is not None:
                                print("SENDING COMMAND TO ROBOT COMMAND SERVER...")
                                
                                # convert to radians for robot
                                rotation_rad = np.radians(tidybot_rotation_deg)
                                
                                # create target pose for robot
                                target_pose = np.array([tidybot_y, tidybot_x, rotation_rad])
                                
                                # send command to the command server
                                print("Starting robot movement via command server...")
                                success = robot_command_sender.move_to_waypoint(
                                    target_pose, 
                                    threshold_pos=0.01,  # 1cm position threshold
                                    threshold_theta=0.01,  # ~0.6 degrees rotation threshold
                                    max_steps=100
                                )
                                
                                if success:
                                    print(" Robot successfully reached target position via command server!")
                                    # get final position for verification
                                    final_pose = robot_command_sender.get_pose()
                                    print(f"Final robot position: [x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, theta={np.degrees(final_pose[2]):.1f}°]")
                                else:
                                    print(" Robot failed to reach target position via command server")

                            # save target keyframe image for visualization
                            follow_traj_dir = "follow-traj"
                            os.makedirs(follow_traj_dir, exist_ok=True)
                            
                            # get the target keyframe image
                            target_keyframe = keyframes[nearest_kf_idx]
                            target_img = target_keyframe.uimg.numpy()  # Get the image as numpy array
                            
                            # Convert from [0,1] range to [0,255] and to BGR for OpenCV
                            target_img_uint8 = (target_img * 255).astype(np.uint8)
                            target_img_bgr = cv2.cvtColor(target_img_uint8, cv2.COLOR_RGB2BGR)
                            
                            # Save the target keyframe image (overwrites each time)
                            target_filename = f"{follow_traj_dir}/target_keyframe.png"
                            cv2.imwrite(target_filename, target_img_bgr)
                            
                            print(f"Target keyframe image saved: {target_filename}")
                            print(f"  - Keyframe {nearest_kf_idx} (Frame {target_keyframe.frame_id})")
                            print(f"  - Target position: x={nearest_translation[0]:.3f}, y={nearest_translation[1]:.3f}, z={nearest_translation[2]:.3f}")
                            
                            # Set target keyframe for visualization
                            with states.lock:
                                states.target_keyframe_idx.value = nearest_kf_idx
                            print(f"🎯 VISUALIZATION: Set target keyframe to {nearest_kf_idx} (should appear BLUE in viewer)")
                            
                            # After giving the movement command, end the script
                            # print(f"GIVEN MOVEMENT COMMAND - Ending trajectory following.")
                            # states.set_mode(Mode.TERMINATED)
                            # break
                    else:
                        # Only print this message occasionally to avoid spam
                        if i % 30 == 0:
                            print("Waiting for relocalization before starting trajectory following...")
        
        # After relocalization, notify that calibration can be run separately
        if args.calib_robot and has_relocalized and not calibration_ran:
            print("\n🎯 RELOCALIZATION COMPLETE!")
            print("SLAM system is now running continuously.")
            print("\n" + "="*60)
            print("TO RUN CALIBRATION:")
            print("1. Open a new terminal")
            print("2. Run: python robot_calibration.py")
            print("3. Press Enter when prompted")
            print("="*60)
            print("SLAM will continue running while calibration runs in the other terminal.")
            calibration_ran = True

        # log time
        if i % 30 == 0:
            current_fps = i / (time.time() - fps_timer)
            print(f"FPS: {current_fps}")

            # update live fps in hybriddataset if in live mode 
            if args.hybrid and switched_to_live and isinstance(dataset, HybridDataset):
                dataset.set_live_fps(current_fps)
                print(f"[LIVE MODE] Frame: {i} (Live FPS: {current_fps})")
        i += 1

          
    if args.save_state:        
        with states.lock:
            states.save_info['should_save'] = True
            states.save_info['save_dir'] = args.save_state
            states.save_info['timestamps'] = dataset.timestamps
        
        print(f"Signaling backend to save state to: {args.save_state}")
        time.sleep(2)  # Give backend time to save

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        
        # For hybrid mode, ensure we have all timestamps available
        if args.hybrid and isinstance(dataset, HybridDataset):
            # Make sure all timestamps are available
            all_timestamps = dataset.timestamps
            print(f"Saving results with {len(all_timestamps)} timestamps for {len(keyframes)} keyframes")
        else:
            all_timestamps = dataset.timestamps
            
        eval.save_traj(save_dir, f"{seq_name}.txt", all_timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, all_timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    backend.join()
    print("done")
    if not args.no_viz:
        viz.join()
    
    # Clean up robot interface
    if robot_interface is not None:
        robot_interface.close()
    if robot_command_sender is not None:
        print("Robot command sender cleanup complete")