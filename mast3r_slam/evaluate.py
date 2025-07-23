import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
import lietorch
import time
import yaml
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes, Frame
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import load_retriever 
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement
import datetime
from pathlib import Path

def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)

    # check if filepath exists
    if dataset.dataset_path:
        seq_name = dataset.dataset_path.stem
    else:
        # live stream
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seq_name = f"live_session_{timestamp}"

    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logfile = pathlib.Path(logdir) / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            if keyframe.frame_id < len(timestamps):
                t = timestamps[keyframe.frame_id]
            else:
                # default
                t = i * 0.1  # 0.1 sec int
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if keyframe.frame_id < len(timestamps):
            t = timestamps[keyframe.frame_id]
        else:
            # default
            t = i * 0.1  # 0.1 sec int
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # xyz rgb array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)

def save_slam_state(save_dir, keyframes, retrieval_database, factor_graph, states, timestamps):
    save_dir = Path(save_dir)
    slam_state_dir = save_dir
    slam_state_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Saving enhanced SLAM state to {slam_state_dir}")

    sequence_metadata = {
        'total_frames_processed': len(timestamps),
        'keyframe_count': len(keyframes),
        'processing_order': [],
        'keyframe_insertion_order': [], 
        'frame_to_keyframe_mapping': {},
        'timestamps': list(timestamps),
        'mode_transitions': [],  # track changing mode
    }

    # save sequentially
    keyframe_insertion_order = []
    frame_to_keyframe_mapping = {}
    
    for kf_idx in range(len(keyframes)):
        kf = keyframes[kf_idx]
        frame_id = kf.frame_id
        keyframe_insertion_order.append(frame_id)
        frame_to_keyframe_mapping[frame_id] = kf_idx
    
    sequence_metadata['keyframe_insertion_order'] = keyframe_insertion_order
    sequence_metadata['frame_to_keyframe_mapping'] = frame_to_keyframe_mapping
    
    sequence_metadata['processing_order'] = list(range(len(timestamps)))

    # intrinsics
    if config["use_calib"]:
        K = keyframes.get_intrinsics()
        sequence_metadata['intrinsics'] = K.cpu().numpy()

    # metadata
    torch.save(sequence_metadata, slam_state_dir / "sequence_metadata.pth")

    # save kf
    keyframes_data = {
        'n_keyframes': len(keyframes),
        'keyframe_data': [],
        'trajectory_data': {
            'poses': [],       
            'pose_timestamps': [],
            'pose_confidence': [],
            'tracking_status': [],
        }
    }

    # detailed kf data
    for i in range(len(keyframes)):
        kf = keyframes[i]
        kf_data = {
            'frame_id': kf.frame_id,
            'keyframe_index': i,
            'T_WC': kf.T_WC.data.cpu().numpy(),
            'X_canon': kf.X_canon.cpu().numpy(),
            'C': kf.C.cpu().numpy(),
            'feat': kf.feat.cpu().numpy(),
            'pos': kf.pos.cpu().numpy(),
            'N': kf.N,
            'N_updates': kf.N_updates,
            'img_shape': kf.img_shape.cpu().numpy(),
            'img_true_shape': kf.img_true_shape.cpu().numpy(),
            'insertion_timestamp': timestamps[kf.frame_id] if kf.frame_id < len(timestamps) else 0,
            'average_confidence': kf.get_average_conf().mean().cpu().numpy() if kf.get_average_conf() is not None else 0,
        }
        keyframes_data['keyframe_data'].append(kf_data)
        
        # traj data
        keyframes_data['trajectory_data']['poses'].append(kf.T_WC.data.cpu().numpy())
        keyframes_data['trajectory_data']['pose_timestamps'].append(timestamps[kf.frame_id] if kf.frame_id < len(timestamps) else 0)
        keyframes_data['trajectory_data']['pose_confidence'].append(kf_data['average_confidence'])
        keyframes_data['trajectory_data']['tracking_status'].append('TRACKING')  # Default, can be enhanced

        # img -> npz
        np.savez_compressed(
            slam_state_dir / f"keyframe_{i:06d}.npz",
            img=kf.img.cpu().numpy(),
            uimg=kf.uimg.cpu().numpy(),
            frame_id=kf.frame_id,
            timestamp=timestamps[kf.frame_id] if kf.frame_id < len(timestamps) else 0
        )

    # kf metadata
    torch.save(keyframes_data, slam_state_dir / "keyframes.pth")

    # factor graph
    graph_data = {
        'ii': factor_graph.ii.cpu(),
        'jj': factor_graph.jj.cpu(),
        'idx_ii2jj': factor_graph.idx_ii2jj.cpu(),
        'idx_jj2ii': factor_graph.idx_jj2ii.cpu(),
        'valid_match_j': factor_graph.valid_match_j.cpu(),
        'valid_match_i': factor_graph.valid_match_i.cpu(),
        'Q_ii2jj': factor_graph.Q_ii2jj.cpu(),
        'Q_jj2ii': factor_graph.Q_jj2ii.cpu(),
        'edge_metadata': {
            'creation_order': [],     
            'edge_types': [],       
            'edge_strengths': [],  
            'optimization_history': []
        }
    }
    
    # analyze edge for type
    ii_list = factor_graph.ii.cpu().tolist()
    jj_list = factor_graph.jj.cpu().tolist()
    
    for idx, (i, j) in enumerate(zip(ii_list, jj_list)):
        if abs(i - j) == 1:
            edge_type = 'consecutive'
        elif abs(i - j) <= 5:
            edge_type = 'local'
        else:
            edge_type = 'loop_closure'
            
        graph_data['edge_metadata']['edge_types'].append(edge_type)
        graph_data['edge_metadata']['creation_order'].append(idx)
        
        # edge strength
        edge_strength = 1.0  # default
        graph_data['edge_metadata']['edge_strengths'].append(edge_strength)

    torch.save(graph_data, slam_state_dir / "factor_graph.pth")

    # retrieval db
    retrieval_data = {
        'modelname': retrieval_database.modelname,
        'kf_counter': retrieval_database.kf_counter,
        'kf_ids': retrieval_database.kf_ids,
        'retrieval_history': {
            'queries': [],  
            'results': [],     
            'query_frames': [],  
            'database_evolution': []
        }
    }
    
    # save curr state of retrieval db
    for i, kf_id in enumerate(retrieval_database.kf_ids):
        retrieval_data['retrieval_history']['database_evolution'].append({
            'step': i,
            'added_keyframe': kf_id,
            'database_size': i + 1
        })

    torch.save(retrieval_data, slam_state_dir / "retrieval_db.pth")

    # save system state
    state_info = {
        'mode': states.get_mode().value if hasattr(states.get_mode(), 'value') else states.get_mode(),
        'last_frame_id': len(keyframes) - 1,
        'processing_complete': True,
        'save_timestamp': time.time(),
        'config_snapshot': dict(config)  # store config
    }
    torch.save(state_info, slam_state_dir / "state_info.pth")

    # summary
    summary = {
        'total_keyframes': len(keyframes),
        'total_edges': len(factor_graph.ii),
        'timeline': sequence_metadata['timestamps'],
        'keyframe_frames': keyframe_insertion_order,
        'final_trajectory_length': len(keyframes_data['trajectory_data']['poses']),
        'processing_stats': {
            'first_timestamp': min(timestamps) if timestamps else 0,
            'last_timestamp': max(timestamps) if timestamps else 0,
            'duration': (max(timestamps) - min(timestamps)) if timestamps else 0,
            'avg_confidence': np.mean([kf['average_confidence'] for kf in keyframes_data['keyframe_data']]) if keyframes_data['keyframe_data'] else 0
        }
    }
    
    with open(slam_state_dir / "summary.yaml", 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"Saved {len(keyframes)} keyframes, {len(factor_graph.ii)} edges, and complete trajectory")
    print(f"Trajectory spans {summary['processing_stats']['duration']:.2f} seconds")
    print(f"Summary saved to {slam_state_dir / 'summary.yaml'}")


def load_slam_state(load_dir, model, keyframes, device="cuda", sequential_replay=True):
    load_dir = Path(load_dir)
    
    print(f"Loading enhanced SLAM state from {load_dir}")

    # load metadata
    sequence_metadata = torch.load(load_dir / "sequence_metadata.pth", map_location=device)
    print(f"Loading sequence with {sequence_metadata['total_frames_processed']} frames, {sequence_metadata['keyframe_count']} keyframes")

    # load intrinsics
    K = None
    if 'intrinsics' in sequence_metadata and config["use_calib"]:
        K = torch.from_numpy(sequence_metadata['intrinsics']).to(device, dtype=torch.float32)
        keyframes.set_intrinsics(K)

    # clear existing kf
    with keyframes.lock:
        keyframes.n_size.value = 0

    # load kf data
    keyframes_data = torch.load(load_dir / "keyframes.pth", map_location=device)
    
    if sequential_replay:
        # load kf in sequential order
        insertion_order = sequence_metadata['keyframe_insertion_order']
        print(f"Replaying keyframe insertion in order: {insertion_order}")
        
        for insertion_step, frame_id in enumerate(insertion_order):
            kf_data = None
            kf_index = None
            for i, kf in enumerate(keyframes_data['keyframe_data']):
                if kf['frame_id'] == frame_id:
                    kf_data = kf
                    kf_index = i
                    break
            
            if kf_data is None:
                print(f"Warning: Could not find keyframe data for frame_id {frame_id}")
                continue
                
            print(f"Inserting keyframe {insertion_step + 1}/{len(insertion_order)}: frame_id={frame_id}")
            
            # load img for kf
            images = np.load(load_dir / f"keyframe_{kf_index:06d}.npz")
            
            # recreate frame
            frame = Frame(
                frame_id=kf_data['frame_id'],
                img=torch.from_numpy(images['img']).to(device),
                img_shape=torch.from_numpy(kf_data['img_shape']).to(device),
                img_true_shape=torch.from_numpy(kf_data['img_true_shape']).to(device),
                uimg=torch.from_numpy(images['uimg']),
                T_WC=lietorch.Sim3(torch.from_numpy(kf_data['T_WC']).to(device)),
                X_canon=torch.from_numpy(kf_data['X_canon']).to(device),
                C=torch.from_numpy(kf_data['C']).to(device),
                feat=torch.from_numpy(kf_data['feat']).to(device),
                pos=torch.from_numpy(kf_data['pos']).to(device),
                N=kf_data['N'],
                N_updates=kf_data['N_updates']
            )

            if K is not None:
                frame.K = K 
            
            keyframes.append(frame)
            
            # delay for debugging
            # time.sleep(1)
    else:
        # load all kf
        for i, kf_data in enumerate(keyframes_data['keyframe_data']):
            images = np.load(load_dir / f"keyframe_{i:06d}.npz")
            
            frame = Frame(
                frame_id=kf_data['frame_id'],
                img=torch.from_numpy(images['img']).to(device),
                img_shape=torch.from_numpy(kf_data['img_shape']).to(device),
                img_true_shape=torch.from_numpy(kf_data['img_true_shape']).to(device),
                uimg=torch.from_numpy(images['uimg']),
                T_WC=lietorch.Sim3(torch.from_numpy(kf_data['T_WC']).to(device)),
                X_canon=torch.from_numpy(kf_data['X_canon']).to(device),
                C=torch.from_numpy(kf_data['C']).to(device),
                feat=torch.from_numpy(kf_data['feat']).to(device),
                pos=torch.from_numpy(kf_data['pos']).to(device),
                N=kf_data['N'],
                N_updates=kf_data['N_updates']
            )

            if K is not None:
                frame.K = K 
            
            keyframes.append(frame)

    # load factor graph
    factor_graph = FactorGraph(model, keyframes, K, device)

    graph_data = torch.load(load_dir / "factor_graph.pth", map_location=device)
    factor_graph.ii = graph_data['ii']
    factor_graph.jj = graph_data['jj']
    factor_graph.idx_ii2jj = graph_data['idx_ii2jj']
    factor_graph.idx_jj2ii = graph_data['idx_jj2ii']
    factor_graph.valid_match_j = graph_data['valid_match_j']
    factor_graph.valid_match_i = graph_data['valid_match_i']
    factor_graph.Q_ii2jj = graph_data['Q_ii2jj']
    factor_graph.Q_jj2ii = graph_data['Q_jj2ii']
    
    # edge stats
    if 'edge_metadata' in graph_data:
        edge_types = graph_data['edge_metadata']['edge_types']
        consecutive_edges = edge_types.count('consecutive')
        local_edges = edge_types.count('local')
        loop_closures = edge_types.count('loop_closure')
        print(f"Loaded edges: {consecutive_edges} consecutive, {local_edges} local, {loop_closures} loop closures")

    # load retrieval db
    retrieval_data = torch.load(load_dir / "retrieval_db.pth", map_location=device)
    retrieval_database = load_retriever(model, retrieval_data['modelname'], device=device)
    retrieval_database.kf_counter = retrieval_data['kf_counter']
    retrieval_database.kf_ids = retrieval_data['kf_ids']

    # rebuild IVF w/ kf
    print("Rebuilding retrieval database...")
    if sequential_replay:
        # insertion order
        for insertion_step, frame_id in enumerate(sequence_metadata['keyframe_insertion_order']):
            kf_idx = sequence_metadata['frame_to_keyframe_mapping'][frame_id]
            frame = keyframes[kf_idx]
            
            feat = retrieval_database.prep_features(frame.feat)
            feat_np = feat[0].cpu().numpy()
            id_np = insertion_step * np.ones(feat_np.shape[0], dtype=np.int64)
            retrieval_database.add_to_database(feat_np, id_np, None)
            
            if (insertion_step + 1) % 10 == 0:
                print(f"Rebuilt retrieval DB: {insertion_step + 1}/{len(sequence_metadata['keyframe_insertion_order'])}")
    else:
        for i in range(len(keyframes)):
            frame = keyframes[i]
            feat = retrieval_database.prep_features(frame.feat)
            feat_np = feat[0].cpu().numpy()
            id_np = i * np.ones(feat_np.shape[0], dtype=np.int64)
            retrieval_database.add_to_database(feat_np, id_np, None)

    # load timestamps and traj data
    timestamps = sequence_metadata['timestamps']
    trajectory_data = keyframes_data['trajectory_data']
    
    print(f"Loaded trajectory with {len(trajectory_data['poses'])} poses")
    print(f"Trajectory confidence range: {min(trajectory_data['pose_confidence']):.3f} - {max(trajectory_data['pose_confidence']):.3f}")

    # load state info
    state_info = torch.load(load_dir / "state_info.pth", map_location=device)
    
    print(f"Successfully loaded:")
    print(f"  - {len(keyframes)} keyframes")
    print(f"  - {len(factor_graph.ii)} edges") 
    print(f"  - Complete trajectory spanning {trajectory_data['pose_timestamps'][-1] - trajectory_data['pose_timestamps'][0]:.2f} seconds")
    print(f"  - Retrieval database with {retrieval_database.kf_counter} entries")

    return factor_graph, retrieval_database, timestamps, K, state_info, trajectory_data