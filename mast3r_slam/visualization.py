import dataclasses
import weakref
from pathlib import Path
import numpy as np
import os
import re
import time
from plyfile import PlyData

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from mast3r_slam.frame import Mode
from mast3r_slam.geometry import get_pixel_coords
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from mast3r_slam.config import load_config, config, set_global_config


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


class Window(WindowEvents):
    title = "MASt3R-SLAM"
    window_size = (1960, 1080)

    def __init__(self, states, keyframes, main2viz, viz2main, **kwargs):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        self.clear = hex2rgba("#1E2326", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg()
        self.keyframes = keyframes
        self.states = states

        self.show_all = True
        self.show_keyframe_edges = True
        self.culling = True
        self.follow_cam = True

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = True
        self.show_curr_pointmap = True
        self.show_axis = True

        self.textures = dict()
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_img, self.kf_img = Image(), Image()
        self.curr_img_np, self.kf_img_np = None, None

        self.main2viz = main2viz
        self.viz2main = viz2main

        # point cloud clicking 
        self.clicked_points = []  # store clicked point coords
        self.point_clicking_enabled = True
        self.keyframes = keyframes

        # load predicted poses
        self.predicted_poses = self.load_predicted_poses()
        self.show_predicted_poses = True
        self.show_ground_truth_poses = True
        self.load_preds_enabled = False 
        self.selected_keyframe_idx = 0 
        self.debug_printed = False
        
        # cache for ply point cloud data
        self.ply_points = None
        self.ply_colors = None
        self.ply_loaded = False

    def load_predicted_poses(self):
        predicted_poses = []
        
        pred_file = "calib-results/kf-preds/predicted_wheel_poses.txt"
        gt_file = "calib-results/kf-preds/ground_truth_camera_poses_all.txt"
        gt_wheel_file = "calib-results/kf-preds/ground_truth_wheel_poses.txt"
        gt_transposed_wheel_file = "calib-results/kf-preds/ground_truth_transposed_wheel_poses.txt"
        
        if os.path.exists(pred_file) and os.path.exists(gt_file) and os.path.exists(gt_wheel_file):
            try:
                pred_positions = []
                pred_quaternions = []
                
                with open(pred_file, 'r') as f:
                    content = f.read()

                    pos_matches = re.findall(r"Position:\s+\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]", content)
                    quat_matches = re.findall(r"Quaternion:\s+\[w=([-\d.]+),\s*x=([-\d.]+),\s*y=([-\d.]+),\s*z=([-\d.]+)\]", content)
                    
                    for pos_match, quat_match in zip(pos_matches, quat_matches):
                        pos = [float(x) for x in pos_match]
                        quat = [float(x) for x in quat_match]
                        pred_positions.append(pos)
                        pred_quaternions.append(quat)
                
                gt_positions = []
                gt_quaternions = []
                
                with open(gt_file, 'r') as f:
                    content = f.read()

                    pos_matches = re.findall(r"Position:\s+\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]", content)
                    quat_matches = re.findall(r"Quaternion:\s+\[w=([-\d.]+),\s*x=([-\d.]+),\s*y=([-\d.]+),\s*z=([-\d.]+)\]", content)
                    
                    for pos_match, quat_match in zip(pos_matches, quat_matches):
                        pos = [float(x) for x in pos_match]
                        quat = [float(x) for x in quat_match]
                        gt_positions.append(pos)
                        gt_quaternions.append(quat)
                
                gt_wheel_positions = []
                gt_wheel_quaternions = []
                
                with open(gt_wheel_file, 'r') as f:
                    content = f.read()

                    pos_matches = re.findall(r"Position:\s+\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]", content)
                    quat_matches = re.findall(r"Quaternion:\s+\[w=([-\d.]+),\s*x=([-\d.]+),\s*y=([-\d.]+),\s*z=([-\d.]+)\]", content)
                    
                    for pos_match, quat_match in zip(pos_matches, quat_matches):
                        pos = [float(x) for x in pos_match]
                        quat = [float(x) for x in quat_match]
                        gt_wheel_positions.append(pos)
                        gt_wheel_quaternions.append(quat)
                
                gt_transposed_wheel_positions = []
                gt_transposed_wheel_quaternions = []
                
                if os.path.exists(gt_transposed_wheel_file):
                    with open(gt_transposed_wheel_file, 'r') as f:
                        content = f.read()

                        pos_matches = re.findall(r"Position:\s+\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]", content)
                        quat_matches = re.findall(r"Quaternion:\s+\[w=([-\d.]+),\s*x=([-\d.]+),\s*y=([-\d.]+),\s*z=([-\d.]+)\]", content)
                        
                        for pos_match, quat_match in zip(pos_matches, quat_matches):
                            pos = [float(x) for x in pos_match]
                            quat = [float(x) for x in quat_match]
                            gt_transposed_wheel_positions.append(pos)
                            gt_transposed_wheel_quaternions.append(quat)
                
                # store loaded data
                self.gt_positions = gt_positions
                self.gt_quaternions = gt_quaternions
                self.pred_positions = pred_positions
                self.pred_quaternions = pred_quaternions
                self.gt_wheel_positions = gt_wheel_positions
                self.gt_wheel_quaternions = gt_wheel_quaternions
                self.gt_transposed_wheel_positions = gt_transposed_wheel_positions
                self.gt_transposed_wheel_quaternions = gt_transposed_wheel_quaternions
                
                print(f"Loaded {len(pred_positions)} predicted poses, {len(gt_positions)} ground truth camera poses (all 16), {len(gt_wheel_positions)} ground truth wheel poses, and {len(gt_transposed_wheel_positions)} transposed wheel poses from text files")
                
                # convert to Sim3
                N = max(len(pred_positions), len(gt_positions), len(gt_wheel_positions), len(gt_transposed_wheel_positions))
                for i in range(N):
                    if i < len(pred_positions):
                        pred_pos = pred_positions[i]
                        pred_quat = pred_quaternions[i]
                    else:
                        pred_pos = [0, 0, 0]
                        pred_quat = [1, 0, 0, 0]
                    if i < len(gt_positions):
                        gt_pos = gt_positions[i]
                        gt_quat = gt_quaternions[i]
                    else:
                        gt_pos = [0, 0, 0]
                        gt_quat = [1, 0, 0, 0]
                    if i < len(gt_wheel_positions):
                        gt_wheel_pos = gt_wheel_positions[i]
                        gt_wheel_quat = gt_wheel_quaternions[i]
                    else:
                        gt_wheel_pos = [0, 0, 0]
                        gt_wheel_quat = [1, 0, 0, 0]
                    if i < len(gt_transposed_wheel_positions):
                        gt_transposed_wheel_pos = gt_transposed_wheel_positions[i]
                        gt_transposed_wheel_quat = gt_transposed_wheel_quaternions[i]
                    else:
                        gt_transposed_wheel_pos = [0, 0, 0]
                        gt_transposed_wheel_quat = [1, 0, 0, 0]

                    # pred wheel pose
                    pred_sim3_data = np.zeros(8, dtype=np.float32)
                    pred_sim3_data[:3] = pred_pos  # translation (xyz)
                    pred_sim3_data[3:7] = pred_quat  # quaternion (wxyz)
                    pred_sim3_data[7] = 1.0  # scale
                    
                    # gt camera pose
                    gt_sim3_data = np.zeros(8, dtype=np.float32)
                    gt_sim3_data[:3] = gt_pos 
                    gt_sim3_data[3:7] = gt_quat
                    gt_sim3_data[7] = 1.0 
                    
                    # gt wheel pose
                    gt_wheel_sim3_data = np.zeros(8, dtype=np.float32)
                    gt_wheel_sim3_data[:3] = gt_wheel_pos 
                    gt_wheel_sim3_data[3:7] = gt_wheel_quat
                    gt_wheel_sim3_data[7] = 1.0
                    
                    # gt transposed wheel pose
                    gt_transposed_wheel_sim3_data = np.zeros(8, dtype=np.float32)
                    gt_transposed_wheel_sim3_data[:3] = gt_transposed_wheel_pos
                    gt_transposed_wheel_sim3_data[3:7] = gt_transposed_wheel_quat
                    gt_transposed_wheel_sim3_data[7] = 1.0

                    predicted_poses.append({
                        'predicted': lietorch.Sim3(torch.from_numpy(pred_sim3_data).unsqueeze(0).float()),
                        'ground_truth': lietorch.Sim3(torch.from_numpy(gt_sim3_data).unsqueeze(0).float()),
                        'ground_truth_wheel': lietorch.Sim3(torch.from_numpy(gt_wheel_sim3_data).unsqueeze(0).float()),
                        'ground_truth_transposed_wheel': lietorch.Sim3(torch.from_numpy(gt_transposed_wheel_sim3_data).unsqueeze(0).float())
                    })
                
                print(f"Successfully loaded {len(predicted_poses)} predicted wheel poses, ground truth camera poses, and ground truth wheel poses for visualization")
                return predicted_poses
                
            except Exception as e:
                print(f"Error loading predicted poses: {e}")
                return []
        else:
            print(f"Predicted pose files not found: {pred_file}, {gt_file}")
            return []

    def load_ply_data(self):
        if self.ply_loaded:
            return True
            
        ply_file_path = "saved-states/realsense-map/point_cloud.ply"
        if not os.path.exists(ply_file_path):
            print(f"⚠️  PLY file not found: {ply_file_path}")
            return False
        
        try:
            plydata = PlyData.read(ply_file_path)
            vertices = plydata['vertex']
            
            # extract points 
            self.ply_points = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
            self.ply_colors = np.column_stack([vertices['red'], vertices['green'], vertices['blue']])
            
            print(f"📊 Loaded {len(self.ply_points)} points from PLY file for point clicking")
            self.ply_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading PLY file: {e}")
            return False

    def render(self, t: float, frametime: float):
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        curr_frame = self.states.get_frame()
        h, w = curr_frame.img_shape.flatten()
        self.frustums.make_frustum(h, w)

        self.curr_img_np = curr_frame.uimg.numpy()
        self.curr_img.write(self.curr_img_np)

        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
        if self.follow_cam:
            T_WC = cam_T_WC.matrix().numpy().astype(
                dtype=np.float32
            ) @ translation_matrix(np.array([0, 0, -2], dtype=np.float32))
            self.camera.follow_cam(np.linalg.inv(T_WC))
        else:
            self.camera.unfollow_cam()
        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )

        with self.keyframes.lock:
            N_keyframes = len(self.keyframes)
            dirty_idx = self.keyframes.get_dirty_idx()

        for kf_idx in dirty_idx:
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            X = self.frame_X(keyframe)
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)

            if keyframe.frame_id not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures[keyframe.frame_id] = ptex, ctex, itex
                ptex, ctex, itex = self.textures[keyframe.frame_id]
                itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())

        for kf_idx in range(N_keyframes):
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            if kf_idx == N_keyframes - 1:
                self.kf_img_np = keyframe.uimg.numpy()
                self.kf_img.write(self.kf_img_np)

            # check if target keyframe (blue)
            target_kf_idx = -1
            try:
                with self.states.lock:
                    target_kf_idx = self.states.target_keyframe_idx.value
            except (KeyError, AttributeError):
                pass
            
            if kf_idx == target_kf_idx and target_kf_idx >= 0:
                color = [0, 0, 1, 1] # blue target kf
            else:
                color = [1, 0, 0, 1] # red original kf
                
            if self.show_keyframe:
                self.frustums.add(
                    as_SE3(keyframe.T_WC.cpu()),
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            if self.show_all:
                self.render_pointmap(keyframe.T_WC.cpu(), w, h, ptex, ctex, itex)

        # render predicted wheel poses purple
        if self.show_predicted_poses and self.predicted_poses and self.load_preds_enabled:
            for i, pose_data in enumerate(self.predicted_poses):
                if 'predicted' in pose_data:
                    pred_pose = pose_data['predicted']
                    
                    self.frustums.add(
                        as_SE3(pred_pose.cpu()),
                        scale=self.frustum_scale * 0.8,
                        color=[1, 0, 1, 1],
                        thickness=self.line_thickness * self.scale,
                    )

        # render ground truth wheel poses yellow
        if self.show_ground_truth_poses and self.predicted_poses and self.load_preds_enabled:
            for i, pose_data in enumerate(self.predicted_poses):
                if 'ground_truth_wheel' in pose_data:
                    gt_wheel_pose = pose_data['ground_truth_wheel']
                    
                    self.frustums.add(
                        as_SE3(gt_wheel_pose.cpu()),
                        scale=self.frustum_scale * 1.0,
                        color=[1, 1, 0, 1],
                        thickness=self.line_thickness * self.scale * 1.2,
                    )

        # render ground truth camera poses orange
        if self.show_ground_truth_poses and self.predicted_poses and self.load_preds_enabled:
            for i, pose_data in enumerate(self.predicted_poses):
                if 'ground_truth' in pose_data:
                    gt_pose = pose_data['ground_truth']
                    
                    self.frustums.add(
                        as_SE3(gt_pose.cpu()),
                        scale=self.frustum_scale * 1.2,
                        color=[1, 0.5, 0, 1],
                        thickness=self.line_thickness * self.scale * 1.5,
                    )

        # render transposed ground truth wheel poses white
        if self.show_predicted_poses and self.predicted_poses and self.load_preds_enabled:
            for i, pose_data in enumerate(self.predicted_poses):
                if 'ground_truth_transposed_wheel' in pose_data:
                    gt_transposed_wheel_pose = pose_data['ground_truth_transposed_wheel']
                    
                    self.frustums.add(
                        as_SE3(gt_transposed_wheel_pose.cpu()),
                        scale=self.frustum_scale * 1.1,
                        color=[1, 1, 1, 1],
                        thickness=self.line_thickness * self.scale * 1.3,
                    )

        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
            if ii.numel() > 0 and jj.numel() > 0:
                T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )
        if self.show_curr_pointmap and self.states.get_mode() != Mode.INIT:
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()
            h, w = curr_frame.img_shape.flatten()
            X = self.frame_X(curr_frame)
            C = curr_frame.C.cpu().numpy().astype(np.float32)
            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            itex.write(depth2rgb(X[..., -1], colormap="turbo"))
            self.render_pointmap(
                curr_frame.T_WC.cpu(),
                w,
                h,
                ptex,
                ctex,
                itex,
                use_img=True,
                depth_bias=self.depth_bias,
            )

        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self.render_ui()

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)

        imgui.spacing()
        shader_options = [
            "surfelmap.glsl",
            "trianglemap.glsl",
        ]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )

        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i

        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()

        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        
        # checkbox for showing predicted poses
        _, self.show_predicted_poses = imgui.checkbox("show_predicted_poses", self.show_predicted_poses)
        _, self.show_ground_truth_poses = imgui.checkbox("show_ground_truth_poses", self.show_ground_truth_poses)
        
        imgui.spacing()
        imgui.text("Point Cloud Clicking:")
        _, self.point_clicking_enabled = imgui.checkbox("Enable point clicking", self.point_clicking_enabled)
        if self.point_clicking_enabled:
            # check relocalization state
            try:
                with self.keyframes.lock:
                    is_relocalized = self.keyframes.relocalized_flag.value == 1
            except Exception as e:
                is_relocalized = False
                
            if is_relocalized:
                imgui.text("System relocalized - Point clicking active")
                imgui.text("Hold Shift + Left-click on points to get 3D coordinates")
                imgui.text(f"Clicked points: {len(self.clicked_points)}")
                if self.clicked_points:
                    latest_point = self.clicked_points[-1]
                    imgui.text(f"Latest: [{latest_point['position'][0]:.3f}, {latest_point['position'][1]:.3f}, {latest_point['position'][2]:.3f}]")
                    imgui.text(f"Color: RGB({latest_point['color'][0]}, {latest_point['color'][1]}, {latest_point['color'][2]})")
                    imgui.text(f"Distance: {latest_point['distance']:.3f}")
                    
                    # show relative pos if possible
                    if 'relative_pos' in latest_point:
                        imgui.text(f"Relative: [{latest_point['relative_pos'][0]:.3f}, {latest_point['relative_pos'][1]:.3f}, {latest_point['relative_pos'][2]:.3f}]")
            else:
                imgui.text("Waiting for relocalization...")
                imgui.text("Point clicking will be enabled once relocalized")
        else:
            imgui.text("Point clicking disabled")
        
        imgui.spacing()
        imgui.text("Color Legend:")
        imgui.text("  Green: Current camera")
        imgui.text("  Red: Original keyframes")
        imgui.text("  Yellow: Ground truth wheel poses")
        imgui.text("  Blue: Target keyframe")
        imgui.text("  Purple: Predicted wheel poses")
        imgui.text("  Orange: Ground truth camera poses (SLAM)")
        imgui.text("  White: Transposed ground truth wheel poses (x→-z, y→x, z→y)")
        
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()

        gui_size = imgui.get_content_region_available()
        scale = gui_size[0] / self.curr_img.texture.size[0]
        scale = min(self.scale, scale)
        size = (
            self.curr_img.texture.size[0] * scale,
            self.curr_img.texture.size[1] * scale,
        )
        image_with_text(self.kf_img, size, "kf", same_line=False)
        image_with_text(self.curr_img, size, "curr", same_line=False)

        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0):
        w, h = int(w), int(h)
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = T_WC.matrix().numpy().astype(np.float32).T

        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        vao.render(mode=moderngl.POINTS, vertices=w * h)
        vao.release()

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)

    def handle_mouse_click(self, x, y):
        if not self.point_clicking_enabled:
            return
            
        # check if relocalized
        try:
            with self.keyframes.lock:
                is_relocalized = self.keyframes.relocalized_flag.value == 1
        except Exception as e:
            is_relocalized = False
            
        if not is_relocalized:
            print("Point clicking disabled: System not yet relocalized!")
            return
        
        # load ply data
        if not self.load_ply_data():
            print("   Point clicking requires the PLY file to be saved with the SLAM state.")
            return
        
        points = self.ply_points
        colors = self.ply_colors
        
        # get camera matrices
        view_matrix = self.camera.gl_matrix()
        proj_matrix = self.camera.proj_mat.gl_matrix()
        
        # convert screen coords to normalized device coords
        window_size = self.wnd.size
        ndc_x = (2.0 * x) / window_size[0] - 1.0
        ndc_y = 1.0 - (2.0 * y) / window_size[1]
        
        # create ray from camera through click point
        ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
        ray_eye = np.linalg.inv(proj_matrix) @ ray_clip
        ray_eye[2] = -1.0
        ray_eye[3] = 0.0
        ray_world = np.linalg.inv(view_matrix) @ ray_eye
        ray_direction = ray_world[:3] / np.linalg.norm(ray_world[:3])
        ray_origin = -view_matrix[:3, :3].T @ view_matrix[:3, 3]
        
        # find closest point in ply data
        closest_point = None
        closest_distance = float('inf')
        closest_point_idx = -1
        
        # vector from ray origin for all points
        vecs_to_points = points - ray_origin
        
        # project onto ray direction
        projection_lengths = np.dot(vecs_to_points, ray_direction)
        
        # closest points on ray to each 3D point
        closest_points_on_ray = ray_origin + projection_lengths[:, np.newaxis] * ray_direction
        
        # distances from 3D points to closest points on ray
        distances = np.linalg.norm(points - closest_points_on_ray, axis=1)
        
        # filter points that are in front of the camera and within reasonable distance
        valid_mask = (projection_lengths > 0) & (distances < 0.5)  # 0.5m threshold
        
        if np.any(valid_mask):
            # find closest valid point
            valid_indices = np.where(valid_mask)[0]
            closest_idx = valid_indices[np.argmin(distances[valid_indices])]
            closest_point = points[closest_idx]
            closest_distance = distances[closest_idx]
            closest_point_idx = closest_idx
            
            # get color of clicked point
            clicked_color = colors[closest_idx]
            
            # get current camera position
            curr_frame = self.states.get_frame()
            if curr_frame is not None:
                cam_T_WC = curr_frame.T_WC
                cam_translation = cam_T_WC.data[0, :3].cpu().numpy()
                
                # calculate relative pos from camera to clicked point
                relative_pos = closest_point - cam_translation
                
                # store clicked point data with relative pos
                point_data = {
                    'position': closest_point.copy(),
                    'color': clicked_color,
                    'point_idx': closest_point_idx,
                    'distance': closest_distance,
                    'timestamp': time.time(),
                    'relative_pos': relative_pos,
                    'camera_pos': cam_translation
                }
                
                # print clicked point info with camera reference
                print(f"Point clicked at: [{closest_point[0]:.3f}, {closest_point[1]:.3f}, {closest_point[2]:.3f}]")
                print(f"   Camera position: [{cam_translation[0]:.3f}, {cam_translation[1]:.3f}, {cam_translation[2]:.3f}]")
                print(f"   Relative to camera: [{relative_pos[0]:.3f}, {relative_pos[1]:.3f}, {relative_pos[2]:.3f}]")
                print(f"   Color: RGB({clicked_color[0]}, {clicked_color[1]}, {clicked_color[2]})")
                print(f"   Distance: {closest_distance:.3f}")
                print(f"   Point index: {closest_point_idx}")
            else:
                # fallback if no current frame
                point_data = {
                    'position': closest_point.copy(),
                    'color': clicked_color,
                    'point_idx': closest_point_idx,
                    'distance': closest_distance,
                    'timestamp': time.time()
                }
                
                print(f"Point clicked at: [{closest_point[0]:.3f}, {closest_point[1]:.3f}, {closest_point[2]:.3f}]")
                print(f"   Color: RGB({clicked_color[0]}, {clicked_color[1]}, {clicked_color[2]})")
                print(f"   Distance: {closest_distance:.3f}")
                print(f"   Point index: {closest_point_idx}")
            
            self.clicked_points.append(point_data)
            
            # send clicked point to main.py
            self.states.clicked_point[0] = float(closest_point[0])
            self.states.clicked_point[1] = float(closest_point[1])
            self.states.clicked_point[2] = float(closest_point[2])
            self.states.point_clicked.value = 1
        else:
            print("No valid points found within clicking threshold")

    def mouse_press_event(self, x, y, button):
        super().mouse_press_event(x, y, button)
        
        # left click + shift for point selection
        if button == 1:  # Left mouse button
            # Check if Shift key is held down
            import glfw
            shift_pressed = glfw.get_key(self.wnd._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                           glfw.get_key(self.wnd._window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            
            if shift_pressed:
                print(f"Click detected at ({x}, {y})")
                self.handle_mouse_click(x, y)


def run_visualization(cfg, states, keyframes, main2viz, viz2main, load_preds=False, enable_click=False) -> None:
    set_global_config(cfg)

    config_cls = Window
    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=config_cls.title,
        size=config_cls.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = config_cls(
        states=states,
        keyframes=keyframes,
        main2viz=main2viz,
        viz2main=viz2main,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
    )
    
    # Set the load_preds flag
    window_config.load_preds_enabled = load_preds
    
    # Set the enable_click flag
    window_config.point_clicking_enabled = enable_click
    
    # Avoid the event assigning in the property setter for now
    # We want the even assigning to happen in WindowConfig.__init__
    # so users are free to assign them in their own __init__.
    window._config = weakref.ref(window_config)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    while not window.is_closing:
        current_time, delta = timer.next_frame()

        if window_config.clear_color is not None:
            window.clear(*window_config.clear_color)

        # Always bind the window framebuffer before calling render
        window.use()

        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    state = window_config.state
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)
