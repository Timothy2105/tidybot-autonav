import pathlib
import re
import cv2
from natsort import natsorted
import numpy as np
import torch
import pyrealsense2 as rs
import yaml
import time 

from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config

HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception as e:
    HAS_TORCHCODEC = False


class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.rgb_files = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics = None
        self.use_calibration = config["use_calib"]
        self.save_results = True

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Call get_image before timestamp for realsense camera
        img = self.get_image(idx)
        timestamp = self.get_timestamp(idx)
        return timestamp, img

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        img = cv2.imread(self.rgb_files[idx])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image(self, idx):
        img = self.read_img(idx)
        if self.use_calibration:
            img = self.camera_intrinsics.remap(img)
        return img.astype(self.dtype) / 255.0

    def get_img_shape(self):
        img = self.read_img(0)
        raw_img_shape = img.shape
        img = resize_img(img, self.img_size)
        # 3XHxW, HxWx3 -> HxW, HxW
        return img["img"][0].shape[1:], raw_img_shape[:2]

    def subsample(self, subsample):
        self.rgb_files = self.rgb_files[::subsample]
        self.timestamps = self.timestamps[::subsample]

    def has_calib(self):
        return self.camera_intrinsics is not None


class TUMDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = tstamp_rgb[:, 0]

        match = re.search(r"freiburg(\d+)", dataset_path)
        idx = int(match.group(1))
        if idx == 1:
            calib = np.array(
                [517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
            )
        if idx == 2:
            calib = np.array(
                [520.9, 521.0, 325.1, 249.7, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172]
            )
        if idx == 3:
            calib = np.array([535.4, 539.2, 320.1, 247.6])
        W, H = 640, 480
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)


class EurocDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        # For Euroc dataset, the distortion is too much to handle for MASt3R.
        # So we always undistort the images, but the calibration will not be used for any later optimization unless specified.
        self.use_calibration = True
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "mav0/cam0/data.csv"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=",", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [
            self.dataset_path / "mav0/cam0/data" / f for f in tstamp_rgb[:, 1]
        ]
        self.timestamps = tstamp_rgb[:, 0]
        with open(self.dataset_path / "mav0/cam0/sensor.yaml") as f:
            self.cam0 = yaml.load(f, Loader=yaml.FullLoader)
        W, H = self.cam0["resolution"]
        intrinsics = self.cam0["intrinsics"]
        distortion = np.array(self.cam0["distortion_coefficients"])
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, W, H, [*intrinsics, *distortion], always_undistort=True
        )

    def read_img(self, idx):
        img = cv2.imread(self.rgb_files[idx], cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class ETH3DDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = tstamp_rgb[:, 0]
        calibration = np.loadtxt(
            self.dataset_path / "calibration.txt",
            delimiter=" ",
            dtype=np.float32,
            skiprows=0,
        )
        _, (H, W) = self.get_img_shape()
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calibration)


class SevenScenesDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(
            list((self.dataset_path / "seq-01").glob("*.color.png"))
        )
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype)
        fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, 640, 480, [fx, fy, cx, cy]
        )


class RealsenseDataset(MonocularDataset):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.pipeline = rs.pipeline()
        # self.h, self.w = 720, 1280
        self.h, self.w = 480, 640
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, self.w, self.h, rs.format.bgr8, 60
        )
        self.profile = self.pipeline.start(self.rs_config)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        # self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        # self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.save_results = True

        if self.use_calibration:
            rgb_intrinsics = self.rgb_profile.get_intrinsics()
            self.camera_intrinsics = Intrinsics.from_calib(
                self.img_size,
                self.w,
                self.h,
                [
                    rgb_intrinsics.fx,
                    rgb_intrinsics.fy,
                    rgb_intrinsics.ppx,
                    rgb_intrinsics.ppy,
                ],
            )

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        frameset = self.pipeline.wait_for_frames()
        timestamp = frameset.get_timestamp()
        timestamp /= 1000
        self.timestamps.append(timestamp)

        rgb_frame = frameset.get_color_frame()
        img = np.asanyarray(rgb_frame.get_data())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        return img

class HybridDataset(MonocularDataset):
    """
    Hybrid dataset that first processes an MP4 file, then switches to Realsense live feed.
    """
    def __init__(self, mp4_path):
        super().__init__()
        self.mp4_path = pathlib.Path(mp4_path)
        
        # Initialize MP4 dataset
        self.mp4_dataset = MP4Dataset(mp4_path)
        
        # Initialize Realsense dataset (but don't start pipeline yet)
        self.realsense_dataset = None
        self.is_live_mode = False
        self.mp4_processed_frames = 0
        
        # Copy initial properties from MP4 dataset
        self.use_calibration = self.mp4_dataset.use_calibration
        self.dataset_path = self.mp4_dataset.dataset_path
        self.save_results = True
        
        # Resolution checking variables
        self.mp4_resolution = None
        self.realsense_resolution = None
        
        # Track total frames processed across both modes
        self.total_frames_processed = 0
        
        # track actual FPS for live mode
        self.live_fps = 30.0 # default estimate
        self.live_fps_tracker = {'start_time': None, 'frame_count': 0}

        # store effective fps after skip-frames for MP4
        self.mp4_effective_fps = self.mp4_dataset.fps / self.mp4_dataset.stride

    def set_live_fps(self, fps):
        if fps > 0:
            self.live_fps = fps

    def _init_realsense(self):
        """Initialize Realsense dataset and verify resolution match."""
        print("Initializing Realsense for hybrid mode...")
        self.realsense_dataset = RealsenseDataset()
        
        # Get MP4 resolution if not already stored
        if self.mp4_resolution is None:
            test_img = self.mp4_dataset.read_img(0)
            self.mp4_resolution = (test_img.shape[1], test_img.shape[0])  # (width, height)
            print(f"MP4 resolution: {self.mp4_resolution[0]}x{self.mp4_resolution[1]}")
        
        # Get Realsense resolution
        self.realsense_resolution = (self.realsense_dataset.w, self.realsense_dataset.h)
        print(f"Realsense resolution: {self.realsense_resolution[0]}x{self.realsense_resolution[1]}")
        
        # Check if resolutions match
        if self.mp4_resolution != self.realsense_resolution:
            print(f"WARNING: Resolution mismatch!")
            print(f"MP4: {self.mp4_resolution[0]}x{self.mp4_resolution[1]}")
            print(f"Realsense: {self.realsense_resolution[0]}x{self.realsense_resolution[1]}")
            print("This may cause issues with the SLAM system.")
            
            # Ask user if they want to continue
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                raise ValueError("Resolution mismatch - aborting hybrid mode")
        else:
            print("Resolution match confirmed!")
            
        # If calibration was used for MP4, try to use it for Realsense too
        if self.use_calibration and self.mp4_dataset.camera_intrinsics:
            print("Attempting to use MP4 calibration for Realsense...")
            # This might need adjustment based on your specific use case
            self.realsense_dataset.use_calibration = True
            
    def switch_to_live(self):
        """Switch from MP4 to live Realsense feed."""
        if not self.is_live_mode:
            print(f"\nSwitching to live mode after processing {self.mp4_processed_frames} MP4 frames...")
            print(f"MP4 was processed at effective FPS: {self.mp4_effective_fps:.2f} (60fps with skip-frames={self.mp4_dataset.stride})") 
            if self.realsense_dataset is None:
                self._init_realsense()
            self.is_live_mode = True
            # Ensure timestamps list is properly initialized from MP4
            if len(self.timestamps) == 0 and len(self.mp4_dataset.timestamps) > 0:
                self.timestamps = self.mp4_dataset.timestamps.copy()
            print(f"Timestamps available: {len(self.timestamps)}")
            print("Now in live mode!")

            # Initialize live FPS tracking   
            self.live_fps_tracker['start_time'] = time.time() 
            self.live_fps_tracker['frame_count'] = 0
            
    def __len__(self):
        if self.is_live_mode:
            return 999999  # Unlimited for live mode
        else:
            return len(self.mp4_dataset)
            
    def __getitem__(self, idx):
        # For hybrid mode, we need to maintain continuous timestamps
        if not self.is_live_mode and idx >= len(self.mp4_dataset):
            self.switch_to_live()
            
        if self.is_live_mode:
            # Track live FPS 
            if self.live_fps_tracker['start_time'] is not None: 
                self.live_fps_tracker['frame_count'] += 1 
                elapsed = time.time() - self.live_fps_tracker['start_time'] 
                if elapsed > 2.0:  # Update FPS every 2 seconds
                    measured_fps = self.live_fps_tracker['frame_count'] / elapsed 
                    self.live_fps = measured_fps        
                    # Reset tracker               
                    self.live_fps_tracker['start_time'] = time.time() 
                    self.live_fps_tracker['frame_count'] = 0
            
            # Get image and create timestamp based on total frames
            img = self.get_image(idx)
            # Create continuous timestamp using live FPS   
            if len(self.timestamps) > 0:
                # Continue from last MP4 timestamp with assumed framerate
                last_timestamp = self.timestamps[-1]
                time_increment = 1.0 / self.live_fps
                timestamp = last_timestamp + time_increment
            else:
                timestamp = self.total_frames_processed / self.live_fps
            self.timestamps.append(timestamp)
            self.total_frames_processed += 1
            return timestamp, img
        else:
            # Use MP4 dataset
            self.mp4_processed_frames = idx + 1
            self.total_frames_processed = idx + 1
            timestamp, img = self.mp4_dataset[idx]
            # Make sure to append timestamp to our list
            if len(self.timestamps) <= idx:
                self.timestamps.append(timestamp)
            return timestamp, img
            
    def get_timestamp(self, idx):
        # Return the timestamp from our maintained list
        if idx < len(self.timestamps):
            return self.timestamps[idx]
        else:
            # This shouldn't happen if __getitem__ is called properly
            print(f"Warning: Timestamp requested for idx {idx} but only have {len(self.timestamps)} timestamps")
            return idx / self.live_fps
            
    def read_img(self, idx):
        if self.is_live_mode:
            # For live mode, always read fresh frame from camera
            return self.realsense_dataset.read_img(0)
        else:
            return self.mp4_dataset.read_img(idx)
            
    def get_image(self, idx):
        if self.is_live_mode:
            # Always read fresh frame for live mode
            img = self.realsense_dataset.read_img(0)
            if self.realsense_dataset.use_calibration and self.realsense_dataset.camera_intrinsics:
                img = self.realsense_dataset.camera_intrinsics.remap(img)
            return img.astype(self.dtype) / 255.0
        else:
            return self.mp4_dataset.get_image(idx)
            
    def get_img_shape(self):
        # Use MP4 dataset for shape since we start with it
        return self.mp4_dataset.get_img_shape()
        
    def subsample(self, subsample):
        # Only subsample the MP4 part
        self.mp4_dataset.subsample(subsample)
        # update effective FPS
        self.mp4_effective_fps = self.mp4_dataset.fps / self.mp4_dataset.stride
        
    def has_calib(self):
        if self.is_live_mode and self.realsense_dataset:
            return self.realsense_dataset.has_calib()
        else:
            return self.mp4_dataset.has_calib()


class Webcam(MonocularDataset):
    def __init__(self):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = None
        # load webcam using opencv
        self.cap = cv2.VideoCapture(-1)
        self.save_results = False

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        ret, img = self.cap.read()
        if not ret:
            raise ValueError("Failed to read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.timestamps.append(idx / 30)

        return img


class MP4Dataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        if HAS_TORCHCODEC:
            self.decoder = VideoDecoder(str(self.dataset_path))
            self.fps = self.decoder.metadata.average_fps
            self.total_frames = self.decoder.metadata.num_frames
        else:
            print("torchcodec is not installed. This may slow down the dataloader")
            self.cap = cv2.VideoCapture(str(self.dataset_path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.stride = config["dataset"]["subsample"]
        # Pre-calculate timestamps for all frames
        self.timestamps = []
        for i in range(self.__len__()):
            # correct timestamp calculation: frame_index * stride
            self.timestamps.append(i * self.stride / self.fps)

    def __len__(self):
        return self.total_frames // self.stride

    def __getitem__(self, idx):
        img = self.get_image(idx)
        timestamp = self.get_timestamp(idx)
        return timestamp, img

    def get_timestamp(self, idx):
        if idx < len(self.timestamps):
            return self.timestamps[idx]
        else:
            # Calculate on the fly if needed
            return idx * self.stride / self.fps

    def read_img(self, idx):
        if HAS_TORCHCODEC:
            img = self.decoder[idx * self.stride]  # c,h,w
            img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx * self.stride)
            ret, img = self.cap.read()
            if not ret:
                raise ValueError("Failed to read image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        return img

    def get_image(self, idx):
        img = self.read_img(idx)
        if self.use_calibration and self.camera_intrinsics:
            img = self.camera_intrinsics.remap(img)
        return img.astype(self.dtype) / 255.0

    def subsample(self, subsample):
        # Update stride and recalculate timestamps
        self.stride = subsample
        self.timestamps = []
        for i in range(self.__len__()):
            # recalculate with new stride
            self.timestamps.append(i * self.stride / self.fps)


class RGBFiles(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(list((self.dataset_path).glob("*.png")))
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype) / 30.0


class Intrinsics:
    def __init__(self, img_size, W, H, K_orig, K, distortion, mapx, mapy):
        self.img_size = img_size
        self.W, self.H = W, H
        self.K_orig = K_orig
        self.K = K
        self.distortion = distortion
        self.mapx = mapx
        self.mapy = mapy
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            np.zeros((H, W, 3)), self.img_size, return_transformation=True
        )
        self.K_frame = self.K.copy()
        self.K_frame[0, 0] = self.K[0, 0] / scale_w
        self.K_frame[1, 1] = self.K[1, 1] / scale_h
        self.K_frame[0, 2] = self.K[0, 2] / scale_w - half_crop_w
        self.K_frame[1, 2] = self.K[1, 2] / scale_h - half_crop_h

    def remap(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    @staticmethod
    def from_calib(img_size, W, H, calib, always_undistort=False):
        if not config["use_calib"] and not always_undistort:
            return None
        fx, fy, cx, cy = calib[:4]
        distortion = np.zeros(4)
        if len(calib) > 4:
            distortion = np.array(calib[4:])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_opt = K.copy()
        mapx, mapy = None, None
        center = config["dataset"]["center_principle_point"]
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (W, H), 0, (W, H), centerPrincipalPoint=center
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, K_opt, (W, H), cv2.CV_32FC1
        )

        return Intrinsics(img_size, W, H, K, K_opt, distortion, mapx, mapy)


def load_dataset(dataset_path, hybrid=False):
    if hybrid:
        if not dataset_path.endswith(('.mp4', '.avi', '.MOV', '.mov')):
            raise ValueError(f"Hybrid mode needs a valid MP4/video file")
        return HybridDataset(dataset_path)

    split_dataset_type = dataset_path.split("/")
    if "tum" in split_dataset_type:
        return TUMDataset(dataset_path)
    if "euroc" in split_dataset_type:
        return EurocDataset(dataset_path)
    if "eth3d" in split_dataset_type:
        return ETH3DDataset(dataset_path)
    if "7-scenes" in split_dataset_type:
        return SevenScenesDataset(dataset_path)
    if "realsense" in split_dataset_type:
        return RealsenseDataset()
    if "webcam" in split_dataset_type:
        return Webcam()

    ext = split_dataset_type[-1].split(".")[-1]
    if ext in ["mp4", "avi", "MOV", "mov"]:
        return MP4Dataset(dataset_path)
    return RGBFiles(dataset_path)
