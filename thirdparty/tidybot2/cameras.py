# Author: Jimmy Wu
# Date: October 2024

import matplotlib
matplotlib.use('Agg')

import threading
import time
import cv2 as cv
import numpy as np
import os
# from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
# from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
# from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2
# from kinova import DeviceConnection
from constants import BASE_CAMERA_SERIAL, WRIST_CAMERA_SERIAL, DEPTH_CAMERA_SERIAL
import matplotlib.pyplot as plt

import pyrealsense2 as rs


class Camera:
    def __init__(self):
        self.image = None
        self.last_read_time = time.time()
        threading.Thread(target=self.camera_worker, daemon=True).start()

    def camera_worker(self):
        # Note: We read frames at 30 fps but not every frame is necessarily
        # saved during teleop or used during policy inference
        while True:
            # Reading new frames too quickly causes latency spikes
            while time.time() - self.last_read_time < 0.0333:  # 30 fps
                time.sleep(0.0001)
            _, bgr_image = self.cap.read()
            self.last_read_time = time.time()
            if bgr_image is not None:
                self.image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    def get_image(self):
        return self.image

    def close(self):
        self.cap.release()

class LogitechCamera(Camera):
    def __init__(self, serial, frame_width=640, frame_height=360, focus=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focus = focus  # Note: Set this to 100 when using fisheye lens attachment
        self.serial = serial
        self.cap = self.get_cap(serial)
        super().__init__()

    def get_cap(self, serial):
        print(f"Attempting to open camera with serial {serial}")
        
        # Try opening the camera with different methods
        cap = None
        if serial == WRIST_CAMERA_SERIAL:
            # For wrist camera, try video1 first
            methods = [
                lambda: cv.VideoCapture(1),  # Try video1 first for wrist camera
                lambda: cv.VideoCapture('/dev/video1'),
                lambda: cv.VideoCapture(0),  # Fallback to video0
                lambda: cv.VideoCapture('/dev/video0'),
            ]
        else:
            # For base camera, try video0 first
            methods = [
                lambda: cv.VideoCapture(0),  # Try video0 first for base camera
                lambda: cv.VideoCapture('/dev/video0'),
                lambda: cv.VideoCapture(1),  # Fallback to video1
                lambda: cv.VideoCapture('/dev/video1'),
            ]
        
        for method in methods:
            try:
                cap = method()
                if cap.isOpened():
                    print(f"Successfully opened camera using {method.__name__}")
                    break
            except Exception as e:
                print(f"Failed to open camera using {method.__name__}: {e}")
                continue
        
        if cap is None or not cap.isOpened():
            raise Exception(f"Could not open camera device")
            
        # Set the format to MJPG (Motion-JPEG) which is supported by the camera
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Important - results in much better latency

        # Disable autofocus
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

        # Read several frames to let settings (especially gain/exposure) stabilize
        for _ in range(30):
            ret, _ = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame during initialization")
            cap.set(cv.CAP_PROP_FOCUS, self.focus)  # Fixed focus

        # Print actual camera settings
        print(f"Camera settings for {serial}:")
        print(f"Width: {cap.get(cv.CAP_PROP_FRAME_WIDTH)} (expected {self.frame_width})")
        print(f"Height: {cap.get(cv.CAP_PROP_FRAME_HEIGHT)} (expected {self.frame_height})")
        print(f"Buffer size: {cap.get(cv.CAP_PROP_BUFFERSIZE)}")
        print(f"Autofocus: {cap.get(cv.CAP_PROP_AUTOFOCUS)}")
        print(f"Focus: {cap.get(cv.CAP_PROP_FOCUS)}")

        # Check all settings match expected
        assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == self.frame_width
        assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == self.frame_height
        assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
        assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
        assert cap.get(cv.CAP_PROP_FOCUS) == self.focus

        return cap

def find_fisheye_center(image):
    # Find contours
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Fit a minimum enclosing circle around all contours
    return cv.minEnclosingCircle(np.vstack(contours))

def check_fisheye_centered(image):
    height, width, _ = image.shape
    center, _ = find_fisheye_center(image)
    if center is None:
        return True
    return abs(width / 2 - center[0]) < 0.05 * width and abs(height / 2 - center[1]) < 0.05 * height

class KinovaCamera(Camera):
    def __init__(self):
        # GStreamer video capture (see https://github.com/Kinovarobotics/kortex/issues/88)
        # Note: max-buffers=1 and drop=true are added to reduce latency spikes
        self.cap = cv.VideoCapture('rtspsrc location=rtsp://192.168.1.10/color latency=0 ! decodebin ! videoconvert ! appsink sync=false max-buffers=1 drop=true', cv.CAP_GSTREAMER)
        # self.cap = cv.VideoCapture('rtsp://192.168.1.10/color', cv.CAP_FFMPEG)  # This stream is high latency but works with pip-installed OpenCV
        assert self.cap.isOpened(), 'Unable to open stream. Please make sure OpenCV was built from source with GStreamer support.'

        # Apply camera settings
        threading.Thread(target=self.apply_camera_settings, daemon=True).start()
        super().__init__()

        # Wait for camera to warm up
        image = None
        while image is None:
            image = self.get_image()

        # Make sure fisheye lens did not accidentally get bumped
        if not check_fisheye_centered(image):
            raise Exception('The fisheye lens on the Kinova wrist camera appears to be off-center')

    def apply_camera_settings(self):
        # Note: This function adds significant camera latency when it is called
        # directly in __init__, so we call it in a separate thread instead

        # Use Kortex API to set camera settings
        with DeviceConnection.createTcpConnection() as router:
            device_manager = DeviceManagerClient(router)
            vision_config = VisionConfigClient(router)

            # Get vision device ID
            device_handles = device_manager.ReadAllDevices()
            vision_device_ids = [
                handle.device_identifier for handle in device_handles.device_handle
                if handle.device_type == DeviceConfig_pb2.VISION
            ]
            assert len(vision_device_ids) == 1
            vision_device_id = vision_device_ids[0]

            # Check that resolution, frame rate, and bit rate are correct
            sensor_id = VisionConfig_pb2.SensorIdentifier()
            sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_settings = vision_config.GetSensorSettings(sensor_id, vision_device_id)
            try:
                assert sensor_settings.resolution == VisionConfig_pb2.RESOLUTION_640x480  # FOV 65 ± 3° (diagonal)
                assert sensor_settings.frame_rate == VisionConfig_pb2.FRAMERATE_30_FPS
                assert sensor_settings.bit_rate == VisionConfig_pb2.BITRATE_10_MBPS
            except:
                sensor_settings.sensor = VisionConfig_pb2.SENSOR_COLOR
                sensor_settings.resolution = VisionConfig_pb2.RESOLUTION_640x480
                sensor_settings.frame_rate = VisionConfig_pb2.FRAMERATE_30_FPS
                sensor_settings.bit_rate = VisionConfig_pb2.BITRATE_10_MBPS
                vision_config.SetSensorSettings(sensor_settings, vision_device_id)
                assert False, 'Incorrect Kinova camera sensor settings detected, please restart the camera to apply new settings'

            # Disable autofocus and set manual focus to infinity
            # Note: This must be called after the OpenCV stream is created,
            # otherwise the camera will still have autofocus enabled
            sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
            sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
            sensor_focus_action.manual_focus.value = 0
            vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)

class RealSenseCamera:
    def __init__(self, serial, frame_width=640, frame_height=360, focus=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focus = focus
        self.serial = serial
        self.image = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.running = True
        self.thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.thread.start()

    def camera_worker(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                # Convert BGR to RGB for consistency with other cameras
                self.image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
            time.sleep(0.001)  # Small sleep to avoid hogging CPU

    def get_image(self):
        return self.image

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.pipeline.stop()

def main():
    # ... (other setup code)
    camera = None
    frame_count = 0
    try:
        print("Initializing RealSense camera...")
        camera = RealSenseCamera(DEPTH_CAMERA_SERIAL)
        time.sleep(1)
        print(f"Starting recording. Press Ctrl+C to stop.")
        print(f"Saving images to: {session_dir}")
        while True:
            image = camera.get_image()
            if image is not None:
                bgr_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                frame_path = os.path.join(session_dir, f'frame_{frame_count:06d}.jpg')
                cv.imwrite(frame_path, bgr_image)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Recorded {frame_count} frames...")
            time.sleep(1/30)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        if camera is not None:
            camera.close()
        cv.destroyAllWindows()
        print(f"Recording complete. {frame_count} frames saved to {session_dir}")
