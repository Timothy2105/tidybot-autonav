# Autonomous Navigation for TidyBot: Text-Based Object Querying in Reconstructed Point Cloud Environments

An end-to-end pipeline for **environment mapping, relocalization, calibration, and autonomous robot control** using Intel RealSense, Mast3r-SLAM, and object detection.  

## Project Presentation

- 🎥 **Watch the presentation on this project!**  
  [View Presentation](https://youtu.be/d9sZ_n-h4RY)

- 📝 **View the slides directly**  
  [Open Slides](https://docs.google.com/presentation/d/1r2Wjr1D6ob6gD2j3K7_vl23RylHqccIKcDmurCBtggI/edit?usp=sharing)

## Quick Setup
### Prereqs

- Intel RealSense Viewer (for .bag recording)

- ffmpeg (for .bag → .mp4 conversion)

- Python environment with project dependencies installed

- Mast3r-SLAM models/checkpoints as required by your config

### Record & Convert

- Record with RealSense Viewer → save as .bag.

- Convert .bag → .mp4 (follow your bag→mp4 guide).

## Offline Processing
1) Run Mast3r-SLAM and save state
```bash
python main.py --dataset <path/to/video>.mp4 \
  --config config/base.yaml \
  --save-as <scene-name> \
  --skip-frames 15 \
  --save-state saved-states/<scene-name>
```

2) Process keyframes
```bash
python process_kf.py --dir saved-states/<scene-name>
```

4) Manual calibration (viser)
```bash
python interactive_robot_interface.py \
  --saved-folder saved-states/<scene-name> \
  --save-transformed-ply --save
```

> Tip: When aligning in viser, align the axes with the orbit, not just what the point cloud looks like.

## Live Run (4 Terminals)

Terminal 1 — SLAM (with relocalization):
```bash
python main.py --dataset realsense \
  --config config/base.yaml \
  --skip-frames 120 \
  --load-state saved-states/<scene-name>
```

Terminal 2 — After relocalized (apply/save transforms):
```bash
python interactive_robot_interface.py \
  --saved-folder saved-states/<scene-name> \
  --save-transformed-ply --save
```

Terminal 3 — Send robot commands:
```bash
python send_cmd.py
```

Terminal 4 — Object detection + monitoring:
```bash
python object-detection/gpt.py \
  --dir saved-states/<scene-name> \
  --save-annotated --monitor
```

> `--monitor` lets you query for the object sent by `hand_eye_calib_viser.py`.

## Notes

- Use `--save-state` / `--load-state` to store and upload SLAM sessions.

- If relocalization succeeds once but fails after, re-check keyframe coverage and the loaded state fidelity.

- Keep wheel odometry and camera frame transforms calibrated; verify axis conventions after each map update.
