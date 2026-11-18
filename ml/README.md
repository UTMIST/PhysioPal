# ML template (parked)

The original [UTMIST ML Seed](https://github.com/UTMIST/utmist-ml-seed) training
code (Hydra configs, ResNet/CIFAR-10, YOLOv8), kept for when PhysioPal needs a
trained model (e.g. a learned form classifier). Not used by the pose pipeline.

```bash
uv pip install -e "..[ml]"   # from this folder
cd ml
python src/train.py
python -m pytest tests -v
```

Docs: [docs/](docs/). Note `opencv-python` (pulled by ultralytics) and
`opencv-contrib-python` (pulled by mediapipe) can conflict; if `import cv2`
breaks, uninstall both and reinstall `opencv-contrib-python`.
