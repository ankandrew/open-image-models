# Changelog

## [0.6.0](https://github.com/ankandrew/open-image-models/compare/v0.5.1...v0.6.0) (2026-07-27)


### Features

* add commonly used methods in BoundingBox (i.e. `from_xywh`, `from_cxcywh`, etc). ([edb6ad6](https://github.com/ankandrew/open-image-models/commit/edb6ad6ef3bec4d9ad4f17241dc8f871a5af08d1))
* add support for RF-DETR, true dynamic batch inference and create detector factory ([94fa426](https://github.com/ankandrew/open-image-models/commit/94fa426160f090939e1cfe6798ab1847f391ee4f))
* make COCO pre-trained RF-DETR obj detectors available ([ba327d7](https://github.com/ankandrew/open-image-models/commit/ba327d7d56be7ec06263f9342771c8cd935ff0a8))
* support label map for `class_labels` ([c2de117](https://github.com/ankandrew/open-image-models/commit/c2de1171a2b6b8e5751eebba375275b2e7c4ddf8))


### Bug Fixes

* add pre-post processing for RF-DETR object detectors ([bae7c06](https://github.com/ankandrew/open-image-models/commit/bae7c061e1fe5530c0e5d2a26d1172dc7dac914a))
* add release please GH workflow and config files ([8cdc970](https://github.com/ankandrew/open-image-models/commit/8cdc9705fc3d8e3a6879555d66c7547c9b043723))
* linting issue in YoloV9ObjectDetector ([81db522](https://github.com/ankandrew/open-image-models/commit/81db5227c9c9c21c6f8a05bae84bbf8aa2dbd166))
* mypy issue in YoloV9ObjectDetector predict ([81e3028](https://github.com/ankandrew/open-image-models/commit/81e302877ed2a25d86dfe3d432c281a3f6d145ba))
* ONNX Runtime lock for Python 3.10 ([f944802](https://github.com/ankandrew/open-image-models/commit/f944802d2ecc6f56520469bf56fe37c960e4635a))
* preserve PlateDetectorModel compatibility ([23011ac](https://github.com/ankandrew/open-image-models/commit/23011ac0add2838f4ffe440c3655a15e1d7cffd5))
* update .gitignore ([fa3ddbb](https://github.com/ankandrew/open-image-models/commit/fa3ddbb8ff4c8ee72f74bf3eca5102e3235ee393))
* update lib versions ([b6a329c](https://github.com/ankandrew/open-image-models/commit/b6a329c02b3e9f9edb065241aeeb27b4d2b02ce9))
* use DetectionModelName for better type hinting in `create_detector` ([c68a10b](https://github.com/ankandrew/open-image-models/commit/c68a10bfc0d03ca419c4753a2530cb67fc4b1933))
