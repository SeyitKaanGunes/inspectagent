# Sample images - attribution

The images and masks in this folder are a small curated subset of the **VisA (Visual
Anomaly) dataset** by Amazon, released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

* Dataset: VisA, `VisA_20220922.tar` - <https://github.com/amazon-science/spot-diff>
* Paper: Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer. *SPot-the-Difference
  Self-Supervised Pre-training for Anomaly Detection and Segmentation.* ECCV 2022.
* Copyright (c) Amazon.com, Inc. or its affiliates; licensed under CC BY 4.0.

Changes made by InspectAgent: images were selected from the held-out evaluation split
(`data/splits.json`), re-encoded as JPEG (quality 92) at full resolution, and the
ground-truth masks were binarized to 0/255 PNG. File names were changed to
`<category>_<normal|anomaly>_<original stem>`. The original VisA path of every file is
recorded in `samples.json` (`source`).

The same attribution applies to the extracted dataset under `data/visa/` (not
redistributed with this repository) and to any figure or trace derived from it.
