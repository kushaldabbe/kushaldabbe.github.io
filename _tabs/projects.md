---
# the default layout is 'page'
icon: fas fa-code
order: 5
---

Selected projects I've built, each with the stack and the hard parts.
Write-ups live on the [blog]({{ '/archives/' | relative_url }}); code lives on [GitHub](https://github.com/kushaldabbe).

## [autolabel3d](https://github.com/kushaldabbe/autolabel3d)

*Python · PyTorch · Grounding DINO · SAM 2 · Depth Anything V2 · ONNX Runtime*

- Open-vocabulary **3D auto-labeling pipeline** for autonomous-vehicle perception: generates KITTI-format 3D bounding boxes from raw dashcam video or nuScenes scenes, with no manual annotation.
- Four-stage architecture: Grounding DINO for open-vocabulary 2D detection, SAM 2 for temporal mask propagation, Depth Anything V2 for monocular depth, and PCA-based 3D box fitting with metric-scale recovery.
- Exported the depth model to ONNX with full graph optimization (constant folding, operator fusion, shape inference), achieving a **1.5-3x inference speedup** on CoreML and CUDA.

## [Shadow Network](https://github.com/kushaldabbe/shadow-network)

*Python · FastAPI · React.js · Mistral AI · ElevenLabs*

- AI-powered **Cold War spy-agency simulator**, built at the Mistral AI Worldwide Hackathon 2026 (Tokyo). The player directs a covert intelligence network across Tehran, Kyiv, Hong Kong.
- Multi-agent architecture where each operative runs **private autonomous reasoning** (hidden from the player and the orchestrator), simulating real intelligence fog-of-war: secret compliance, deception, defection.
- A rogue engine autonomously triggers defections from dynamic loyalty scores; ElevenLabs delivers real-time voice transmissions per operative.

## [FootyVision](https://github.com/kushaldabbe/footyvision)

*Python · YOLOv8 · BoT-SORT · OpenCV · scikit-learn*

- End-to-end **football vision analytics pipeline**: broadcast footage in, track-level player intelligence, team-aware tactical views, and possession statistics out.
- Fine-tuned YOLOv8s (4 classes) to **mAP50 = 0.807** at ~53 FPS; BoT-SORT with global motion compensation cut player ID fragmentation from ~630 to **85 unique IDs**.
- Automatic team classification via HSV histograms + KMeans; pitch-keypoints pose model + RANSAC homography maps players into pitch coordinates for a synchronized top-down radar.

## [PitchTwin]({{ '/posts/pitchtwin/' | relative_url }})

*Python · BoT-SORT + ReID · homography · jersey OCR · Three.js / WebGL*

- Real-time **3D digital twin of a football match**, reconstructed from plain 2D broadcast video: tactical top-down view, free-orbit camera, and a first-person view from any player.
- Own CV pipeline for tracking and calibration; no proprietary league data required.

{: .prompt-tip }
>
> More on the way: LLM inference benchmarks, fine-tuning experiments, and serving-infrastructure notes.
>
