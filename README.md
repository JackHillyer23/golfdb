# Golf Swing Event Detection

A comparative study of CNN-LSTM architectures for automated golf swing 
event detection using the GolfDB dataset. This project extends the original 
SwingNet implementation by McNally et al. by investigating MobileNetV3-Large 
and EfficientNet-B0 as alternative backbone architectures.

## Project Structure
```
golfdb/
├── model.py                    # Original MobileNetV2 model (authors)
├── model_v2_torchvision.py     # MobileNetV2 using torchvision
├── model_v3.py                 # MobileNetV3-Large implementation
├── model_efficientnet.py       # EfficientNet-B0 implementation
├── train.py                    # Original training script (authors)
├── train_v2_scratch.py         # MobileNetV2 training from scratch
├── train_v3.py                 # MobileNetV3-Large training
├── train_efficientnet.py       # EfficientNet-B0 training
├── train_efficientnet_continued.py  # EfficientNet-B0 extended training
├── eval.py                     # Original evaluation script (authors)
├── eval_v3.py                  # MobileNetV3-Large evaluation
├── eval_efficientnet.py        # EfficientNet-B0 evaluation
├── eval_v2_scratch.py          # MobileNetV2 scratch evaluation
├── eval_v2_allsplits.py        # MobileNetV2 all splits evaluation
├── eval_v3_allsplits.py        # MobileNetV3-Large all splits evaluation
├── eval_efficientnet_allsplits.py  # EfficientNet-B0 all splits evaluation
├── dataloader.py               # Dataset loading and preprocessing
├── util.py                     # Utility functions including PCE calculation
├── test_video.py               # Video inference script (authors V2)
├── test_video_efficientnet.py  # Video inference script (EfficientNet)
├── generate_graphs.py          # PCE comparison and heatmap generation
├── generate_allsplits_graphs.py # All splits PCE graph generation
├── check_model.py              # Model architecture inspection utility
├── models/                     # Saved model checkpoints
│   ├── swingnet_1800.pth.tar  # Authors pretrained weights
│   ├── swingnet_v3_.pth.tar  # MobileNetV3 checkpoints
│   └── swingnet_efficientnet_.pth.tar  # EfficientNet checkpoints
└── data/
├── videos_160/             # Preprocessed 160x160 video clips
├── train_split_.pkl       # Training split annotations
└── val_split_.pkl         # Validation split annotations
```

## Requirements

```bash
conda create -n golf_swing python=3.9
conda activate golf_swing
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python pandas matplotlib numpy
```

## Results Summary

| Model | Split 1 PCE | Avg PCE (all splits) |
|---|---|---|
| MobileNetV2 (Authors) | 0.715 | 0.813 |
| MobileNetV2 (Scratch) | 0.612 | 0.641 |
| MobileNetV3-Large | 0.661 | 0.650 |
| EfficientNet-B0 (Optimised) | 0.719 | 0.754 |

## Training

**Train MobileNetV2 from scratch:**
```bash
python train_v2_scratch.py
```

**Train MobileNetV3-Large:**
```bash
python train_v3.py
```

**Train EfficientNet-B0:**
```bash
python train_efficientnet.py
```

**Continue EfficientNet-B0 training (optimisation stage):**
```bash
python train_efficientnet_continued.py
```

## Evaluation

**Evaluate on split 1:**
```bash
python eval_efficientnet.py
python eval_v3.py
python eval_v2_scratch.py
```

**Evaluate across all 4 splits:**
```bash
python eval_efficientnet_allsplits.py
python eval_v3_allsplits.py
python eval_v2_scratch_allsplits.py
python eval_baseline_allsplits.py
```

## Video Inference

**Run inference on a video using authors baseline:**
```bash
python test_video.py -p your_video.mp4
```

**Run inference using optimised EfficientNet-B0:**
```bash
python test_video_efficientnet.py -p your_video.mp4
```

Note: Input video should be filmed face-on in landscape orientation 
for best results.

## Generating Graphs

**Generate PCE comparison and heatmap graphs:**
```bash
python generate_graphs.py
```

**Generate all splits PCE graphs:**
```bash
python generate_allsplits_graphs.py
```

## Dataset

The GolfDB dataset is required to run training and evaluation. 
Place preprocessed 160x160 video clips in data/videos_160/ and 
annotation pickle files in data/. 

Dataset available at: https://github.com/wmcnally/golfdb

## Pretrained Weights

The authors' pretrained MobileNetV2 weights (swingnet_1800.pth.tar) 
should be placed in the models/ folder.

Available at: https://github.com/wmcnally/golfdb

## Hardware Notes

GPU training requires PyTorch with CUDA support. Note that the 
NVIDIA RTX 5060 (CUDA capability sm_120) is not supported by 
stable PyTorch builds at the time of writing. All training in 
this project was conducted on CPU.

## References

McNally W., Vats K., Pinto T., Dulhanty C., McPhee J., Wong A. 
"GolfDB: A Video Database for Golf Swing Sequencing." 
CVPR Workshops, 2019.

## Requirements

```bash
conda create -n golf_swing python=3.9
conda activate golf_swing
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python pandas matplotlib numpy
```

## Results Summary

| Model | Split 1 PCE | Avg PCE (all splits) |
|---|---|---|
| MobileNetV2 (Authors) | 0.715 | 0.813 |
| MobileNetV2 (Scratch) | 0.612 | 0.641 |
| MobileNetV3-Large | 0.661 | 0.650 |
| EfficientNet-B0 (Optimised) | 0.719 | 0.754 |

## Training

**Train MobileNetV2 from scratch:**
```bash
python train_v2_scratch.py
```

**Train MobileNetV3-Large:**
```bash
python train_v3.py
```

**Train EfficientNet-B0:**
```bash
python train_efficientnet.py
```

**Continue EfficientNet-B0 training (optimisation stage):**
```bash
python train_efficientnet_continued.py
```

## Evaluation

**Evaluate on split 1:**
```bash
python eval_efficientnet.py
python eval_v3.py
python eval_v2_scratch.py
```

**Evaluate across all 4 splits:**
```bash
python eval_efficientnet_allsplits.py
python eval_v3_allsplits.py
python eval_v2_scratch_allsplits.py
python eval_baseline_allsplits.py
```

## Video Inference

**Run inference on a video using authors baseline:**
```bash
python test_video.py -p your_video.mp4
```

Note: Input video should be filmed from behind the ball and golfer in landscape orientation 
for best results.

## Generating Graphs

**Generate PCE comparison and heatmap graphs:**
```bash
python generate_graphs.py
```

**Generate all splits PCE graphs:**
```bash
python generate_allsplits_graphs.py
```

## Dataset

The GolfDB dataset is required to run training and evaluation. 
Place preprocessed 160x160 video clips in data/videos_160/ and 
annotation pickle files in data/. 

Dataset available at: https://github.com/wmcnally/golfdb

## Pretrained Weights

The authors' pretrained MobileNetV2 weights (swingnet_1800.pth.tar) 
should be placed in the models/ folder.

Available at: https://github.com/wmcnally/golfdb

## Hardware Notes

GPU training requires PyTorch with CUDA support. Note that the 
NVIDIA RTX 5060 (CUDA capability sm_120) is not supported by 
stable PyTorch builds at the time of writing. All training in 
this project was conducted on CPU.

## References
