# FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment

This repository contains the FineDiving dataset and PyTorch implementation for Temporal Segmentation Attention. (CVPR 2022)

[[Project Page]](Coming soon) [[arXiv]](Coming soon) [[Dataset]](https://pan.baidu.com/s/1v85-np2FbS0J4UfAEiI4mg)

## Dataset

### Lexicon
We construct a fine-grained video dataset organized by both semantic and temporal structures, where each structure contains two-level annotations.
- For semantic structure, the action-level labels describe the action types of athletes and the step-level labels depict the sub-action types of consecutive steps in the procedure, where adjacent steps in each action procedure belong to different sub-action types. A combination of sub-action types produces an action type.

- In temporal structure, the action-level labels locate the temporal boundary of a complete action instance performed by an athlete. During this annotation process, we discard all the incomplete action instances and filter out the slow playbacks. The step-level labels are the starting frames of consecutive steps in the action procedure.

### Annotation
Given a raw diving video, the annotator utilizes our defined lexicon to label each action and its procedure. We accomplish two annotation stages from coarse- to fine-grained. The coarse stage is to label the action type for each action instance and its temporal boundary accompanied with the official score. The fine-grained stage is to label the sub-action type for each step in the action procedure and record the starting frame of each step, utilizing an effective [Annotation Toolbox](https://github.com/coin-dataset/annotation-tool).

The annotation information is saved in [`FineDiving_coarse_annotation.pkl`](Annotations/FineDiving_coarse_annotation.pkl) and [`FineDiving_fine-grained_annotation.pkl`](Annotations/FineDiving_fine-grained_annotation.pkl).

| Field Name          | Type         | Description                                                                                                           |
| ------------------- | ----------------------------| --------------------------------------------------------------------------------------------------------------------- |
| `action_type`          | string                     | Description of the action type.                                                                                              |
| `(x, y)`                  | string                  | Instance ID.                                                                                              |
| `dive_score`          | float                       | Diving score of the action instance.                                                        |
| `difficulty`             | float                    | Difficulty of the action type.                                                                           |
| `start_frame`       | int                         | Start frame of the action instance. |
| `end_frame`        | int                         | End frame of the action instance.  |
| `judge_scores`    | list | Judge scores.                                                                                                |
| `steps_transit_frames`    | array     | Frame index of step transitions.                                                                             |
| `frames_labels`              | array             | Step-level labels of the frames.                                                                                       |
| `sub-action_types`              | dict           | Description of the sub-action type.                                                                                 |

### Statistics
The FineDiving dataset consists of 3000 video samples, crossed 52 action types, 29 sub-action types, and 23 difficulty degree types.

### Download
- Untrimmed_Videos: Diving competition videos in Olympics, World Cup, World Championships, and European Aquatics Championships.
- Trimmed_Video_Frames: Trimmed video frames extracted from untrimmed videos and can be directly used for AQA. [[Baidu Drive]](https://pan.baidu.com/s/1v85-np2FbS0J4UfAEiI4mg) or [[Google Drive]](https://drive.google.com/drive/folders/1KIsocnoL1fSkogljdtOswEljYjFhrtgj?usp=sharing)
- Annotations: Coarse- and fine-grained annotations including the action-level labels for untrimmed videos and the step-level labels for action instances. [`FineDiving_coarse_annotation.pkl`](Annotations/FineDiving_coarse_annotation.pkl) [`FineDiving_fine-grained_annotation.pkl`](Annotations/FineDiving_fine-grained_annotation.pkl)
- Train/Test Split: 75 percent of samples are for training and 25 percent are for testing. [train_split.pkl](Annotations/train_split.pkl) [test_split.pkl](Annotations/test_split.pkl)

We have made the full dataset available on [[Baidu Drive]](https://pan.baidu.com/s/1v85-np2FbS0J4UfAEiI4mg) (extract number: 0624).

## Code
### Requirement
- Python 3.7.9
- Pytorch 1.7.1
- torchvision 0.8.2
- timm 0.3.4
- torch_videovision
```
pip install git+https://github.com/hassony2/torch_videovision
```

### The FineDiving Dataset for AQA
- The data structure should be:
```
$DATASET_ROOT
├── FineDiving
|  ├── FINADivingWorldCup2021_Men3m_final_r1
|     ├── 0
|        ├── 00489.jpg
|        ...
|        └── 00592.jpg
|     ...
|     └── 11
|        ├── 14425.jpg
|        ...
|        └── 14542.jpg
|  ...
|  └── FullMenSynchronised10mPlatform_Tokyo2020Replays_2
|     ├── 0
|     ...
|     └── 16
|        
├── fine-grained_annotation_for_aqa.pkl
├── train_split.pkl
└── test_split.pkl
```

### Pretrain Model
The Kinetics pretrained I3D downloaded from the reposity [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth)
```
model_rgb.pth
```

### Experimental Setting
```
FineDiving_TSA.yaml
```

### Training and Testing
```
bash train_test.sh TSA FineDiving 0,1
```

**Contact:** [xujinglinlove@gmail.com](mailto:xujinglinlove@gmail.com)
