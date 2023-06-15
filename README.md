# FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment

Created by [Jinglin Xu*](https://xujinglin.github.io/), [Yongming Rao*](https://raoyongming.github.io/), [Xumin Yu](https://yuxumin.github.io/), [Guangyi Chen](https://chengy12.github.io/), Jie Zhou, [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=zh-CN)

This repository contains the FineDiving dataset and PyTorch implementation for Temporal Segmentation Attention. (CVPR 2022, Oral)

[[Project Page]](https://finediving.ivg-research.xyz) [[arXiv]](https://arxiv.org/pdf/2204.03646.pdf)

## Dataset

### Lexicon

We construct a fine-grained video dataset organized by both semantic and temporal structures, where each structure contains two-level annotations.

- For semantic structure, the action-level labels describe the action types of athletes and the step-level labels depict the sub-action types of consecutive steps in the procedure, where adjacent steps in each action procedure belong to different sub-action types. A combination of sub-action types produces an action type.

- In temporal structure, the action-level labels locate the temporal boundary of a complete action instance performed by an athlete. During this annotation process, we discard all the incomplete action instances and filter out the slow playbacks. The step-level labels are the starting frames of consecutive steps in the action procedure.

### Annotation

Given a raw diving video, the annotator utilizes our defined lexicon to label each action and its procedure. We accomplish two annotation stages from coarse- to fine-grained. The coarse-grained stage is to label the action type for each action instance and its temporal boundary accompanied with the official score. The fine-grained stage is to label the sub-action type for each step in the action procedure and record the starting frame of each step, utilizing an effective [Annotation Toolbox](https://github.com/coin-dataset/annotation-tool).

The annotation information is saved in `FineDiving_coarse_annotation.pkl` and `FineDiving_fine-grained_annotation.pkl`.

| Field Name    | Type   | Description                          | Field Name             | Type  | Description                         |
| ------------- | ------ | ------------------------------------ | ---------------------- | ----- | ----------------------------------- |
| `action_type` | string | Description of the action type.      | `sub-action_types`     | dict  | Description of the sub-action type. |
| `(x, y)`      | string | Instance ID.                         | `judge_scores`         | list  | Judge scores.                       |
| `dive_score`  | float  | Diving score of the action instance. | `frames_labels`        | array | Step-level labels of the frames.    |
| `difficulty`  | float  | Difficulty of the action type.       | `steps_transit_frames` | array | Frame index of step transitions.    |
| `start_frame` | int    | Start frame of the action instance.  | `end_frame`            | int   | End frame of the action instance.   |

### Statistics

The FineDiving dataset consists of 3000 video samples, covering 52 action types, 29 sub-action types, and 23 difficulty degree types.

### Download

To download the FineDiving dataset, please sign the [Release Agreement](agreement/Release_Agreement.pdf) and send it to Dr. Xu (xujinglinlove@gmail.com). By sending the application, you are agreeing and acknowledging that you have read and understand the notice. We will reply with the file and the corresponding guidelines right after we receive your request!

- Untrimmed_Videos: Diving competition videos in Olympics, World Cup, World Championships, and European Aquatics Championships.
- Trimmed_Video_Frames: Trimmed video frames are extracted from untrimmed video frames (obtained by [data_process.py](data_preparation/data_process.py) from untrimmed videos) and can be directly used for AQA.
- Annotations: Coarse- and fine-grained annotations including the action-level labels for untrimmed videos and the step-level labels for action instances.
- Train/Test Split: 75 percent of samples are for training and 25 percent are for testing.

After downloading the dataset, put the annotation files (`FineDiving_coarse_annotation.pkl`, `FineDiving_fine-grained_annotation.pkl`, `Sub_action_Types_Table.pkl`, and `fine-grained_annotation_aqa.pkl`) and train_test_split files (`train_split.pkl` and `test_split.pkl`) into the `FineDiving/Annotations/` folder.

## Code for Temporal Segmentation Attention (TSA)

### Requirement

- Python 3.7.9
- Pytorch 1.7.1
- torchvision 0.8.2
- timm 0.3.4
- torch_videovision

```
pip install git+https://github.com/hassony2/torch_videovision
```

### Data Preperation

- The prepared dataset and annotations `fine-grained_annotation_aqa.pkl` will be provided in our reply to your application.
- If you want to prepare the data by yourself, please see [data_process](data_preparation/data_process.py) for some helps. We provide codes for processing the data from a video to frames.

```
video_dir = "./FINADiving"           # the path to untrimmed videos (.mp4)
base_dir = "./FINADiving_jpgs"       # the path to untrimmed video frames (.jpgs)
save_dir = "./FINADiving_jpgs_256"   # the path to resized untrimmed video frames (used in our approach) 
```

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
└──
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

### Training and Evaluation

```
# train a model on FineDiving
bash train.sh TSA FineDiving 0,1

# resume the training process on FineDiving
bash train.sh TSA FineDiving 0,1 --resume

# test a trained model on FineDiving
bash test.sh TSA FineDiving 0,1 ./experiments/TSA/FineDiving/default/last.pth
# last.pth is obtained by train.sh and saved at "experiments/TSA/FineDiving/default/"
```

**Contact:** [xujinglinlove@gmail.com](mailto:xujinglinlove@gmail.com)
