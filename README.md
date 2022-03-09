# FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment

This repository contains FineDiving dataset and PyTorch implementation for Temporal Segmentation Attention (TSA). (CVPR 2022)

## Dataset

### Lexicon
We construct a fine-grained video dataset organized by both semantic and temporal structures, where each structure contains two-level annotations.
- For semantic structure, the action-level labels describe the action types of athletes and the step-level labels depict the sub-action types of consecutive steps in the procedure, where adjacent steps in each action procedure belong to different sub-action types. A combination of sub-action types produces an action type.

- In temporal structure, the action-level labels locate the temporal boundary of a complete action instance performed by an athlete. During this annotation process, we discard all the incomplete action instances and filter out the slow playbacks. The step-level labels are the starting frames of consecutive steps in the action procedure.

### Annotation
Given a raw diving video, the annotator utilizes our defined lexicon to label each action and its procedure. We accomplish two annotation stages from coarse- to fine-grained. The coarse stage is to label the action type for each action instance and its temporal boundary accompanied with the official score. The fine-grained stage is to label the sub-action type for each step in the action procedure and record the starting frame of each step.

### Statistics
The FineDiving dataset consists of 3000 video samples, crossed 52 action types, 29 sub-action types, and 23 difficulty degree types.

### Download
- Untrimmed_Videos: Diving competition videos in Olympics, World Cup, World Championships, and European Aquatics Championships.
- Trimmed_Video_Frames: Trimmed video frames extracted from untrimmed videos and can be directly used for AQA.
- Annotations: Coarse- and fine-grained annotations including the action-level labels for untrimmed videos and the step-level labels for action instances.
- Train/Test Split: 75 percent of samples are for training and 25 percent are for testing.

We have made the full dataset available on [[Baidu Drive]](https://pan.baidu.com/s/1v85-np2FbS0J4UfAEiI4mg) (extract number: 0624) and Google Dirve.

