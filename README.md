# FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment

This repository contains FineDiving dataset and PyTorch implementation for Temporal Segmentation Attention (TSA). (CVPR 2022)

## Dataset

### Lexicon
We construct a fine-grained video dataset organized by both semantic and temporal structures, where each structure contains two-level annotations.
- For semantic structure, the action-level labels describe the action types of athletes and the step-level labels depict the sub-action types of consecutive steps in the procedure, where adjacent steps in each action procedure belong to different sub-action types. A combination of sub-action types produces an action type. For instance, for an action type ``5255B'', the steps belonging to the sub-action types ``Back'', ``2.5 Somersaults Pike'', and ``2.5 Twists'' are executed sequentially.

- In temporal structure, the action-level labels locate the temporal boundary of a complete action instance performed by an athlete. During this annotation process, we discard all the incomplete action instances and filter out the slow playbacks. The step-level labels are the starting frames of consecutive steps in the action procedure. For example, for an action belonging to the type ``5152B'', the starting frames of consecutive steps are 18930, 18943, 18957, 18967, and 18978, respectively. 

