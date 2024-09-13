In the paper "DeepVANet: A Deep End-to-End Network for Multi-modal Emotion Recognition," the emotional states predicted by the model are based on the valence and arousal dimensions, but the study does not specifically categorize them into discrete emotional states like joy, anger, or sadness.

Instead, the model predicts whether the valence and arousal are "High" or "Low," effectively classifying emotions into four possible states based on the combination of these two binary dimensions:

1. **High Valence, High Arousal**
2. **High Valence, Low Arousal**
3. **Low Valence, High Arousal**
4. **Low Valence, Low Arousal**

The model's focus is on predicting these valence and arousal levels rather than directly mapping these predictions to named emotional states like joy or sadness. The classification is continuous within the valence-arousal space, allowing for a broad and flexible representation of emotions without explicitly defining them as specific types. Therefore, while the model captures these dimensional predictions, it does not explicitly predict discrete emotional states directly named in the study.

==the model can classify emotions into four discrete categories:==
==there is no emotion is classified or recognized in the wheel slice ==

Deap Dataset training structure : 
```bash 
/dataset
    /DEAP
        /subject_01
            /videos
                - video_01.mp4
                - video_02.mp4
                ...
            /bio_signals
                - bio_01.csv
                - bio_02.csv
                ...
        /subject_02
            /videos
                - video_01.mp4
                - video_02.mp4
                ...
            /bio_signals
                - bio_01.csv
                - bio_02.csv
                ...
        ...
    /MAHNOB-HCI
        /subject_01
            /videos
                - video_01.mp4
                - video_02.mp4
                ...
            /bio_signals
                - bio_01.csv
                - bio_02.csv
                ...
        /subject_02
            /videos
                - video_01.mp4
                - video_02.mp4
                ...
            /bio_signals
                - bio_01.csv
                - bio_02.csv
                ...
        ...
    /labels
        /DEAP
            - labels.csv  # Contains valence/arousal scores or binary labels (High/Low) per video segment.
        /MAHNOB-HCI
            - labels.csv  # Contains valence/arousal scores or binary labels (High/Low) per video segment.

```



![[DeepVANet - humancomputer-interaction - interact-2021-2021-254-262.pdf]]