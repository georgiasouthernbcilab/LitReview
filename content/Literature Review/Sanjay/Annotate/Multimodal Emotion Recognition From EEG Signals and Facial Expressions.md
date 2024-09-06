

Datasets:
The study used two standard datasets:
1. DEAP dataset
2. MAHNOB-HCI dataset

Data formats and structure:

EEG data:
- For both datasets, EEG signals from 32 channels were used
- Sampling rate was downsampled to 128 Hz 
- EEG was filtered from 4.0 Hz to 45 Hz
- Eye artifacts were removed

Labels:
- Self-assessments of valence and arousal on a scale of 1-9
- Values <5 considered "low", 5-9 considered "high"
- Binary classification: High Arousal/Low Arousal and High Valence/Low Valence

Video/Photo data:
- Facial expression videos were recorded
- 5 frames per second extracted from videos
- Face detected and aligned using 68 facial landmarks
- Face images cropped

Data segmentation:
- Data divided into 1-second segments 
- DEAP: 2400 samples per subject
- MAHNOB-HCI: 1611 samples per subject

Pre-processing:
- For EEG: Baseline removal by subtracting average of first 3 seconds
- For facial data: Face detection, alignment, and cropping

The datasets contained multimodal data (EEG + facial expressions) with valence/arousal labels, segmented into 1-second chunks for training the deep learning model. The EEG and facial data were temporally aligned.

--------
a sample label for this dataset would look something like this:

{
  "valence": "high",
  "arousal": "low"
}

To explain:

1. Valence: This represents the pleasantness of the emotion. It's labeled as either "high" (positive emotion) or "low" (negative emotion).

2. Arousal: This represents the intensity of the emotion. It's labeled as either "high" (excited/aroused state) or "low" (calm/unaroused state).

The paper mentions that they used a ==binary classification approach==, where values less than 5 on the original 1-9 scale were considered "low", while values between 5 and 9 were considered "high".

So for each 1-second segment of data (containing both EEG and facial expression information), there would be a corresponding label indicating whether the valence and arousal were high or low for that moment.

It's worth noting that in the original datasets (DEAP and MAHNOB-HCI), the labels were actually on a continuous scale from 1-9. The researchers in this study converted these to binary labels for their classification task. If you wanted to represent the original continuous labels, a sample might look like:

{
  "valence": 7.5,
  "arousal": 3.2
}

But for the purposes of this study, these values would be converted to "high" and "low" respectively.

Binary Classification : 

![[Pasted image 20240906231842.png]]

![[Multimodal_Emotion_Recognition_From_EEG_Signals_and_Facial_Expressions.pdf]]