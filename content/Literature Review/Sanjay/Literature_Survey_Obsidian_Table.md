## Paper 1
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | Neuro-GPT: Towards A Foundation Model for EEG                    |
| **Architecture Used**                  | EEG Encoder + GPT                     |
| **Loss Function**                  | reconstruction loss                    |
| **Evaluation Method**                  | LOOCV                    |
| **Pretraining Dataset**                  | EEG Corpus (TUH Corpus)                    |
| **Downstream Task**                  | motor imagery                    |
| **Finetune Dataset**                  | BCIC IV Dataset 2a                    |
| **Results**                  | Encoder-only: 0.643, Encoder+GPT: 0.586, Linear: 0.443 (with pre-training)                    |
| **Year Published**                  | 2023                    |
| **Notes**                  | nan                    |

---

## Paper 2
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | EEGFormer: Towards Transferable and Interpretable Large-Scale EEG Foundation Model                    |
| **Architecture Used**                  | vector-quantized Transformer model + (1D-CNN) for feature extraction                    |
| **Loss Function**                  | reconstruction loss + with a vector quantization loss                     |
| **Evaluation Method**                  | LOOCV and fivefold cross-validation                    |
| **Pretraining Dataset**                  | EEG Corpus (TUH Corpus)                    |
| **Downstream Task**                  | normal vs. abnormal EEG detection,annotations of five different artifacts,annotations of slowing events,annotations of seizure events,neonatal seizure detection                    |
| **Finetune Dataset**                  | TUAB corpus,TUAR corpus,TUSL corpus,TUSZ corpus,Neonate dataset                    |
| **Results**                  | BETA dataset: Accuracy of 70.15%, Sensitivity of 69.86%, Specificity of 75.86% SEED dataset: Accuracy of 91.58%, Sensitivity of 89.14%, Specificity of 92.75%DepEEG dataset: Accuracy of 72.19%, Sensitivity of 77.83%, Specificity of 70.95%                    |
| **Year Published**                  | 2024                    |
| **Notes**                  | The EEG signals are segmented into patches and converted to frequency domain using Fast Fourier Transformation (FFT). These patches are then fed into a vector-quantized Transformer encoder.  For finetuning, the pretrained encoder and decoder weights are used, and task-specific layers are trained with an objective similar to the pretraining loss.                    |

---

## Paper 3
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | BrainBERT: Self-Supervised Representation Learning for Intracranial Recordings                    |
| **Architecture Used**                  | Transformer architecture (similar to BERT )                    |
| **Loss Function**                  | L1 reconstruction loss + a content-aware loss.                     |
| **Evaluation Method**                  | Leave one session out                    |
| **Pretraining Dataset**                  |  Invasive Intracranial field potential recordings                    |
| **Downstream Task**                  | Sentence onset detection Speech vs. non-speech classification Pitch classification Volume classification                    |
| **Finetune Dataset**                  | Held-out data from the pretraining subjects. (7 subjects from 22 total subjects)                    |
| **Results**                  | Sentence onset detection: AUC of 0.82 (STFT) and 0.78 (superlet) Speech vs. non-speech: AUC of 0.93 (STFT) and 0.86 (superlet) Pitch detection: AUC of 0.75 (STFT) and 0.62 (superlet) Volume classification: AUC of 0.83 (STFT) and 0.70 (superlet)                    |
| **Year Published**                  | 2023                    |
| **Notes**                  | The recorded signals are converted to spectrograms using either the Short-Time Fourier Transform (STFT) or the superlet transform, which are then fed into the Transformer encoder.During pretraining, a masking strategy similar to BERT's masked language modeling is used.For finetuning, a binary cross-entropy loss was applied to train task-specific linear classifiers on top of the pretrained representations.                    |

---

## Paper 4
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  |  Brant-2: Foundation Model for Brain Signals                    |
| **Architecture Used**                  | Transformer with multi-feed-forward (multi-FFN) Transformer blocks for encoding.                    |
| **Loss Function**                  | Reconstruction Loss (MSE)+ Forecasting Loss (MSE)                    |
| **Evaluation Method**                  | K-fold cross-validation (5-fold)                    |
| **Pretraining Dataset**                  |  EEG Corpus (TUH Corpus) + SEEG corpus                    |
| **Downstream Task**                  | Motor Imagery Classification,Emotion Recognition,Sleep Stage Classification,Seizure Prediction,Seizure Detection                    |
| **Finetune Dataset**                  | SEED dataset,MAYO, FNUSA, CHB-MIT, Siena., SleepEDF, Haaglanden Medisch Centrum (HMC),Clinical dataset from a hospital                    |
| **Results**                  | nan                    |
| **Year Published**                  | 2024                    |
| **Notes**                  | Input Signals are divided into non-overlapping patches, augmented, and combined with time and frequency domain information using CNNs and normalization layers.For finetuning, task-specific classifiers are trained using binary cross-entropy loss.                    |

---

## Paper 5
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | SaE-GBLS: An Effective Self-Adaptive Evolutionary Optimized Graph-Broad Model for EEG-Based Automatic Epileptic Seizure Detection                    |
| **Architecture Used**                  | Broad Learning System (BLS) + Self-Adaptive Evolutionary Algorithm                    |
| **Loss Function**                  | MSE with graph regularization                    |
| **Evaluation Method**                  | K-fold cross-validation (5-fold)                    |
| **Pretraining Dataset**                  | CHB-MIT (EEG data from Children's Hospital Boston-MIT) Kaggle (Intracranial EEG signals from humans and dogs) Bonn (EEG data from Bonn University, Germany) Private clinical dataset from The Second Affiliated Hospital of Guangzhou University of Chinese Medicine                    |
| **Downstream Task**                  |  Automatic detection of epileptic seizures                    |
| **Finetune Dataset**                  |  Same datasets as used for pretraining                    |
| **Results**                  | CHB-MIT Dataset: Accuracy of 92.36%, precision of 91.36%, sensitivity of 93.57%, and F1-score of 92.45%. Kaggle Dataset: Accuracy of 99.26%, precision of 99.08%, sensitivity of 99.44%, and F1-score of 99.26%. Bonn Dataset: Achieved 100% accuracy, precision, sensitivity, and F1-score. Patient Dataset: Accuracy of 99.81%, precision of 99.63%, sensitivity of 100%, and F1-score of 99.81%.                    |
| **Year Published**                  | 2024                    |
| **Notes**                  | They have feature nodes and enhancement nodes which are optimized using self-adaptive evolutionary algorithms, Input EEG signals were segmented into 2-second epochs with a 0.5-second overlap to capture meaningful patterns.Time-domain features were extracted, including mean, standard deviation, peak-to-peak value, variance, minimum, maximum, and several others.                    |

---

## Paper 6
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | PPi: Pretraining Brain Signal Model for Patient-independent Seizure Detection                    |
| **Architecture Used**                  | CNN + LSTM + Transformer Encoder                    |
| **Loss Function**                  |  Contrastive loss + Reconstruction loss                    |
| **Evaluation Method**                  | LOOCV                    |
| **Pretraining Dataset**                  | EEG Corpus (TUEG) and SEEG Corpus                    |
| **Downstream Task**                  |  Patient-independent seizure detection using SEEG data                    |
| **Finetune Dataset**                  | MAYO FNUSA and clinical SEEG data                    |
| **Results**                  | MAYO Dataset: F2-score improvement by 38.10% over the best-performing baseline model FNUSA Dataset: F2-score improvement by 24.98% over the best-performing baseline model Clinical Dataset: F2-score improvement by 54.93%                    |
| **Year Published**                  | 2023                    |
| **Notes**                  |  Involves two novel self-supervised tasks (channel discrimination and context swapping). Channel background subtraction and brain region enhancement are introduced to address domain shifts Channel Discrimination Task: Helps in preserving unique characteristics of each channel by determining whether two sequences are from the same channel.Context Swapping Task: Enhances the coherence of SEEG data by determining if the context has been swapped with another channel.                    |

---

## Paper 7
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | BISeizuRe: BERT-Inspired Seizure Data Representation to Improve Epilepsy Monitoring                    |
| **Architecture Used**                  | (BENDR) + Transformer Encoder                    |
| **Loss Function**                  | Contrastive loss (pre-training) + Sensitivity-Specificity Weighted Cross-Entropy (SSWCE) loss (fine-tuning)                    |
| **Evaluation Method**                  | LOOCV                    |
| **Pretraining Dataset**                  | EEG Corpus (TUEG)                    |
| **Downstream Task**                  | Seizure detection                    |
| **Finetune Dataset**                  | CHB-MIT Scalp EEG Database                    |
| **Results**                  | 72.58% sensitivity and 0.23 false positives per hour (FP/h)                    |
| **Year Published**                  | 2024                    |
| **Notes**                  |  The EEG signals are segmented into non-overlapping windows of 8 seconds each.wo oversampling methods were tested to address the highly unbalanced nature of EEG data: Synthetic Minority Oversampling Technique (SMOTE) and Weighted Random Sampler (selected as the better performing option)Two aggregation criteria were considered: Majority voting and MinPooling (selected as the better performing option with a window length of 3)                    |

---

## Paper 8
| **Column**              | **Details**                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| **Paper Title**         | TS-MoCo: Time-Series Momentum Contrast for Self-Supervised Physiological Representation Learning |
| **Architecture Used**   | Transformer Encoder + GRU-based reconstruction head                                              |
| **Loss Function**       | Reconstruction + Cosine Similarity Loss                                                          |
| **Evaluation Method**   | Classification Accuracy on60-20-20 split for train, validation, and test sets                    |
| **Pretraining Dataset** | SEED,UCIHAR                                                                                      |
| **Downstream Task**     | Emotion Recognition from EEG Human Activity Recognition from inertial sensory data                    |
| **Finetune Dataset**                  | Same datasets as used for pretraining                    |
| **Results**                  | SEED Dataset: Classification accuracy of 0.43 (TS-MoCo) vs. 0.79 (supervised baseline) and 0.42 (TS-TCC) UCIHAR Dataset: Classification accuracy of 0.52 (TS-MoCo) vs. 0.89 (supervised baseline) and 0.90 (TS-TCC)                    |
| **Year Published**                  | 2023                    |
| **Notes**                  | Utilizes self-supervised learning with momentum contrast to learn representations from time-series data without labels. Combines tokenization, positional embeddings, and a transformer encoder to create context-aware representations.                    |

---

## Paper 9
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | LaBraM: Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI                    |
| **Architecture Used**                  | Transformer Encoder + Temporal Encoder (Specifics : Neural Tokenizer with Vector-Quantized Neural Spectrum Prediction)                    |
| **Loss Function**                  | (MSE) for spectrum reconstruction + Masked token prediction loss                    |
| **Evaluation Method**                  | Train-Validation-Test split (80-20 for training and validation within training set)                    |
| **Pretraining Dataset**                  | BCI Competition IV-1 EmobrainGrasp and Lift EEG Challenge Inria BCI Challenge EEG Motor Movement/Imagery Dataset Resting State EEG Data SEED Series Siena Scalp EEG Database etc                    |
| **Downstream Task**                  | Abnormal Detection (TUAB) Event Type Classification (TUEV) Emotion Recognition Gait Prediction                    |
| **Finetune Dataset**                  | TUAB (Temple University Hospital Abnormal EEG Corpus) TUEV (Temple University Hospital Event EEG Corpus)                    |
| **Results**                  | TUAB Dataset: Balanced Accuracy: 0.8140 (Base), 0.8226 (Large), 0.8258 (Huge) AUROC: 0.9022 (Base), 0.9127 (Large), 0.9162 (Huge) TUEV Dataset: Balanced Accuracy: 0.6409 (Base), 0.6581 (Large), 0.6616 (Huge) Cohen’s Kappa: 0.6637 (Base), 0.6622 (Large), 0.6745 (Huge)                    |
| **Year Published**                  | 2024                    |
| **Notes**                  | EEG signals are segmented into patches using a fixed-length time window, followed by temporal encoding, addition of temporal and spatial embeddings, and then passed through the Transformer encoder.                    |

---

## Paper 10
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  |  Brant-X: A Unified Physiological Signal Alignment Framework                    |
| **Architecture Used**                  | EEG foundation model (Brant-2) + EXG encoder                    |
| **Loss Function**                  | InfoNCE Loss                    |
| **Evaluation Method**                  | training, validation, and test 3:1:1                    |
| **Pretraining Dataset**                  | Brant2 Data + CAP, ISRUC, and HMC datasets, including EEG, EOG, ECG, and EMG signals .                    |
| **Downstream Task**                  | sleep stage classification, emotion recognition, freezing of gaits detection, eye movement communication, and arrhythmia detection.                    |
| **Finetune Dataset**                  | Sleep-EDF-78 and Sleep-EDF-20 ,DREAMER ,FoG etc                    |
| **Results**                  | Brant-X achieves state-of-the-art performance across all evaluated tasks.                    |
| **Year Published**                  | 2024                    |
| **Notes**                  | EXG Encoder uses CNN and Transformer.                    |

---

## Paper 11
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  |  EEG-Deformer: A Dense Convolutional Transformer for Brain-computer Interfaces                    |
| **Architecture Used**                  | EEG-Deformer + HCT block + DIP module                    |
| **Loss Function**                  | Cross-Entropy Loss                    |
| **Evaluation Method**                  | LOOCV                    |
| **Pretraining Dataset**                  | Cognitive attention dataset with EEG signals recorded during the Discrimination/Selection Response (DSR) task. Driving fatigue dataset with EEG signals recorded during a 90-minute VR driving task. Cognitive workload dataset with EEG recordings during mental arithmetic tasks.                    |
| **Downstream Task**                  | Cognitive attention classification Driving fatigue classification Cognitive workload classification                    |
| **Finetune Dataset**                  | Same as pretraining datasets                    |
| **Results**                  | Dataset I: 82.72% accuracy, 82.36% macro-F1 score. Dataset II: 79.32% accuracy, 75.83% macro-F1 score. Dataset III: 73.18% accuracy, 69.99% macro-F1 score.                    |
| **Year Published**                  | 2024                    |
| **Notes**                  | Involves shallow feature extraction, hierarchical coarse-to-fine temporal learning, and dense connection of purified information. Uses a hierarchical and dense information purification approach                    |

---

## Paper 12
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks                    |
| **Architecture Used**                  | DGCNN (Graph CNN)                    |
| **Loss Function**                  | Cross entropy with regularization                    |
| **Evaluation Method**                  | Subject-dependent (leave-one-session-out cross-validation) and subject-independent (leave-one-subject-out cross-validation)                    |
| **Pretraining Dataset**                  | SEED dataset, DREAMER dataset                    |
| **Downstream Task**                  | EEG emotion recognition (valence, arousal, and dominance classification)                    |
| **Finetune Dataset**                  | Same as pretraining datasets                    |
| **Results**                  | Average recognition accuracy of 90.4% for subject-dependent experiments on SEED 79.95% for subject-independent cross-validation on SEED Average accuracies of 86.23%, 84.54%, and 85.02% for valence, arousal, and dominance on DREAMER                    |
| **Year Published**                  | 2020                    |
| **Notes**                  |  The DGCNN method dynamically learns the adjacency matrix, capturing intrinsic connections between EEG channels                    |

---

## Paper 13
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | Channel-Aware Self-Supervised Learning for EEG-based BCI                    |
| **Architecture Used**                  | Encoder                    |
| **Loss Function**                  | Weighted cross entropy                    |
| **Evaluation Method**                  | SleepEDF: Leave-one-session-out cross-validation; CHB-MIT: Patient-by-patient evaluation                    |
| **Pretraining Dataset**                  | SleepEDF-20 and CHB-MIT                    |
| **Downstream Task**                  | sleep staging classification and seizure detection                    |
| **Finetune Dataset**                  | Same as pretraining datasets                    |
| **Results**                  | SleepEDF-20: Accuracy of 93.0%, Kappa of 0.861, Macro F1 score of 77.0%, Sensitivity of 76.6% CHB-MIT: Average accuracy of 96.7%, Sensitivity of 81.9%, AUC of 90.0%                    |
| **Year Published**                  | 2023                    |
| **Notes**                  | The input EEG data is resampled, band-pass filtered, and scaled before being processed through two self-supervised learning pathways: spectral and temporal. The spectral pathway uses temporal convolutions to extract frequency-related features, while the temporal pathway uses similar convolutions and 1x1 layers to capture temporal characteristics. These features are integrated using spatial convolutions, pooled, and normalized through adaptive layer normalization. Finally, the normalized features are used for classification tasks like sleep staging and seizure detection,                    |

---

## Paper 14
| **Column**                 | **Details**                    |
|----------------------------|--------------------------------|
| **Paper Title**                  | CLEP: Contrastive Learning for Epileptic Seizure Prediction Using a Spatio-Temporal-Spectral Network                    |
| **Architecture Used**                  |  STS-Net ,Pyramid Convolution Net, Triple Attention Fusion Net, and Spatio Dynamic Graph Convolution Network (sdGCN)                    |
| **Loss Function**                  | EEG Contrastive (EC) loss + cross-entropy loss                    |
| **Evaluation Method**                  | Patient-specific LOOCV                    |
| **Pretraining Dataset**                  | CHB-MIT scalp EEG database                    |
| **Downstream Task**                  | Epileptic seizure prediction                    |
| **Finetune Dataset**                  |  CHB-MIT scalp EEG database and Xuanwu intracranial EEG (iEEG) database                    |
| **Results**                  | CHB-MIT dataset: Sensitivity of 96.7%, False Prediction Rate of 0.072/h Xuanwu dataset: Sensitivity of 95%, False Prediction Rate of 0.087/h                    |
| **Year Published**                  | 2023                    |
| **Notes**                  | Raw EEG signals are processed to extract preictal and interictal periods. The EEG signals are segmented into 5-second clips.The CLEP strategy pretrains the EEG encoder using contrastive learning on source subjects. The model learns to maximize the similarity of intra-class pairs (preictal-preictal and interictal-interictal) and minimize the similarity of inter-class pairs (preictal-interictal).a triple attention layer and dynamic graph convolution is used to handle inter-channel and spatial dependencies                    |

---

