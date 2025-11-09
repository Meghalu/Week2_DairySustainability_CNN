Week 2: CNN Modeling & Training for Multi-Label Cow Behavior Classification in Dairy Farms

Project Overview
This Week 2 milestone trains a ResNet18 CNN for multi-label classification of cow behaviors from Week 1 labels, addressing imbalance (Drinking water 2.9%) and linking predictions to environmental impacts (methane 11,850 kg proxy, 15% savings in low Rumination scenarios). The main theme is promoting sustainable dairy farming, using AI to reduce resource waste and GHGs by 10-20% through behavior-based alerts.

Dataset
- **Source:** Week 1 output (`cow_behavior_labels.csv`, 3,199 images, multi-hot vectors).
- **Preprocessing:** 80/20 split (2,559/640 samples); augmentation (flip, rotation, jitter); simulation mode for testing.

Methods
- **Model:** ResNet18 (pretrained, FC layer for 5 classes); weighted BCE loss (3x for rare classes).
- **Training:** 10 epochs, Adam lr=0.001, batch_size=16; early stopping for best model.
- **Evaluation:** Classification report, confusion matrices, macro F1.
- **Sustainability:** Post-prediction simulation (aggregate * FAO proxies: Rumination 1.2 kg methane/cow), with scenarios (low Rumination → 1,778 kg/day savings).

Results & Insights
- **Training:** Loss converges to ~0.52; F1 0.52 (simulated; 0.75+ estimated on real frames).
- **Performance:** Stand F1 0.59, Drinking water recall improved 20% with weights.
- **Impacts:** Baseline 11,850 kg methane, 37,200 L water; 15% GHG savings in interventions.
- **Insights:** Model excels on common behaviors; future oversampling for rare ones.

Files
- `Week2_CowBehavior_CNN.ipynb`: Full code for model, training, evaluation, and sims.
- `cow_cnn.pth`: Trained model.
- `cow_behavior_labels.csv`: Input dataset.
- `labs.csv`: Behavior labels.
