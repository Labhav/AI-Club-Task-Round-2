# AI-Club-Task-Round-2
Name: Labhav Agarwal
Id: 2025B3PS1033P
Model Performance
* *Accuracy:* 59%
* *F1-Score:* 0.60
* *Bias Check:* The model shows balanced performance across Male and Female speakers (Difference < 5%).
## 🛠️ Training Details
*Architecture:* 2D CNN with 4 Convolutional Blocks + Global Average Pooling.
*Augmentations:* Noise Injection (Random Normal)
    * Time Shifting & Zooming
*Input Features:* Log-Mel Spectrograms (128x130).
*Optimizer:* Adam (Learning Rate: 0.001).
