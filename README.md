# BanglaSignNet: Sequential word level Bangla Sign Language Recognition using Spatio-Temporal Transformers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-1,922%20samples-orange)

A novel deep learning system for continuous Bangla Sign Language (BdSL) recognition at the word level, using a sequence-to-sequence Transformer architecture with pose-based spatio-temporal features.

## 🎯 Project Overview

This project introduces a groundbreaking approach to Bangla Sign Language recognition by treating it as a **sequence-to-sequence translation problem** rather than traditional image classification. The system converts continuous sign language videos into sequences of Bangla glosses (word labels) using a Transformer model with pose keypoint features.

### Key Innovations:
- **🔄 Sequence-to-Sequence Architecture**: Models continuous signing as video-to-gloss translation
- **💡 Pose-Based Features**: Uses MediaPipe holistic pose landmarks for data efficiency
- **⚡ Temporal Modeling**: Transformer architecture captures rich temporal dynamics
- **🌐 Bangla-Focused**: Specifically designed for Bangla Sign Language characteristics

## 📊 Dataset

We use the **Ban-Sign-Sent-9K-V1** dataset with the following characteristics:

- **1,922 annotated sentences** with continuous signing
- **39-word gloss vocabulary** (compact and focused)
- **5.18 average words per gloss sequence**
- **Multiple signers** for better generalization
- **Dual annotations**: Natural Bangla sentences + Gloss sequences

### Sample Data:
```
Natural: "বাবা সকালে চা খান না।"
Gloss:   "বাবা সকাল চা না।"
```

## 🏗️ Architecture

### Model Components:

1. **Pose Encoder**:
   - Input: 375-dimensional pose features (33 pose + 21 hand + 21 hand + 50 face landmarks)
   - Transformer Encoder with positional encoding
   - Output: Encoded spatio-temporal representation

2. **Gloss Decoder**:
   - Transformer Decoder with attention mechanism
   - Input: Previous gloss tokens + encoder memory
   - Output: Sequence of Bangla word glosses

3. **Sequence-to-Sequence Pipeline**:
   ```
   Video Input → Pose Extraction → Transformer Encoder → Transformer Decoder → Gloss Sequence
   ```

### Technical Specifications:
- **Encoder Layers**: 4
- **Decoder Layers**: 4  
- **Model Dimensions**: 256
- **Attention Heads**: 8
- **Vocabulary Size**: 39
- **Parameters**: ~11.7 million

## 📈 Performance

### Current Results:
- **Overall Accuracy**: 48-57% across different test sizes
- **2-word sequences**: ~95% accuracy
- **4-word sequences**: ~57% accuracy
- **Training Time**: ~44 minutes for 58 epochs

### Performance by Test Size:
| Samples | Accuracy |
|---------|----------|
| 50      | 52.00%   |
| 100     | 57.00%   |
| 200     | 57.50%   |
| 300     | 54.33%   |
| 500     | 50.40%   |
| 1000    | 48.00%   |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
MediaPipe
OpenCV
```

### Installation
```bash
# Clone repository
git clone https://github.com/Nafi14078/BanglaSignNet.git
cd BanglaSignNet

# Create virtual environment
python -m venv bdsl_env
source bdsl_env/bin/activate  # Windows: bdsl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Project Structure
```
BanglaSignNet/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Processed pose sequences
│   └── annotations/            # Dataset annotations & vocabulary
├── src/
│   ├── data_processing/        # Pose extraction & dataset creation
│   ├── modeling/               # Transformer model architecture
│   ├── training/               # Training pipelines
│   └── evaluation/             # Testing & evaluation scripts
├── models/                     # Trained model checkpoints
└── results/                    # Evaluation results
```

### Usage

1. **Data Preparation**:
```bash
# Process dataset and extract pose features
python src/data_processing/process_excel_metadata.py
python src/data_processing/pose_extraction_pipeline.py
```

2. **Training**:
```bash
# Train the model
python src/training/full_word_level_trainer.py
```

3. **Evaluation**:
```bash
# Test on 300 samples
python src/evaluation/test_model.py

# Batch testing on different sample sizes
python src/evaluation/batch_test.py
```

4. **Inference**:
```python
from src.modeling.transformer_model import BanglaSignTransformer
from src.data_processing.gloss_dataset import BanglaSignGlossDataset

# Load trained model
model = BanglaSignTransformer(...)
model.load_state_dict(torch.load("models/best_model.pth"))

# Predict on new video
gloss_sequence = model.predict(video_pose_sequence)
```

## 🛠️ Development

### Key Features Implemented:
- ✅ Continuous sequence recognition
- ✅ Pose-based feature extraction
- ✅ Transformer architecture
- ✅ Teacher forcing training
- ✅ Greedy decoding inference
- ✅ Comprehensive evaluation
- ✅ Error analysis tools

### Current Limitations & Future Work:
- 🔄 Improving generalization (current focus)
- 🔄 Handling longer sequences
- 🔄 Real-time inference
- 🔄 Expanded vocabulary support
- 🔄 Multi-modal features (RGB + Pose)

## 📝 Publications & Citation

If you use this work in your research, please cite:

```bibtex
@article{banglasignnet2024,
  title={BanglaSignNet: Sequential Word level Bangla Sign Language Recognition using Spatio-Temporal Transformers},
  author={Ashfak Azad Nafi},
  journal={GitHub Repository},
  year={2024},
  url={https://github.com/Nafi14078/BanglaSignNet}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

### Areas for Contribution:
- Model architecture improvements
- Data augmentation techniques
- Performance optimization
- Additional evaluation metrics
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Bangladesh Government for the Ban-Sign-Sent-9K-V1 dataset
- MediaPipe team for pose estimation capabilities
- PyTorch team for the deep learning framework
- The open-source community for various utilities

## 📞 Contact

For questions, suggestions, or collaborations:
- GitHub Issues: [Create an issue](https://github.com/yourusername/BanglaSignNet/issues)
- Email: your.email@domain.com

---

**⭐ If you find this project useful, please give it a star on GitHub!**
