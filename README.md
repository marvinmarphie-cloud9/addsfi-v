# ADDFS — Automated Deepfake Detection System

ADDFS is an automated deepfake detection system designed to analyze video content and classify it as either **likely authentic** or **likely deepfake**.

The system combines video preprocessing, facial extraction, deep-learning inference, FastAPI services, a browser-based dashboard, and downloadable PDF analysis reports.

The current implementation uses **EfficientNet-B0** as the binary classification model and was trained using data derived from the **FaceForensics++** dataset.

---

## Project Objectives

The project was developed to provide an end-to-end deepfake detection workflow capable of:

- Preparing real and manipulated video datasets
- Extracting video frames
- Detecting and cropping faces
- Training a deep-learning classifier
- Evaluating model performance
- Performing image and video inference
- Providing predictions through a FastAPI backend
- Supporting browser-based video uploads
- Displaying prediction confidence and analysis statistics
- Maintaining recent browser-side analysis history
- Generating downloadable PDF analysis reports

---

## Model Architecture

The detection model uses:

- **Architecture:** EfficientNet-B0
- **Framework:** PyTorch
- **Task:** Binary classification
- **Classes:**
  - `0_real`
  - `1_fake`
- **Input:** Cropped facial images
- **Decision threshold:** 0.50

The final EfficientNet classifier layer is configured for two output classes.

---

## Dataset

The project uses **FaceForensics++** video data.

### Real Sources

Real videos were obtained from:

- YouTube original sequences
- Actor original sequences

### Manipulated Sources

The multi-manipulation training pipeline supports:

- Deepfakes
- Face2Face
- FaceSwap
- NeuralTextures
- FaceShifter
- DeepFakeDetection

The multi-method dataset builder performs source-video-level splitting before frame extraction to reduce the possibility of train/validation leakage.

---

## Dataset Preparation

The multi-manipulation dataset preparation pipeline used approximately:

- **Real source videos:** 1,363
- **Fake source videos:** 1,362
- **Fake methods:** 6

Prepared facial samples:

### Training

- Real faces: **32,174**
- Fake faces: **26,724**
- Total training faces: **58,898**

### Validation

- Real faces: **7,883**
- Fake faces: **6,241**
- Total validation faces: **14,124**

Total prepared facial images:

**73,022**

---

## Model Experiments

Two main models were evaluated during development.

### Model 1 — Deepfakes-Only

The first model was trained primarily using the FaceForensics++ `Deepfakes` manipulation method.

Evaluation results:

| Metric | Result |
|---|---:|
| Accuracy | 86.18% |
| Precision | 88.48% |
| Recall | 83.22% |
| F1 Score | 85.77% |
| ROC-AUC | 0.9562 |

Although Model 1 achieved strong validation metrics, cross-manipulation testing showed that its generalization was limited.

It correctly detected Deepfakes but incorrectly classified several manipulation methods as real.

---

## Model 2 — Multi-Manipulation

The second model was trained using six manipulation families:

- Deepfakes
- Face2Face
- FaceSwap
- NeuralTextures
- FaceShifter
- DeepFakeDetection

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | **78.16%** |
| Precision | **82.69%** |
| Recall | **63.98%** |
| F1 Score | **72.14%** |
| ROC-AUC | **0.8818** |
| Validation Samples | **14,124** |

The model stopped early during training and retained the checkpoint with the strongest validation performance.

While its validation accuracy was lower than Model 1, its cross-manipulation behavior improved substantially.

---

## Cross-Manipulation Spot Testing

One known sample from each manipulation family was tested against Model 2.

| Video Type | Expected | Prediction | Confidence | Result |
|---|---|---|---:|---|
| Original Actor Video | Real | Real | 82.00% | Correct |
| Deepfakes | Fake | Fake | 96.02% | Correct |
| Face2Face | Fake | Fake | 70.82% | Correct |
| FaceSwap | Fake | Fake | 95.57% | Correct |
| NeuralTextures | Fake | Real | 72.52% | Incorrect |
| FaceShifter | Fake | Fake | 71.90% | Correct |
| DeepFakeDetection | Fake | Fake | 61.82% | Correct |

These tests indicate substantially improved cross-manipulation behavior compared with the Deepfakes-only model.

These results are **spot checks rather than a complete cross-dataset benchmark**, and should therefore not be interpreted as statistically complete generalization measurements.

---

## Current Limitation

The most visible cross-manipulation weakness observed during spot testing was **NeuralTextures**.

The NeuralTextures test video was incorrectly classified as authentic:

- Real probability: 72.52%
- Fake probability: 27.48%

Additional training data, targeted augmentation, improved temporal modeling, or architecture changes may improve performance against difficult manipulation techniques.

---

## Processing Pipeline

The ADDFS video analysis pipeline follows this sequence:

```text
Video Upload
     |
     v
Video Validation
     |
     v
Frame Extraction
     |
     v
Face Detection
     |
     v
Face Cropping
     |
     v
EfficientNet-B0 Inference
     |
     v
Probability Aggregation
     |
     v
Real / Deepfake Classification
     |
     v
Browser Results
     |
     v
PDF Analysis Report
```

---

## Video Prediction

For each video:

1. Frames are extracted at a configured interval.
2. Faces are detected from the extracted frames.
3. Each detected face is passed through the trained EfficientNet-B0 model.
4. Fake probabilities from the analyzed faces are averaged.
5. The average probability is compared against the configured threshold.
6. The video is classified as:
   - `0_real`
   - `1_fake`

The prediction output includes:

- Classification
- Confidence
- Real probability
- Fake probability
- Frames analyzed
- Faces analyzed
- Processing time
- Frame interval
- Model architecture
- Model checkpoint

---

## Browser Dashboard

The project includes a browser interface available through FastAPI.

Features include:

- Video file selection
- Drag-and-drop upload
- Analysis progress indicator
- Prediction classification
- Confidence percentage
- Real probability
- Fake probability
- Processing time
- Frames analyzed
- Faces analyzed
- Model information
- Original filename
- Recent analysis history
- Clear-history function
- Downloadable PDF analysis report

Recent history is stored locally in the user's browser using `localStorage`.

---

## PDF Analysis Reports

ADDFS can generate a professional PDF report after a video has been analyzed.

The PDF contains:

- Final classification
- Confidence
- Original filename
- Real probability
- Fake probability
- Decision threshold
- Frames analyzed
- Faces analyzed
- Frame interval
- Processing time
- Video size
- Model architecture
- Model checkpoint
- Automated-analysis disclaimer

PDF generation uses:

```text
ReportLab
```

---

## FastAPI Endpoints

### Prediction Interface

```http
GET /prediction/ui
```

Opens the browser-based deepfake detection interface.

### Direct Video Upload and Prediction

```http
POST /prediction/video-upload
```

Accepts a video file, performs inference, and returns prediction information.

Supported formats:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`

### Predict Previously Stored Video

```http
POST /prediction/video/{stored_filename}
```

Runs prediction against a video already available in the configured upload directory.

### PDF Report

```http
POST /prediction/report
```

Accepts prediction result data and returns a downloadable PDF report.

---

## Project Structure

```text
addsfi-v/
|
|-- frontend/
|   `-- index.html
|
|-- scripts/
|   |-- evaluate.py
|   |-- faceforensics_download.py
|   |-- predict_image.py
|   |-- predict_video.py
|   |-- prepare_dataset.py
|   `-- train.py
|
|-- src/
|   `-- addfs/
|       |-- api/
|       |   |-- prediction.py
|       |   `-- upload.py
|       |
|       |-- config/
|       |   |-- __init__.py
|       |   `-- settings.py
|       |
|       |-- dataset_builder/
|       |   |-- builder.py
|       |   |-- face_processor.py
|       |   |-- frame_extractor.py
|       |   `-- splitter.py
|       |
|       |-- detection/
|       |   `-- deepfake_detector.py
|       |
|       |-- evaluation/
|       |   `-- evaluator.py
|       |
|       |-- inference/
|       |   |-- image_predictor.py
|       |   `-- video_predictor.py
|       |
|       |-- preprocessing/
|       |
|       |-- reporting/
|       |   |-- __init__.py
|       |   `-- report_generator.py
|       |
|       |-- training/
|       |   |-- dataloader.py
|       |   |-- dataset.py
|       |   |-- experiment.py
|       |   `-- trainer.py
|       |
|       `-- utils/
|
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- README.md
|-- pyproject.toml
`-- requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/marvinmarphie-cloud9/addsfi-v.git
cd addsfi-v
```

Checkout the development branch when required:

```bash
git checkout feature/authentication-api
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Environment Configuration

Copy:

```text
.env.example
```

to:

```text
.env
```

The application supports environment-based configuration for paths and major runtime settings.

Important configuration values include:

- FaceForensics++ location
- Prepared dataset location
- Working dataset location
- Results directory
- Models directory
- Upload directory
- Temporary directory
- Batch size
- Number of workers
- Epochs
- Frame interval
- Image size
- Learning rate
- Early stopping patience
- Scheduler patience
- Confidence threshold
- Maximum upload size

The local `.env` file should **not** be committed to Git.

---

## Prepare the Dataset

Run:

```powershell
python scripts\prepare_dataset.py
```

The multi-manipulation version builds training and validation datasets using real videos and multiple manipulation techniques.

---

## Train the Model

Run:

```powershell
python scripts\train.py
```

Training outputs are stored under:

```text
results/training/run_<timestamp>/
```

Typical output includes:

```text
best_model.pt
history.csv
metrics.json
```

---

## Evaluate a Model

Run:

```powershell
python scripts\evaluate.py
```

Evaluation outputs may include:

```text
classification_report.txt
confusion_matrix.png
evaluation_metrics.json
roc_curve.png
```

---

## Predict a Video from the Command Line

Run:

```powershell
python scripts\predict_video.py "path\to\video.mp4"
```

Example output:

```text
Prediction: 1_fake
Confidence: 96.02%
Real probability: 3.98%
Fake probability: 96.02%
Frames analyzed: 14
Faces analyzed: 15
```

---

## Run the Web Application

Start FastAPI:

```powershell
uvicorn addfs.main:app --app-dir src --reload
```

Open:

```text
http://127.0.0.1:8000/prediction/ui
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Supported Video Formats

The upload interface currently supports:

```text
MP4
MOV
AVI
MKV
```

Maximum upload size is controlled through the project configuration.

---

## Configuration Portability

The project was designed so local machine-specific paths do not have to be hardcoded into the application.

Environment configuration allows the system to run across different development machines while keeping datasets and trained models outside Git where appropriate.

---

## Security and Responsible Use

Deepfake detection systems are probabilistic.

A prediction should not be interpreted as absolute proof that media is authentic or manipulated.

ADDFS should be used as a **decision-support system**, particularly where conclusions may have legal, security, journalistic, academic, or reputational consequences.

---

## Future Improvements

Potential future work includes:

- Improved NeuralTextures detection
- Larger cross-manipulation evaluation
- Cross-dataset testing
- Temporal neural-network architectures
- Transformer-based video analysis
- Face tracking across frames
- Attention visualization
- Explainable AI
- Model calibration
- Persistent server-side analysis history
- User authentication
- Database-backed reports
- Deployment through containerized infrastructure
- GPU-enabled production deployment

---

## Technology Stack

- Python
- FastAPI
- PyTorch
- Torchvision
- EfficientNet-B0
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- ReportLab
- HTML
- CSS
- JavaScript
- Git
- GitHub

---

## Project Status

The current ADDFS prototype supports the complete core workflow:

- Dataset preparation
- Multi-manipulation training
- Model evaluation
- Image inference
- Video inference
- FastAPI prediction API
- Browser-based video analysis
- Analysis history
- Processing statistics
- PDF report generation

The multi-manipulation model currently provides improved manipulation-family coverage compared with the original Deepfakes-only baseline.

---

## Disclaimer

This project was developed for research and academic purposes.

Deepfake detection predictions are probabilistic and may contain false positives or false negatives. Results should be independently verified when used in sensitive or consequential environments.