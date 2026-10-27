# NeuroCanvas

A glassmorphism-style BSc CSIT Neural Network project with two AI pipelines:

1. **Doodle Recognition — CNN**
   - Trains a compact Convolutional Neural Network on Google Quick, Draw! data.
   - Six classes: apple, cat, flower, house, star, tree.
   - Draw → preprocess → CNN → class probabilities.

2. **Handwriting → Text — Neural OCR**
   - Uses EasyOCR, a neural-network based OCR engine.
   - Write on the canvas → recognize → edit the text → export to DOCX.

## Run

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install packages

```powershell
pip install -r requirements.txt
```

### 3. Train the CNN once

```powershell
python train_doodle_model.py
```

This downloads the six Quick, Draw! datasets and saves `models/doodle_cnn.keras`.

### 4. Start the app

```powershell
python app.py
```

Open:

`http://127.0.0.1:5000`

## Academic pipeline

**Doodle:** Canvas → grayscale/resize → normalization → CNN → Softmax → prediction.

**Handwriting:** Canvas → image preprocessing → neural OCR → editable text → DOCX.

## Suggested report chapters

- Problem statement
- Objectives
- Dataset
- Data preprocessing
- CNN architecture
- Activation functions
- Loss function
- Backpropagation
- Training/validation/testing
- Accuracy and loss graphs
- Confusion matrix
- OCR pipeline
- System architecture
- Results and limitations
- Future improvements

## Important

The CNN model is intentionally trained locally so you can show your lecturer the actual neural-network training process rather than hiding everything behind an API.

## Important: what gets converted
- Doodle Mode recognizes the drawing as a class using the trained CNN and can save the drawing itself as a normal PNG image.
- Handwriting Mode converts the written strokes into editable text using neural OCR and exports that text to DOCX.
- The CNN does NOT magically turn a doodle into a photorealistic picture. Recognition and image generation are different AI tasks.
