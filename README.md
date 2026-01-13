# Optical Character Recognition (OCR) 

A simple, reproducible pipeline for generating synthetic character templates and training an OCR model to recognize uppercase letters, lowercase letters, and digits.

---

## Project overview 

This repository contains two main notebooks:

- `NoteBooks/template_generator.ipynb` — generates a synthetic template dataset (A–Z, a–z, 0–9) using the Arial TrueType font and saves image files in a `template/` directory.
- `NoteBooks/OCR_Dataset.ipynb` — demonstrates a full OCR pipeline: dataset extraction, image preprocessing, HOG feature extraction, training classifiers (Linear SVM shown), and evaluating performance (accuracy, classification report, confusion matrix, sample predictions).

A `Datasets/template.zip` archive is also included for quick reuse of the template dataset.

---

## Repository structure 

- `Datasets/` — dataset archive(s), e.g. `template.zip`.
- `NoteBooks/` — interactive Jupyter notebooks for dataset generation and training (`template_generator.ipynb`, `OCR_Dataset.ipynb`).
- `README.md` — this file.

---

## Quick start 

1. Clone the repo:

```bash
git clone https://github.com/Mai-Fakhry/Optical-character-recognition-OCR-.git
cd Optical-character-recognition-OCR-
```

2. Option A — Use the included template dataset:

- Unzip `Datasets/template.zip` into the project root (it contains the `template/` directory with images).

2. Option B — Regenerate templates from the notebook:

- Open `NoteBooks/template_generator.ipynb` (Colab badge available at top) and run the cells to install dependencies and create `template/` (uppercase, lowercase, digits).

3. Prepare and run the OCR pipeline:

- Open `NoteBooks/OCR_Dataset.ipynb`. Install dependencies (`kaggle`, `scikit-image`, `scikit-learn`, `opencv-python`, etc.) and follow the notebook's steps to extract features, train a model, and evaluate results.

---

## Dependencies 

Typical Python packages used in the notebooks:

- Python 3.8+
- numpy, pandas
- scikit-image
- scikit-learn
- opencv-python (cv2)
- pillow (PIL)
- matplotlib, seaborn
- kaggle (optional, for dataset download)

Install with pip:

```bash
pip install numpy pandas scikit-image scikit-learn opencv-python pillow matplotlib seaborn kaggle
```

---

## Notable implementation details 

- Image preprocessing: adaptive thresholding, bounding-box cropping, square padding and resizing (default target size: 40×60).
- Feature extraction: Histogram of Oriented Gradients (HOG) features.
- Classifier: Linear SVM (shown in the notebook). The pipeline uses an 80/20 train/test split with stratified sampling.
- Evaluation: accuracy, per-class precision/recall/F1, confusion matrix, and example predictions visualized.

---

## Results & Next steps 

- The baseline SVM trained on HOG features produces competitive accuracy for synthetic/templates-driven OCR. Suggested improvements:
  - Try convolutional neural networks for better invariance
  - Add data augmentation and more varied fonts
  - Balance or expand dataset for under-represented characters

  ---
  
## Contact
-Authors:-Mohammed Magdy Taher
         -Mai Fakhry Mohammed

-Email:-MohammedTaher.6705@gmail.com
       -salemmay87@gmail.com

-Questions or changes? message us.
