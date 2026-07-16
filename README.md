# GeoCLIP AI Locator: Geographic Prediction System

[![Project Status](https://img.shields.io/badge/Status-Active-success.svg)](#)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](#)
[![AI Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](#)

## Overview
The Geo-Locator is an advanced artificial intelligence system built upon the GeoCLIP architecture. It is designed to analyze images visually and deduce their exact geographic location worldwide, or within a specific localized region (e.g., Iraq), by matching image embeddings with a large-scale geospatial vector database.

## Key Features
- **Visual AI (Computer Vision):** Utilizes the CLIP architecture (ViT-L/14) to extract deep visual features from images.
- **Hybrid Search Architecture (v0.1):** Employs FAISS (Facebook AI Similarity Search) with an `IndexIVFFlat` structure for coarse classification across 4,096 geographic cells, enabling sub-second search times across over 5 million global coordinates.
- **Optical Character Recognition (OCR):** Integrates `EasyOCR` to extract bilingual text (English/Arabic) from images, providing critical contextual clues for location resolution.
- **EXIF Metadata Extraction:** Automatically parses hidden GPS metadata from raw images to provide 100% accurate coordinates when available.
- **Localized Regional Database:** Includes a dedicated, high-resolution dataset containing over 55,000 specific landmarks, villages, and streets strictly within Iraq to drastically improve local prediction accuracy.

## Tech Stack
- **Language:** Python 3
- **Deep Learning Framework:** PyTorch (CUDA supported)
- **Vector Search Engine:** FAISS
- **Data Processing:** Pandas, NumPy
- **Image Processing & OCR:** PIL (Pillow), Torchvision, EasyOCR
- **Frontend UI:** Streamlit

## Installation
Ensure you have Python 3 installed. Install the required dependencies using the following commands:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-cpu pandas numpy streamlit pillow easyocr
```

## Index Generation (First-Time Setup)
Before running the application, you must generate the geospatial vector indices. The script has been optimized for memory efficiency (on-the-fly batch processing) to prevent `std::bad_alloc` errors during K-Means training.

**Build the Global Database (5.1 Million points / 4096 cells):**
```bash
python src/build_global_index.py
```

**Build the Regional Database (Iraq):**
```bash
python src/build_iraq_index.py
```

## Usage

### Web Interface
To launch the Streamlit web application:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501`. Upload an image, optionally toggle the regional database constraint, and click "Predict Location". 

### Command Line Interface (CLI)
To run predictions directly from the terminal:

**Global Search:**
```bash
python src/predict.py "path/to/image.jpg"
```

**Regional Search (Iraq):**
```bash
python src/predict.py "path/to/image.jpg" --iraq
```

## Research & Development Notes
- **Hybrid System (Coarse Classification):** Replaced exhaustive L2 distance search with `IndexIVFFlat`. By training the quantizer on a 200,000-sample subset and distributing 5.1M embeddings incrementally, we resolved previous memory leak constraints.
- **Data Fusion:** Combined baseline visual embeddings with EXIF metadata and multi-language OCR to create a robust, multi-modal prediction pipeline.
- **Domain Adaptation:** Overcame geographic bias in the baseline GeoCLIP model by enforcing region-specific FAISS index bounds, forcing the network to match desert/local architecture exclusively with regional datasets.

---

## Academic & Engineering Guide
For an in-depth explanation of the models, the `std::bad_alloc` memory leak solution, how we resolved the geographic bias, and more key code snippets, please refer to the extended academic guide:
**[Read the Academic & Engineering Guide](presentation_guide_extended.md)**
