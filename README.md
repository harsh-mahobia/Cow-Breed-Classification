## Cattle Breed Classification (Streamlit + YOLOv8)

An interactive Streamlit app for classifying cattle breeds using a YOLOv8 classification model.

### Features
- Upload JPG/PNG images for classification
- Cached model loading for fast repeat inferences
- Top-1 prediction with confidence
- Top-k probabilities displayed as progress bars
- Adjustable confidence display threshold

### Requirements
- Python 3.8+
- Packages in `requirements.txt`:
  - ultralytics
  - pillow
  - streamlit

### Setup
1. Create/activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure your trained YOLOv8 classification model file is available (default expected: `yolov8_cattle_best.pt` in the project root). If your filename or path is different, you can change it in the app sidebar when running.

### Run the app

```bash
streamlit run main.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

### Usage
1. In the sidebar:
   - Verify or edit the model path (defaults to `yolov8_cattle_best.pt`).
   - Optionally adjust the confidence display threshold.
2. Upload a cattle image (`.jpg`, `.jpeg`, or `.png`).
3. View:
   - The uploaded image
   - Top-1 predicted breed and confidence
   - Top-k probabilities as progress bars

### Model Notes
- This app expects a YOLOv8 classification model (not detection/segmentation).
- The class labels displayed come from `model.names` embedded in the model.

### Troubleshooting
- Model fails to load:
  - Confirm the model path is correct and the file exists.
  - Ensure the model is a classification model trained with Ultralytics YOLOv8.
- No probabilities shown:
  - Likely not a classification model or the model lacks probability outputs.
- GPU vs CPU:
  - By default, Ultralytics selects an available device. To force CPU, you can run without CUDA installed or configure device selection in code if needed.

### Project Structure (minimal)
```
6-captone/
├─ main.py               # Streamlit app
├─ requirements.txt
├─ yolov8_cattle_best.pt # (example model file, not included in repo)
└─ test_images/          # (optional sample images)
```

### License
This project is provided as-is; include your preferred license if distributing. 


