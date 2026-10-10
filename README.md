# EcoSort: Waste Segregation CNN Web App

The browser lets a user choose or drag in an image, preview it, and press **Scan image**. JavaScript uploads the actual image file to a FastAPI endpoint. The backend preprocesses it and uses the already-trained custom CNN checkpoint to return predicted class and confidence.

## Setup
1. Copy your existing `waste_cnn.pth` checkpoint into the same folder as `app.py`.
2. Open a terminal in this folder and run:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python -m uvicorn app:app --reload
   ```
3. Open `http://127.0.0.1:8000` in your browser.
4. Select an image and press **Scan image**.

The checkpoint must contain `model_state_dict` and `classes`, as saved by the existing training script. The architecture in `model.py` and preprocessing (128x128 plus the specified normalization) must match how the checkpoint was trained. The model file is not included in this ZIP; add your already-trained `waste_cnn.pth`.

## Command-line prediction
The included `predict.py` still supports the earlier path-based approach:

```bash
python predict.py "C:\path\to\image.jpg"
```

## Category grouping
The backend maps class names containing `food`, `leaf`, `leaves`, `paper`, `wood`, `organic`, or `biodegradable` to **Biodegradable** and all other classes to **Non-biodegradable**. This is a simplified project rule; change the set in `app.py` if needed. Confidence is a model score, not a guarantee of correctness.
