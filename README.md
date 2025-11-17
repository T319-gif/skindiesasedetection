
# DERMA CARE - AI Skin Analysis

This is a Next.js web application that uses a custom, locally-run Python CNN model to analyze images of skin and provide potential condition predictions. It also features a GenAI component to give personalized skincare advice based on the model's prediction.

![DERMA CARE Screenshot](https://storage.googleapis.com/stabl-media/6a032890-410a-42c2-8419-fcc61a0a57e3.png)

## Core Features

- **AI-Powered Analysis**: Upload an image of your skin to get an analysis from a custom-trained Convolutional Neural Network (CNN).
- **Local Model Execution**: The Python/TensorFlow model runs on a local Flask server, ensuring user data privacy and leveraging your custom-trained model.
- **Personalized Routines**: Get AI-generated skincare routine recommendations based on the analysis results.
- **Modern UI/UX**: Built with Next.js, ShadCN UI, and Tailwind CSS for a sleek, responsive, and animated user interface.

## Tech Stack

- **Frontend**: Next.js, React, TypeScript
- **Styling**: Tailwind CSS, ShadCN UI
- **AI (Backend Model)**: Python, Flask, TensorFlow/Keras
- **AI (Recommendations)**: Genkit, Google Gemini

---

## Getting Started

This project requires two separate processes to be running simultaneously: the **Next.js website** and the **Python Flask API**.

Update (2025-11-11)
- The Python API now defaults to EfficientNet preprocessing to match the trained model (EfficientNetV2B0). This fixes "same output for different images" issues.
- Class labels are served from `class_labels.json` using the full HAM10000 names.
- The AI skincare routine is disabled by default so the site focuses on the model’s predictions. You can re‑enable it by setting a Gemini/Google API key (see below).

### Prerequisites

- **Node.js**: Version 18 or later.
- **Python**: Version 3.9 or later.
- **Your Trained Model**: You must have your trained model file, named `skin_condition_model.h5`, placed in the root directory of this project.

### Step 1: Set Up the Frontend (Next.js Website)

First, open a terminal, navigate to the project's root directory, and follow these steps:

1.  **Install Dependencies**:
    ```bash
    npm install
    ```

2.  **Create Environment File**:
    Create a file named `.env.local` in the root of your project and add the following line. This tells the website where to find your local Python API.
    ```
    PYTHON_API_URL=http://127.0.0.1:5001/predict
    ```

3.  **Run the Website Server**:
    ```bash
    npm run dev
    ```
    Your website should now be running at `http://localhost:9002`.

### Step 2: Set Up the Backend (Python API)

Now, **open a new, separate terminal** and navigate to the same project root directory.

1.  **Install Python Dependencies**:
    It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the required packages
    pip install Flask tensorflow "Pillow" "python-dotenv" "Flask-Cors" numpy
    ```

2.  **Ensure Model File is Present**:
    Make sure your trained model file, `skin_condition_model.h5`, is in the root directory of the project.

3.  **Run the Python Server** (EfficientNet preprocessing is enabled by default):
    ```bash
    python main.py
    ```
    You should see messages like:
    - `Using EfficientNetV2 preprocess_input`
    - `✅ Model loaded successfully.`

    Notes:
    - Place your trained model file as `skin_condition_model.h5` in the project root, or set `MODEL_PATH=/full/path/to/model.h5`.
    - The file `class_labels.json` controls the display names and class order. Keep it aligned with your training generator order.
    - To switch back to a simple `/255` normalization (not recommended for this model), run: `PREPROCESS=rescale python main.py`.

### Step 3: Use the Application

With both servers running, open `http://localhost:9002`, go to the Analyze section, upload an image, and you’ll see probabilities for all 7 HAM10000 classes rendered as a bar chart and a pie chart.

### Optional: Re‑enable AI skincare routine
If you want to generate a text routine, set an API key and restart the site:
```bash
# .env.local
GEMINI_API_KEY=YOUR_KEY
# or
GOOGLE_API_KEY=YOUR_KEY
```
If a key is not provided, the routine section is hidden and only model predictions are shown.
