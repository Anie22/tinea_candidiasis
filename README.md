# Tinea vs Candidiasis Classifier

A deep learning-powered web application developed using **TensorFlow**, **MobileNetV2**, and **Streamlit** to classify skin lesion images as either **Tinea (Ringworm)** or **Candidiasis**.

This project was developed as part of the **GET 324 Laboratory Exercise 10 (Mini Project)** on AI model development and cloud deployment.

---

## Features

* Binary image classification
* MobileNetV2 transfer learning
* Streamlit web interface
* Confidence-based prediction rejection
* Automatic model loading
* Google Colab compatible
* Ready for cloud deployment

---

## Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* Streamlit
* NumPy
* Pillow
* scikit-learn
* Matplotlib

---

## Project Structure

```text
tinea_candidiasis_project/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── skin_classifier.keras
│   ├── labels.json
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
└── data/
    ├── tinea/
    └── candidiasis/
```

> **Note:** The `data/` directory is used only for model training and should not be committed to GitHub.

---

# Dataset

The model was trained using images of **Tinea** and **Candidiasis** obtained from the **Kaggle Skin Disease Dataset**.

Dataset Link:

https://www.kaggle.com/datasets/pacificrm/skindiseasedataset

For best performance, use a balanced dataset with sufficient images for each class.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Navigate into the project directory:

```bash
cd tinea_candidiasis_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Windows

```bash
venv\Scripts\activate
```

macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training the Model

Train the CNN using:

```bash
python train.py
```

After training, the following files will be generated inside the `model/` directory:

* skin_classifier.keras
* labels.json
* training_history.png
* confusion_matrix.png
* classification_report.txt

---

# Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open the URL displayed in the terminal (typically `http://localhost:8501`).

Upload an image of a skin lesion and wait for the prediction.

---

# Prediction Logic

The application predicts one of the following classes:

* **Tinea**
* **Candidiasis**

To improve reliability, the application uses a confidence threshold.

If the prediction confidence is below the predefined threshold, the application displays:

> *"The uploaded image could not be confidently classified as Tinea or Candidiasis."*

This helps prevent unreliable predictions for unrelated skin conditions or non-skin images.

---

# Deployment

The application can be deployed on **Streamlit Community Cloud**.

Steps:

1. Push the project to GitHub.
2. Ensure the `model/` folder is included.
3. Create a new Streamlit application.
4. Select `app.py` as the entry point.
5. Deploy.

---

# GitHub Repository Contents

Include:

* app.py
* train.py
* README.md
* requirements.txt
* .gitignore
* model/

Do **not** upload:

* data/
* venv/
* **pycache**/

---

# Future Improvements

* Increase the dataset size for improved accuracy.
* Support additional skin disease classes.
* Integrate Grad-CAM for model interpretability.
* Improve image quality assessment before prediction.
* Deploy a mobile-friendly version.

---

# License

This project was developed for academic purposes as part of the **GET 324 Mini Project**.
