# 🦠 Mpox Skin Lesion Classification Using Deep Learning

A Streamlit-based web application for **binary classification of skin lesion images into Mpox and Others** using a PyTorch DenseNet169 deep learning model.

The application allows users to upload a skin image and view the model's predicted class and corresponding prediction confidence or probability, depending on the inference implementation.

## 📌 Project Overview

Mpox skin lesions can resemble lesions caused by other skin diseases. This project explores the use of deep learning for automated image-based classification to support research into skin lesion analysis.

The application uses a DenseNet169 model trained on a binary skin-image dataset and provides a web interface built with Streamlit.

### Key Features

- 🖼️ Upload skin lesion images through a web interface.
- 🧠 Predict between two classes: `Mpox` and `Others`.
- 📊 Display the predicted class and model-generated probability, if enabled in the app.
- ⚡ Run inference using PyTorch and DenseNet169.
- 🌐 Access the application through a browser after deployment.
- 🔬 Support research and experimentation in AI-based skin lesion classification.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning and model inference |
| Torchvision | DenseNet169 architecture and image transformations |
| Streamlit | Web application interface |
| Pillow (PIL) | Image loading and processing |
| Kaggle | Model training and experimentation |
| GitHub | Source code and version control |
| Streamlit Community Cloud | Web application deployment |

## 🧠 Model Information

- **Architecture:** DenseNet169
- **Framework:** PyTorch
- **Task:** Binary image classification
- **Input:** Skin lesion image
- **Output classes:** `Mpox` and `Others`
- **Model checkpoint:** `densenet169_Hyperdataset_5_(Binary)_last_checkpoint.pth`

### Classification Classes

| Class | Description |
|---|---|
| Mpox | The model predicts that the image belongs to the Mpox class. |
| Others | The model predicts that the image belongs to the non-Mpox class used during training. |

The `Others` class represents the non-Mpox categories included in the training dataset. It should not be interpreted as a definitive identification of a particular alternative skin disease.

### Image Preprocessing

The inference pipeline should match the preprocessing used during model training. The intended preprocessing configuration is:

1. Convert the input image to RGB.
2. Resize the image to 224 × 224 pixels.
3. Convert the image to a PyTorch tensor.
4. Normalize the image using ImageNet mean and standard deviation.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Important:** Verify that the image size, normalization, class order, model architecture, and checkpoint loading procedure match the actual training configuration.

## 📂 Project Structure

```text
mpox-streamlit-app/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    └── densenet169_Hyperdataset_5_(Binary)_last_checkpoint.pth
```

- `app.py` — Streamlit application and image prediction interface.
- `requirements.txt` — Python dependencies.
- `models/` — Directory containing the trained model checkpoint.
- `README.md` — Project documentation.

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shadowstacker21/mpox-streamlit-app.git
```

Navigate to the project directory:

```bash
cd mpox-streamlit-app
```

Replace `YOUR_USERNAME` and the repository name with your actual GitHub details.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
streamlit
torch
torchvision
Pillow
```

Use compatible dependency versions for your development and deployment environments.

### 4. Add the Trained Model

Place the checkpoint in the `models/` directory:

```text
models/densenet169_Hyperdataset_5_(Binary)_last_checkpoint.pth
```

Ensure that `app.py` references this exact path and loads the checkpoint using the correct DenseNet169 architecture and checkpoint format.

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

### Deployment Steps

1. Push the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using GitHub.
4. Create a new application and select your repository.
5. Set the main file path to `app.py`.
6. Deploy the application.
7. Test the deployed URL with sample images.

### Deployment Considerations

- Ensure that the model checkpoint is accessible during deployment.
- Large checkpoint files may require Git LFS or external model storage.
- Verify the memory and resource requirements of DenseNet169.
- Check that all Python dependencies are compatible.
- Test image preprocessing and prediction outputs after deployment.

## 📊 Evaluation and Research

This project focuses on image-based binary classification. A rigorous evaluation should include:

- Accuracy
- Precision
- Recall (sensitivity)
- Specificity
- F1-score
- ROC-AUC
- Confusion matrix
- External validation on images from independent sources

Evaluation should be conducted on held-out test data and, where possible, independent external datasets. Results should be reported only after they have been measured and verified.

## 🔬 Potential Research Extensions

The project can be extended with:

- Explainable AI techniques such as Grad-CAM.
- External dataset validation.
- Class imbalance analysis.
- Threshold selection and calibration.
- Robustness testing across image sources and acquisition conditions.
- Comparison with other deep learning architectures.
- Data-centric evaluation and error analysis.

These are potential extensions and should not be considered implemented features unless they are present in the actual codebase.

## ⚠️ Medical Disclaimer

**This application is intended for research and educational purposes only.**

The model's output is a machine-learning prediction, not a confirmed diagnosis or a clinically validated risk assessment. Prediction probabilities may not represent calibrated real-world probabilities. The application must not replace professional medical evaluation, laboratory testing, or advice from qualified healthcare professionals.

## 👨‍💻 Author

**Alamin Hossain**

- GitHub: [shadowstacker21](https://github.com/shadowstacker21)
- LinkedIn: [Alamin Hossain](https://www.linkedin.com/in/alamin-hossain-854456330/)

## 📄 License

Choose and add an appropriate open-source license before distributing the project. If the dataset or model weights come from third-party sources, follow their respective licenses and usage conditions.
