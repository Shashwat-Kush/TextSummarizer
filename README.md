# End-to-End Text Summarizer Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-orange)

## 📌 Overview
This is a comprehensive End-to-End Machine Learning project that implements an abstractive text summarization system using the **Google Pegasus** model (`google/pegasus-cnn_dailymail`). The project is built with a production-grade MLOps pipeline structure, handling everything from data ingestion to model deployment. It utilizes the **SAMSum dataset** to fine-tune the model for dialogue summarization tasks.

## 🚀 Key Features
* **Modular Architecture:** The code is organized into distinct pipeline stages (Ingestion, Validation, Transformation, Training, Evaluation) for scalability and maintainability.
* **State-of-the-Art Model:** Fine-tunes the Pegasus transformer model for high-quality abstractive summaries.
* **Custom Evaluation:** Implements ROUGE metric calculation (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum) to rigorously assess model performance.
* **Experiment Tracking:** Separates research notebooks (`research/`) from production code (`src/`).
* **Deployment Ready:** Includes `app.py` for API access and `Dockerfile` for containerization.

## 🛠️ Tech Stack
* **Programming Language:** Python 3.8+
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Data Processing:** Pandas, Datasets
* **MLOps Utilities:** Box (ConfigBox), Ensure, Accelerate
* **Infrastructure:** Docker, GitHub Actions (CI/CD)

## 📂 Project Structure

```bash
TextSummarizer/
├── .github/workflows/    # CI/CD configurations
├── config/               # Configuration files (config.yaml)
├── research/             # Jupyter notebooks for initial experiments
├── src/                  # Source code
│   └── textSummarizer/
│       ├── components/   # Core logic (Ingestion, Trainer, etc.)
│       ├── config/       # Configuration managers
│       ├── pipeline/     # Pipeline execution scripts
│       ├── entity/       # Dataclasses for type safety
│       └── constants/    # Global constants
├── artifacts/            # Generated assets (Datasets, Models, Metrics)
├── app.py                # Application entry point (API/Web)
├── main.py               # Main pipeline execution script
├── params.yaml           # Hyperparameters (Epochs, Batch Size, etc.)
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
├── Dockerfile            # Docker image configuration
└── template.py           # Script for folder structure generation
```
## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Shashwat-Kush/TextSummarizer.git](https://github.com/Shashwat-Kush/TextSummarizer.git)
    cd TextSummarizer
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    conda create -n summary python=3.8 -y
    conda activate summary
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### 1. Run the Training Pipeline
To execute the complete pipeline (Ingestion $\rightarrow$ Validation $\rightarrow$ Transformation $\rightarrow$ Training $\rightarrow$ Evaluation):
```bash
python main.py
