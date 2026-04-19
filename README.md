# SpamShield AI

Premium Streamlit application for classifying SMS messages as spam or not spam using a strong TF-IDF machine learning pipeline.

## Project Structure

```text
spamshield-ai/
├── app.py
├── model.py
├── preprocess.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── artifacts/
    ├── model.pkl
    ├── vectorizer.pkl
    └── model_metadata.json
```

The `artifacts/` files are created after training and should exist before running the app.

## Setup

1. Download the Kaggle **SMS Spam Collection Dataset**.
2. Place `spam.csv` in the project root.
3. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## Training Command

```bash
python model.py --data spam.csv
```

This trains Naive Bayes, Logistic Regression, and Random Forest models, compares their metrics, automatically selects the best model by F1-score, and saves:

```text
artifacts/model.pkl
artifacts/vectorizer.pkl
artifacts/model_metadata.json
```

## Run Command

```bash
streamlit run app.py
```

Open the local URL shown in your terminal.

## Deployment on Streamlit Cloud

1. Push this project to GitHub.
2. Make sure `requirements.txt`, `app.py`, `model.py`, `preprocess.py`, and the trained `artifacts/` folder are committed.
3. Go to Streamlit Cloud and create a new app from your GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

If you do not commit the trained artifacts, the deployed app will ask you to train the model first.
