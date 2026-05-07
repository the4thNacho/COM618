# COM618 — Diabetes Readmission Predictor

Predicts 30-day hospital readmission risk using the UCI Diabetes 130-US Hospitals dataset.

## Dataset

Download the dataset from UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Extract the zip and place the folder in the project root so the path looks like:

```
diabetes+130-us+hospitals+for+years+1999-2008/
    diabetic_data.csv
```

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the app

```bash
python app.py
```

Then open the URL shown in the terminal (e.g. `http://127.0.0.1:5000`).

The app will clean the data and train the model automatically on first run.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Dataset overview |
| `/cleaning` | Data cleaning pipeline |
| `/exploration` | EDA charts |
| `/model` | Live readmission risk prediction |
| `/performance` | Model performance dashboard |

## Options

```bash
python app.py --retrain   # Force model retraining
```
