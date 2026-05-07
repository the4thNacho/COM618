# COM618 — Diabetes Readmission Predictor

Predicts 30-day hospital readmission risk using the UCI Diabetes 130-US Hospitals dataset.

## Requirements

- Python 3.10 or later

## Dataset

Download the dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Extract the zip and place the folder in the project root so the structure looks like this:

```
diabetes+130-us+hospitals+for+years+1999-2008/
    diabetic_data.csv
app.py
predictor.py
...
```

## Setup

**1. Create a virtual environment:**

Windows:
```
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies:**
```
pip install -r requirements.txt
```

## Running the app

```
python app.py
```

Then open the URL shown in the terminal (e.g. `http://127.0.0.1:5000`).

> **Note:** The first run will take a few minutes — the app cleans the data and trains the model automatically before starting.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Dataset overview |
| `/cleaning` | Data cleaning pipeline |
| `/exploration` | EDA charts |
| `/model` | Live readmission risk prediction |
| `/performance` | Model performance dashboard |

## Options

```
python app.py --retrain   # Force model retraining
```
