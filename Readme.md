
# Anime Recommender System 🎬

A hybrid anime recommendation system built with Python, Flask, and ML pipelines.
It supports recommending anime based on **user ID** (collaborative/hybrid) and (optionally) content-based similarity.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Adding New Data / Retraining](#adding-new-data--retraining)
- [Limitations &amp; Future Work](#limitations--future-work)
- [License](#license)

---

## Project Structure

- `app.py` — Flask application entry point.
- `src/pipeline/prediction_pipeline.py` — contains your hybrid recommendation logic.
- `utils/helper.py` — utility functions: data loading, TF-IDF, similarity, etc.
- `static/` & `templates/` — CSS, HTML for front-end UI.
- `artifacts/` — serialized models, vectorizers, embeddings, etc.
- `logs/` — logs of the app.
- `requirements.txt` — Python dependencies.
- `test.py` — some testing scripts.

---

## Features

- Generate anime recommendations for **existing users** (by user ID) using collaborative/hybrid methods.
- Support content-based / similarity-based recommendations (based on anime metadata).
- Web UI using Flask: user enters a user ID, sees list of recommendations.
- Modular pipeline so you can plug in improvements (e.g. embeddings, better similarity models).

---

## Prerequisites

- Python 3.8+
- `pip`, `virtualenv` (or conda)
- Dataset: anime metadata (CSV) and ratings data
- (Optional) Precomputed models or vectorizers stored under `artifacts/`

---

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/AIAkashMukherjee/Anime-Recommender-System.git
   cd Anime-Recommender-System
   ```



### How to Run

### Steps

Clone the repo

```
git clone https://github.com/AIAkashMukherjee/Anime-Recommender-System.git
```

#### Step 1 -> Create Enviornment

virtualenv my_env

```
source my_env/bin/activate
```

#### Step 2 -> Install the requirements

```
pip install - requirements.txt
```
