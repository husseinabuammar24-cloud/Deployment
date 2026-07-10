# 🏠 Immo Eliza - Model Deployment

A machine learning system that predicts Belgian real estate prices, exposed through a **FastAPI** backend and a **Streamlit** frontend.

- **API (backend):** deployed on [Render](https://render.com) as a native Python web service
- **Web app (frontend):** [Streamlit](https://streamlit.io) app that calls the API
- **Model:** scikit-learn regression model trained on cleaned Belgian property listings

---

## Project structure

```
Deployment/
├── app/
│   ├── __init__.py
│   └── main.py                   # FastAPI app
├── Data/
│   ├── SaleCleanFinal.csv
│   └── SaleCleanForAnalysis.csv
├── dev/
│   ├── Mynotebook.ipynb          # exploration notebook
│   └── train_model.py            # training script
├── models/
│   ├── model.pkl                 # trained regression model
│   ├── encoder.pkl                # fitted OneHotEncoder
│   ├── scaler.pkl                 # fitted scaler for numeric features 
│   ├── impute_values.pkl          # fitted imputation values 
│   └── feature_names.pkl          # training-time column order
├── Pipeline/
│   ├── main.py                    # FastAPI app 
│   └── train_model_pipeline.py    # training script 
├── streamlit/
│   └── Streamlit_app.py           # Streamlit frontend
├── venv/
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

`app/`, `dev/`, and `models/` are what's currently deployed. `Pipeline/` is a newer version — training and preprocessing combined into a single scikit-learn `Pipeline`, saved as one `pipeline.pkl` — not yet switched over on Render.

---

## How it works

1. A user fills in property details (surface, bedrooms, location, condition, etc.) in the **Streamlit** app — here: https://immo-eliza-model-deployment.streamlit.app/.
2. Streamlit sends that data as JSON to the **FastAPI** `/predict` endpoint.
3. The API preprocesses the input using the *same* fitted encoder, scaler, and imputer from training, runs it through the model, and returns a predicted price.
4. Streamlit displays the result.

![How it works](images/how%20it%20works.png)

---

## API

### Endpoints

| Method | Route      | Description                                  |
| ------ | ---------- | --------------------------------------------- |
| GET    | `/`        | Health check — returns model status           |
| POST   | `/predict` | Takes property details, returns a price       |

### Example request

```json
POST /predict
{
  "longitude": 4.3517,
  "latitude": 50.8503,
  "property_type": "House",
  "property_subtype": "Villa",
  "date_of_construction": 1998.0,
  "property_condition": "Good",
  "livable_surface": 150.0,
  "number_of_bedrooms": 3.0,
  "number_of_bathrooms": 2.0,
  "elevator": false,
  "furnished": false,
  "availability": "Immediately",
  "province": "Brussels",
  "land_surface": 300.0,
  "balcony": true,
  "swimming_pool": false
}
```

### Example response

```json
{
  "predicted_price": 233390.0
}
```

Full interactive docs (Swagger UI) are available at `/docs` on the deployed API.

### Run the API locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
# → http://127.0.0.1:8001/docs
```

---

## Streamlit app

A simple form-based interface for non-technical users: fill in property details, click **Predict price**, get an instant estimate.

### Run locally

```bash
cd streamlit
pip install -r ../requirements.txt
streamlit run Streamlit_app.py
```

This opens the app at `http://localhost:8501`. The `API_URL` at the top of `Streamlit_app.py` points to the deployed Render API — update it if you redeploy the backend elsewhere.

> **Note:** Render's free tier spins down after inactivity. The first request after idle time can take 30–60 seconds while the API wakes up — the app shows a spinner during this.

---

## Model

Trained with scikit-learn on cleaned Belgian property listing data (`Data/SaleCleanFinal.csv` / `Data/SaleCleanForAnalysis.csv`).

The deployed API (`app/main.py`) loads four separately fitted artifacts and applies them in order:

- Missing-value imputation (`impute_values.pkl`)
- One-hot encoding of categorical features (`encoder.pkl`)
- Numeric feature scaling (`scaler.pkl`)
- The trained model (`model.pkl`)

See `dev/train_model.py` for the training and preprocessing steps that produced these.

A newer version lives in `Pipeline/` — it combines imputation, encoding, scaling, and the model into a single `sklearn.pipeline.Pipeline`, saved as one `pipeline.pkl` (via `Pipeline/train_model_pipeline.py`, served by `Pipeline/main.py`). This isn't deployed yet; switch Render's start command to `uvicorn Pipeline.main:app ...` when you're ready to cut over.

---

## Deployment

| Component  | Platform                        | Live URL                                                   |
| ---------- | -------------------------------- | ------------------------------------------------------------ |
| GitHub repo| GitHub                          | https://github.com/husseinabuammar24-cloud/Deployment        |
| API        | Render (native Python runtime)  | https://immo-eliza-model-deployment.onrender.com              |
| API docs   | Render (Swagger UI)             | https://immo-eliza-model-deployment.onrender.com/docs         |
| Streamlit  | Streamlit Community Cloud        | https://immo-eliza-model-deployment.streamlit.app             |

**Render service settings:**

| Setting        | Value                                                     |
| -------------- | ----------------------------------------------------------- |
| Runtime        | Python 3                                                    |
| Build Command  | `pip install -r requirements.txt`                            |
| Start Command  | `uvicorn app.main:app --host 0.0.0.0 --port $PORT`           |

### Verify the API is live

```bash
curl https://immo-eliza-model-deployment.onrender.com/
```

Expect a JSON health-check response like `{"status": "healthy", "model": "property_price_v1", "model_loaded": true}`.

> Render's free tier spins down when idle, so the first request after inactivity can take 30–60 seconds while it wakes up.

---

## Tech stack

Python · FastAPI · scikit-learn · pandas · Render · Streamlit