# BikeShare Demand Intelligence

Professional Streamlit dashboard for the Bike Sharing Demand Prediction project.

## Files to put in the GitHub repository

- `app.py`
- `requirements.txt`
- `bike_demand_model.joblib` (exported from your notebook)

## 1. Export the trained model from the notebook

Run the notebook through Section 7, especially the cell that saves:

```python
joblib.dump(final_model, "bike_demand_model.joblib")
```

Download `bike_demand_model.joblib` from your notebook environment and place it beside `app.py`.

Important: upload the complete fitted Scikit-learn pipeline saved by the notebook. Do not upload only the HistGradientBoosting estimator.

## 2. Test locally (recommended)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown in the terminal, usually `http://localhost:8501`.

## 3. Publish using Streamlit Community Cloud

1. Create a GitHub repository, for example `bike-sharing-demand-prediction`.
2. Upload `app.py`, `requirements.txt`, and `bike_demand_model.joblib` to the repository root.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Select **Create app** and choose your repository, branch, and `app.py` as the main file.
5. Deploy. Streamlit will install `requirements.txt` and build the app.
6. Open the generated `*.streamlit.app` URL and test a prediction.

## Notes

- The app reads the category levels from the fitted OneHotEncoder in your saved pipeline. This keeps categorical choices aligned with the model's training data.
- The app builds the same 20 predictor columns used in the notebook, including the original `yr` value (2011 or 2012).
- Date input is limited to 2011–2012, the period represented in the notebook's training dataset.
- Temperature, feels-like temperature, humidity and wind speed use the normalized 0–1 scale used in the dataset.
- The notebook reports test MAE 49.83, RMSE 73.78, and R² 88.24%.
- This is a Streamlit-hosted app; a separate FastAPI service is not required.

Before publishing, check the Scikit-learn version used to train the artifact (`import sklearn; print(sklearn.__version__)`) and use that same version in `requirements.txt` if the cloud app reports a model-version compatibility warning.
