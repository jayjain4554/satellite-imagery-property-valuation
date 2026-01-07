# Satellite Imagery–Based Property Valuation

**Multimodal Regression using Tabular Data and Satellite Images**

## Overview

This project explores whether satellite imagery can add value to traditional real estate price prediction models. Property prices are usually predicted using structured attributes such as square footage, number of rooms, construction quality, and location. While these features are strong predictors, they do not explicitly capture environmental and neighborhood context.

To address this, I build a **multimodal regression pipeline** that combines:

* **Tabular property features**, and
* **Satellite imagery** fetched using latitude and longitude coordinates.

The project compares a strong **tabular-only baseline** against a **CNN + tabular multimodal model**, and further analyzes the role of satellite images using visual explainability (Grad-CAM).

---

## Dataset

The base dataset consists of residential housing records with the following key attributes:

* **Target**

  * `price`

* **Structural Features**

  * `bedrooms`, `bathrooms`
  * `sqft_living`, `sqft_above`, `sqft_basement`
  * `floors`, `condition`, `grade`

* **Neighborhood Features**

  * `sqft_living15`, `sqft_lot15`
  * `view`, `waterfront`

* **Geographic Features**

  * `lat`, `long`

Satellite images are programmatically downloaded using latitude and longitude values. Due to partial availability of imagery, a clean subset of properties with valid satellite images is used for multimodal training.

---

## Satellite Image Collection

Satellite images are downloaded using the **ESRI World Imagery API**.
For each property, a small bounding box around its coordinates is used to capture local neighborhood context.

* Image format: PNG
* Resolution: 256 × 256
* Separate folders for train and test images

The image downloading logic is implemented in:

```
data_fetcher.py
```

---

## Exploratory Data Analysis (EDA)

Key findings from EDA include:

* Property prices show **strong right skew**, motivating a log transformation of the target variable.
* Features such as `sqft_living`, `grade`, and neighborhood averages (`sqft_living15`) are highly correlated with price.
* Spatial plots of latitude and longitude reveal **clear geographic clustering of high-value properties**, justifying the inclusion of satellite imagery for contextual information.

---

## Preprocessing

The preprocessing pipeline includes:

* Log transformation of the target: `log(price + 1)`
* Standardization of numerical features
* Strict alignment between tabular rows and satellite images
* Geospatial feature engineering using **KMeans clustering** on latitude and longitude
* Train–validation split while preserving correct image–tabular mapping

All preprocessing steps are implemented in:

```
preprocessing.ipynb
```

---

## Models

### 1. Tabular-Only Baseline

A strong tabular baseline is built using **Histogram-based Gradient Boosting**, which is well-suited for structured data and captures non-linear interactions effectively.

This model serves as a benchmark to evaluate whether satellite imagery provides meaningful improvements.

---

### 2. Multimodal Model (Tabular + Satellite Images)

The multimodal architecture consists of:

* **Image Branch**

  * Pretrained ResNet-18 CNN
  * CNN weights frozen during training to prevent overfitting
  * Extracts high-level visual embeddings from satellite images

* **Tabular Branch**

  * Multi-layer perceptron (MLP) processing standardized tabular features

* **Fusion Layer**

  * Concatenation of image and tabular embeddings
  * Fully connected regression head predicting log-price

Model training and evaluation are implemented in:

```
model_training.ipynb
```

---

## Training Strategy

* Loss function: Mean Squared Error (MSE) in log-price space
* Optimizer: Adam / AdamW
* Learning rate scheduling based on validation loss
* Early stopping used to prevent overfitting
* CNN backbone kept frozen to prioritize tabular learning

---

## Evaluation and Results

All models are evaluated in **real price space** by converting predictions back from log-scale.

### Performance Summary

| Model                     | Data Used        | R²         | RMSE (Price) |
| ------------------------- | ---------------- | ---------- | ------------ |
| Tabular Gradient Boosting | Tabular only     | ~0.89      | ~132,000     |
| Multimodal CNN + MLP      | Tabular + Images | ~0.63      | Higher       |

**Key takeaway:**
Tabular features already explain most of the price variance. Satellite imagery does not significantly improve accuracy but adds interpretability.

---

## Explainability with Grad-CAM

Grad-CAM is applied to the CNN branch of the multimodal model to visualize which regions of satellite images influence predictions.

Observations:

* Green cover and open spaces are emphasized in suburban areas
* Road networks and infrastructure are highlighted in dense urban regions
* High-value neighborhoods show attention to organized layouts

This confirms that satellite imagery provides **contextual insights**, even when it does not outperform tabular models in raw accuracy.

---

## Final Predictions

The final submission file is generated using the **best-performing tabular model** for accuracy.

```
final_price_predictions.csv
```

Format:

```
id,predicted_price
```

---

## Project Structure

```
.
├── preprocessing.ipynb
├── model_training.ipynb
├── final_predictions.ipynb
├── data_fetcher.py
├── final_price_predictions.csv
├── requirements.txt
└── README.md
```

Raw data and satellite images are excluded from the repository for size and reproducibility reasons.

---

## Key Learnings

* Strong tabular features can dominate predictive performance in real estate valuation.
* Multimodal models are valuable for **interpretability**, not always accuracy.
* Careful data alignment is critical in multimodal learning.
* Proper evaluation in real price space is essential for meaningful RMSE comparisons.

---

## Future Work

* Use higher-resolution or multi-spectral satellite imagery
* Incorporate temporal price trends
* Explore attention-based fusion mechanisms
* Add external geographic features (distance to amenities, transport hubs)

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Author

**Jay Jain**
Department of Chemical Engineering
IIT Roorkee

---

