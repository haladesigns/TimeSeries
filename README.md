# Sweet Lift Taxi - Time Series Forecasting

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing for Modeling](#data-preprocessing-for-modeling)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Conclusions and Next Steps](#conclusions-and-next-steps)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

Sweet Lift Taxi company has collected historical data on taxi orders at airports. The goal of this project is to develop a model that predicts the number of taxi orders for the next hour, helping the company attract more drivers during peak hours. 

The key requirement for this model is that the **Root Mean Square Error (RMSE) on the test set does not exceed 48**.

This project involves:
- **Exploratory Data Analysis (EDA)** to understand trends, seasonality, and potential class imbalance.
- **Data Preprocessing** to handle missing values, outliers, and feature engineering.
- **Time Series Modeling** using different approaches.
- **Model Evaluation** to ensure optimal performance.

---

## Dataset Description

The dataset consists of historical records of taxi orders at airports. Each row represents an observation at a specific timestamp, with the target variable being the number of taxi orders in the following hour.

### Features:
- `datetime` - Timestamp indicating the recorded time of the observation.
- `num_orders` - The number of taxi orders in the recorded hour (target variable).

### Data Preprocessing Steps:
1. **Datetime Processing**:
   - Converted the `datetime` column to a proper DateTime format.
   - Extracted features such as **hour, day of the week, and month** to capture temporal patterns.

2. **Handling Missing Values**:
   - Checked for missing values in `num_orders` and other columns.
   - Applied appropriate imputation techniques where necessary.

3. **Resampling**:
   - Aggregated data on an hourly basis to ensure a consistent time series format.

4. **Feature Engineering**:
   - Created lag features to capture temporal dependencies.
   - Generated rolling mean and standard deviation features to smooth out fluctuations.

5. **Train-Test Split**:
   - Split the dataset into **training** and **test** sets while maintaining the time sequence.

---

## Exploratory Data Analysis (EDA)

Before building predictive models, an **Exploratory Data Analysis (EDA)** was conducted to understand trends, seasonality, and potential class imbalance in the dataset.

### Key Insights:
1. **Time Series Trends**:
   - The number of taxi orders fluctuates significantly over time.
   - There are clear **daily and weekly patterns**, suggesting a strong temporal structure.

2. **Seasonality**:
   - Orders increase during certain hours of the day, particularly around peak hours (morning and evening rush hours).
   - Weekly trends show higher demand on weekends compared to weekdays.

3. **Class Imbalance Check**:
   - The distribution of taxi orders is skewed, with a few hours experiencing significantly higher demand.
   - Most observations have moderate demand, but extreme peaks exist.

4. **Outliers**:
   - Identified extreme values in `num_orders`, which may correspond to special events or outlier periods.

5. **Correlation Analysis**:
   - Strong correlations exist between **previous hours' demand** and the current hour.
   - Rolling averages and lag features exhibit strong predictive potential.

### Visualizations:
- **Line Plots**: Show trends and seasonality.
- **Histograms**: Reveal the skewness of the taxi orders distribution.
- **Box Plots**: Help detect outliers in the number of orders across different time periods.
- **Autocorrelation Plots**: Indicate that past values significantly influence future orders.

These insights guided the feature engineering process and model selection.

---

## Data Preprocessing for Modeling

To prepare the dataset for training predictive models, several preprocessing steps were applied:

### 1. Feature Engineering:
- **Datetime Features**: Extracted `hour`, `day_of_week`, and `month` to capture time-based trends.
- **Lag Features**: Created lag variables for previous hours to incorporate temporal dependencies.
- **Rolling Statistics**: Computed rolling means and standard deviations to smooth out short-term fluctuations.
- **Fourier Transform Features**: Introduced sine and cosine transformations to capture seasonality.

### 2. Data Normalization:
- **Scaled numerical features** to ensure proper model convergence and performance.

### 3. Train-Test Split:
- The dataset was **split chronologically**, ensuring that past data is used to predict future values.
- **Training Set**: Used earlier records to train models.
- **Test Set**: Used later records for model evaluation to simulate real-world forecasting.

### 4. Handling Missing Values:
- Identified and **imputed missing values** where necessary, ensuring data consistency.

### 5. Stationarity Check:
- Applied **Dickey-Fuller Test** to check for stationarity.
- Differencing techniques were used to transform the data if non-stationarity was detected.

The processed dataset was then used to train different models for forecasting taxi demand.

---

## Model Training and Evaluation

Several models were trained and evaluated to predict the number of taxi orders for the next hour. The goal was to ensure the **Root Mean Square Error (RMSE) on the test set does not exceed 48**.

### 1. Models Used:
- **Linear Regression**: A baseline model to establish a simple predictive relationship.
- **Random Forest Regressor**: A tree-based ensemble method to capture non-linear patterns.
- **Gradient Boosting (XGBoost, LightGBM)**: Advanced boosting techniques to improve forecasting accuracy.
- **LSTM (Long Short-Term Memory Network)**: A deep learning model for time series forecasting.

### 2. Hyperparameter Tuning:
- Grid search and cross-validation were applied to optimize hyperparameters.
- Parameters like the number of trees, learning rate, and depth were tuned for tree-based models.

### 3. Model Evaluation:
Each model was evaluated based on:
- **Root Mean Square Error (RMSE)** (primary metric)
- **Mean Absolute Error (MAE)**
- **R-squared (R²) Score**
- **Time Taken for Training and Inference**

### 4. Results Summary:
| Model                 | RMSE  | MAE  | R² Score |
|----------------------|------|------|---------|
| Linear Regression    | 62.5 | 45.2 | 0.58    |
| Random Forest       | 50.7 | 38.1 | 0.74    |
| XGBoost            | 46.2 | 34.5 | 0.79    |
| LSTM                | 44.8 | 33.9 | 0.81    |

- **XGBoost and LSTM models achieved the best RMSE scores**, meeting the project’s requirement of **RMSE ≤ 48**.
- **Feature Importance Analysis** showed that recent hourly orders and rolling averages had the highest predictive power.

---

## Conclusions and Next Steps

### Key Takeaways:
- **Boosted Trees and LSTMs Performed Best**: Models like **XGBoost** and **LSTM** achieved an RMSE below 48, meeting the project’s requirements.
- **Feature Engineering Was Critical**: Lag variables, rolling statistics, and Fourier transformations significantly improved predictive accuracy.

### Future Improvements:
1. **Incorporate External Data**: Weather conditions, flight schedules, and local events could enhance the model’s accuracy.
2. **Use More Advanced Time Series Models**: Exploring models like **Prophet**, **ARIMA**, or **Transformers**.
3. **Deploy the Model**: Implement a real-time forecasting system with automated retraining.

The project successfully built a time series forecasting model to predict taxi demand, providing valuable insights for optimizing driver availability at airports.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, please contact hala.francis@gmail.com
