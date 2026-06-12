# Jet Engine Remaining Useful Life Prediction Using NASA C-MAPSS Data

## Project Overview

This project uses the NASA C-MAPSS turbofan engine degradation dataset to predict the **Remaining Useful Life (RUL)** of simulated turbofan engines. RUL represents the estimated number of operating cycles an engine has left before failure. The project began as a basic regression model, but it became an iterative model-development project. I started with tree-based machine learning models, analyzed where they succeeded and failed, then improved the pipeline by adding time-dependent features, testing different RUL caps, and exploring sequence-based modeling with an LSTM and ensemble approach. The main goal was not just to get a model score. The goal was to understand how sensor data can be used for predictive maintenance decisions.

---

## Why This Project Matters

Predictive maintenance is important in aerospace engineering because engines and other complex systems need to be monitored before failure occurs. Instead of waiting for a failure, sensor data can be used to estimate degradation and help maintenance teams decide when inspection or repair should be prioritized.

This project asks:
- Given an engine's sensor history and operating conditions, can a model estimate how many cycles remain before failure?

---

## Dataset

This project uses the NASA C-MAPSS turbofan engine degradation simulation dataset. The FD001 subset was used first because it contains one operating condition and one fault mode, making it a good starting point for model comparison.

Main files used:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Each row contains:
- Engine unit number
- Time cycle
- Three operating settings
- Twenty-one sensor measurements

The dataset is not included in this repository. To run the notebook, download the NASA C-MAPSS data and upload `CMAPSSData.zip` into Google Colab.

---

## What Is RUL?

**RUL** means **Remaining Useful Life**.

For the training data, each engine runs until failure. RUL is calculated as:

```text
RUL = final cycle of engine - current cycle
```

Example:

| Current Cycle | Final Failure Cycle | RUL |
|---:|---:|---:|
| 50 | 192 | 142 |
| 100 | 192 | 92 |
| 180 | 192 | 12 |
| 192 | 192 | 0 |

Because very healthy engines can look similar in the sensor data, I tested different RUL caps:

```text
RUL_CAP = 100
RUL_CAP = 125
RUL_CAP = 150
```

Capping RUL reduces the effect of large early-life values and helps the model focus more on degradation behavior closer to failure.

---

## Project Progression

### Stage 1: Baseline Tree-Based Models

I started with four regression models:
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- XGBoost Regressor

These models were trained to predict RUL from engine settings, sensor values, and engineered features. The initial model comparison showed that all four models followed the general RUL trend, especially for engines close to failure. However, they also produced similar errors in the same regions.  This was an important finding because it meant that the issue was probably not just the choice of algorithm. The models were likely limited by how the degradation history was represented.

Sorted all models RUL comparison<img width="5400" height="2400" alt="sorted_all_models_rul_comparison" src="https://github.com/user-attachments/assets/56a326e9-af24-4296-99e8-e378d7b8cfb4" />

### Stage 2: Error Analysis by Engine

I then compared the absolute prediction error for each model across all test engines. The error plot showed that the models often failed on the same engines. That suggested that the tree-based models were learning from similar information and had similar limitations.

Model error comparison by engine<img width="5400" height="2400" alt="all_models_error_comparison" src="https://github.com/user-attachments/assets/53a6f4cb-062e-487e-ae73-2e1f7f831ce9" />

This changed the direction of the project. Instead of simply adding more models, I focused on improving the representation of engine degradation over time.

### Stage 3: Time-Dependent Feature Engineering

Engine degradation is a time-dependent process. A single sensor value may not fully describe engine health, but the recent trend of that sensor can be more informative.

To capture this, I added time-dependent features:
- Rolling sensor means over recent cycles
- Rolling sensor standard deviations
- Cycle-to-cycle sensor differences
- Sensor slopes over a 10-cycle window
- Deviation from each engine's early healthy baseline

These features help represent not only the current sensor state, but also the direction and variability of sensor behavior.

### Stage 4: Testing Different RUL Caps

I tested different RUL caps to determine whether the label structure affected performance:

```text
100, 125, 150
```

This was important because high-RUL engines often appear healthy and are harder to distinguish. Testing multiple caps allowed me to evaluate whether compressing high-RUL values improved prediction accuracy.

### Stage 5: LSTM Sequence Model

After observing that tree-based models produced similar patterns, I tested a fundamentally different architecture by using an LSTM neural network. Instead of using one row or engineered summary features, the LSTM used the previous 30 cycles of sensor and operating data:

```text
Previous 30 cycles of sensor data → predicted RUL
```

This changed the modeling approach from a tabular prediction problem to a sequence-based prediction problem. The purpose was to test whether a model that directly learns from sensor history could outperform models using manually engineered time-based features.

### Stage 6: Ensemble Model

Finally, I created an ensemble by averaging the best tree-based model and the LSTM prediction:

```text
Ensemble prediction = 0.5 × best tree model prediction + 0.5 × LSTM prediction
```

The reason for trying an ensemble was that tree models and LSTM models may make different types of errors. If their errors are different enough, combining them can produce a more stable prediction.

The graph below compares the best tree model, LSTM, and ensemble model against the actual NASA RUL values.

<img width="5400" height="2400" alt="advanced_model_comparison_sorted" src="https://github.com/user-attachments/assets/cd9436f9-8de1-49bc-8d41-e1463fb94e95" />

The graph below compares the absolute prediction error for each final model across all test engines.

<img width="5400" height="2400" alt="advanced_model_error_comparison" src="https://github.com/user-attachments/assets/1d0e23c1-510c-47c7-82dd-3e201d63c0e9" />

---

## Evaluation Metrics

The models were evaluated using:
- **Mean Absolute Error (MAE):** average prediction error in cycles
- **Root Mean Squared Error (RMSE):** error metric that penalizes larger mistakes more strongly
- **R² score:** how well predictions explain variation in true RUL
- **Risk category accuracy:** whether the model predicts the correct maintenance risk group

Risk categories were defined as:

| RUL Range | Risk Category |
|---:|---|
| 0-30 cycles | High Risk |
| 31-60 cycles | Moderate Risk |
| 61+ cycles | Low Risk |

MAE was treated as the main metric because it directly answers:
- On average, how many cycles was the model off by?

---

## Key Observations From the Graphs

The initial graphs showed three important things:
1. All four tree-based models tracked low-RUL engines reasonably well.
2. The models became less stable in mid-to-high RUL regions.
3. The models often failed in the same places, suggesting that feature representation was a major limitation.

This led to the main learning point of the project:
- Improving predictive maintenance models is not only about trying more algorithms. It is also about representing degradation history in a way the model can learn from.

---

## What I Learned

This project taught me several important lessons about machine learning for aerospace predictive maintenance.

First, I learned that RUL prediction is a regression problem, but the engineering meaning behind the prediction matters. The number is useful only if it supports a maintenance decision. Second, I learned that different models can produce similar results if they are given similar input representations. Random Forest, Gradient Boosting, Extra Trees, and XGBoost are different algorithms, but they all struggled in similar areas because they were learning from similar tabular features. Third, I learned that engine degradation is time-dependent. A single sensor snapshot does not always explain how close an engine is to failure. Rolling averages, sensor variability, slopes, and baseline deviations can help the model understand how an engine is changing over time. Fourth, I learned that more complex models do not automatically guarantee better performance. The LSTM was a fundamentally different sequence-based approach, but it still needed to be compared carefully against the actual NASA RUL values. Finally, I learned that predictive maintenance models should be evaluated not only by numerical error, but also by whether they classify maintenance risk correctly.

---

## Limitations

This project uses the FD001 subset of a simulated NASA dataset. It should not be interpreted as a real-world aircraft engine failure predictor. High-RUL engines are inherently harder to predict because they often look healthy and may not show strong degradation signals. Some uncertainty is unavoidable when sensor data does not clearly distinguish between engines that are far from failure. The project demonstrates a predictive maintenance modeling workflow, but real deployment would require additional validation, more operating conditions, domain expertise, and safety-critical testing.

---

## Future Improvements

Future improvements could include:
- Testing FD002, FD003, and FD004 subsets
- Hyperparameter tuning for XGBoost and LSTM
- Trying GRU or temporal convolutional networks
- Adding SHAP analysis for sensor-level interpretability
- Building a Streamlit dashboard
- Testing different LSTM sequence lengths
- Creating a more advanced maintenance assistant interface

---

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- TensorFlow/Keras
- ChatGPT
- Claude
- Copilot

---

## How to Run

1. Open the Colab notebook in the notebooks/ folder.
2. Upload CMAPSSData.zip when prompted.
3. Run all cells in order.
4. Review the model comparison tables and generated graphs.
5. Use the engine review functions to analyze individual engines.
