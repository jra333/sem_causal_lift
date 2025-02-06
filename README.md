# Structural Equation Modeling (SEM) Analysis App

An interactive web application built with [Streamlit](https://streamlit.io/) that enables users to perform Structural Equation Modeling (SEM) on their data. The app integrates data upload, preprocessing (with time series support), dynamic model specification, SEM analysis using [semopy](https://github.com/semopy/semopy), and AI-generated interpretations via Google Gemini.

## Features

- **Data Upload:** Easily upload CSV files containing your dataset.
- **Data Preprocessing:** 
  - Select and convert a date column.
  - Aggregate and prepare time series data.
  - Optionally include lag features and time-based attributes (month, quarter, trend).
- **Dynamic SEM Model Specification:** 
  - Generate a default SEM model based on selected metrics.
  - Customize the relationships between variables using a simple syntax (e.g., `dependent ~ coefficient*independent`).
- **SEM Analysis:** 
  - Run SEM using the `semopy` library.
  - Standardize data and extract parameter estimates.
  - Display model fit measures with descriptions.
- **AI Interpretation:** 
  - Generate concise interpretations of your SEM results using Google Gemini AI (requires a valid API key).
- **Results Download:** 
  - Download the parameter estimates and model fit measures as CSV files.

## App Usage

- **Upload CSV File:** Upload your CSV file containing the dataset.
- **Select Date and Metrics:**
    - Choose the date column.
    - Select at least two numeric metrics for SEM analysis.
- **Data Preparation:**
    - Choose the frequency (`W` for weekly or `M` for monthly).
    - Optionally, include lag features and additional time features.
- **Model Specification:**
    - Use the default model generated from your selected metrics or customize the specification.
    - The model specification should follow the format: `dependent ~ coefficient*independent`.
- **Run SEM Analysis:**
    - Click the "Run SEM Analysis" button to execute the analysis.
- **AI-Generated Interpretation:**
    - Provide a valid Gemini API key to generate an AI-based interpretation of the SEM results.
- **Download Results:**
    - Download the parameter estimates and model fit measures using the provided download buttons.

## Impact Analysis Notebook

In addition to the SEM Analysis App, this repository includes an advanced Jupyter Notebook for integrated impact analysis:
  
- **Notebook Path:** `sem_notebooks/ccl_impact_analysis_modeling.ipynb`
- **Purpose:**
  - Performs comprehensive data aggregation (daily, weekly, and monthly) on raw sales data.
  - Implements advanced visualization techniques including time series plots, correlation heatmaps, and channel comparison plots.
  - Conducts integrated causal impact analysis combining Random Forest and OLS regression approaches.
  - Provides detailed elasticity analysis with statistical inferences and seasonal impact assessments.

## Configuration

To enable AI-generated interpretations in the SEM Analysis App, enter your Gemini API key in the provided text field within the app interface.  
You can obtain the API key from [Google Generative AI](https://makersuite.google.com/app/apikey).

## Troubleshooting

- **Data Issues:** Ensure your CSV file is correctly formatted and contains a valid date column.
- **Metric Selection:** Verify that at least two metrics are selected for SEM analysis.
- **Model Specification:** Follow the required syntax for model specifications. Refer to error messages in the app for guidance.
- **API Key:** Make sure your Gemini API key is valid for generating AI interpretations.
