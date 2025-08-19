# Solution for Data Premiere League 3rd Edition – Economic Forecasting and Resilience Modeling

This repository contains the code and methodology for our solution to the Data Premiere League 3rd Edition, focusing on economic forecasting and resilience modeling. The solution leverages advanced machine learning techniques, including hyperparameter tuning with Optuna, to generate accurate predictions.

## Prerequisites

*   A Google account
*   Access to Google Colab
*   The processed datasets (available from the provided drive link)

## Setup and Execution

Follow these steps to run the program and reproduce the results:

1.  **Upload Processed Datasets**: Download the `processed_datasets` folder from the provided drive link. Before running any code cells, upload these datasets to your Google Colab session environment (e.g., by dragging them into the file sidebar on the left).

2.  **Create a New Notebook**: Open [Google Colab](https://colab.research.google.com/) and create a new notebook.

3.  **Install Optuna**: The first step is to install the Optuna library for hyperparameter optimization. In the first cell of your Colab notebook, run:
    ```python
    !pip install optuna
    ```

4.  **Copy and Execute Code**: Copy the code from the provided solution files into individual cells in your Colab notebook.

5.  **View Output**: The results, including data previews, model training logs, evaluation metrics, and predictions, will be displayed directly below the corresponding code cells upon execution.

**Kindly refer the below given links for the stated purposes**
**For Processed Datasets** : [Google Drive](https://drive.google.com/drive/folders/1L0s1ud7wDYClk_p-8LW7Hqdl3hU6mnLT?usp=drive_link)
**Colab Link 1** : [Google Colab](https://colab.research.google.com/drive/1zfQE8qyCtDzZgqIgQupQGrcmj7lAc7I_?usp=sharing)
**Colab link 2** : [Google Colab](https://colab.research.google.com/drive/1WKcnPgky32jfdlz5058mgwGwmMZhVgeg?usp=sharing#scrollTo=H797T9koSZE6)

NOTE : Colab notebooks will take through our complete effort for this soultion, trying different models like RF and then how we arrived at LSTM (as discuused in depth below). Kindly run in T4 GPU to avoid errors. Training time might take from 10 - 30 minutes depending on various reasons.

## Data Preprocessing Overview

This section details the preprocessing steps applied to the raw datasets to prepare them for economic forecasting and resilience modeling.

### 1. Trade Data (Import/Export Files)

**Initial Processing:**
- The original import and export data were provided as two separate CSV files for each category.
- The first step involved merging these parts to create complete `import` and `export` datasets.

**Analysis and Transformation:**
- Each merged file was thoroughly analyzed to understand its structure, missing values, and potential for feature engineering.
- The data was processed to create final `export_processed.csv` and `import_processed.csv` files (available in the drive link).
- The key objective of this transformation was to reorganize the columns to calculate a **`TotalDependencyIndex`** for a given country in a given year. This index quantifies one country's economic dependency on another through trade.
- This structured organization was crucial for efficiently solving the complex queries required for the economic modeling tasks.

### 2. Core Economic Indicators Dataset

**Initial Format:**
- The dataset was provided in a time series format (wide format), with years as columns.

**Transformation to Normal Form:**
- The first major step was **unpivoting** the data to convert it into a long format (normal form). This resulted in a structure with columns like `[Country, Series_Code, Year, Value]`, which is essential for effective analysis and merging with other datasets.

**Handling Missing Values:**
- **Interpolation** was the primary method attempted to fill missing values. However, standard interpolation over the entire time series was deemed unsuitable for this economic data.
- A targeted approach was adopted:
    - Countries were sorted and processed individually.
    - For each country, the time period (2000-2024) was focused on.
    - Interpolation was applied with a `limit=1` to only fill gaps of a single missing year, preventing the creation of unrealistic, smoothly interpolated long-term trends.
- Other methods like forward-fill (`ffill`) and backward-fill (`bfill`) were tested but were found to be less appropriate for maintaining the integrity and consistency of the economic indicators.
- Some values were intentionally left as `NaN` to avoid introducing bias and to maintain consistency where data was truly unavailable. Imputed column was added in addition to shows the nature of the value present.

**Final Integration:**
- After cleaning and transforming, the data was grouped and aggregated based on the `Series_Code` and other relevant columns.
- The concept of economic dependency from the trade data was integrated. The **Herfindahl-Hirschman Index (HHI)** was calculated for both import and export partners for each country to measure trade concentration.
- This HHI, combined with the `TotalDependencyIndex` logic, was used to calculate an overall measure of a country's dependency on external trade partners. This synthesized feature was then merged into the core economic indicators dataset to enrich it for the forecasting models.This forms the integrated_master dataset.
### 3. Natural Disasters Dataset

**Data Cleaning and Filtering**
The raw disaster data was first reduced by removing numerous administrative and geographic columns not required for economic modeling. The records were then filtered to a focused scope: only the 25 target countries, events from the year 2000 onward, and exclusively recent (`Historic == 'No'`) and natural disasters. Key numeric columns for fatalities and financial impact were cleaned, with missing values set to zero. All cost figures reported in '000 US$ were converted to full US dollars for accurate analysis.

**Aggregation and Index Creation**
The event-level data was grouped by country and year. For each group, sums were calculated for totals like deaths and costs, while averages were computed for metrics like magnitude and event duration. From this aggregated data, new indices were engineered. A `severity_index` was created by combining magnitude and damage, and a `disaster_recovery_score` was derived from event duration and reconstruction costs to quantify resilience.

### 4. Population and Demographics Dataset

**Standardization and Reshaping**
The dataset was filtered for the 25 countries and the five key population elements. The 'Value' column, representing thousands of people, was converted to actual numbers. A quality filter was applied to retain only data flagged as official estimates (`Flag == 'X'`). The core processing step involved pivoting the long-form table into a wide format, creating distinct columns for each demographic metric indexed by country and year.

**Derived Metric Calculation**
Using the new wide-format table, foundational demographic ratios were computed. These included the urbanization percentage and the gender ratio. To analyze trends, year-over-year population growth was calculated. A simple categorical index was also created to flag countries with declining growth rates as having a higher potential ageing population.

### 5. Crop and Livestock Production Dataset

**Unit Standardization and Data Reconstruction**
The initial processing addressed heterogeneous unit reporting across the FAO dataset. Yield values underwent conversion to standardized kg/ha units, addressing entries reported in hg/ha (conversion factor: 0.1) and tonnes/ha (conversion factor: 1000). Area harvested data was normalized to hectares, with values originally reported in thousands of hectares scaled appropriately. The core computational logic implemented was: `estimated_production_tonnes = (yield_kg_ha * area_ha) / 1000`. Missing values were handled through group-wise linear interpolation (`method="linear"`) at the `['Area', 'Item']` granularity, preserving commodity-specific trends while maintaining numerical stability.

**High-Dimensional Aggregation and Anomaly Detection**
The transformation pipeline employed a weighted aggregation scheme to compute national-level statistics. The weighted average yield was calculated as `∑(yield_kg_ha * area_ha) / ∑(area_ha)` with null handling through `fillna(0)`. Feature engineering incorporated: 1) a food production index derived from yield-area product, 2) yield variability computed as a rolling standard deviation (`rolling(window=3, min_periods=1).std()`), and 3) crop diversity via `nunique()` aggregation. Quality controls included outlier detection (`yield_kg_ha > 10000`) and production discrepancy flags based on threshold comparisons between estimated and reported values.

### 6. Resilience Indicators Dataset

**Structural Transformation and Type Conversion**
The processing pipeline addressed the World Bank format through a melt operation with `id_vars=["Country Name", "Country Code", "Series Name", "Series Code"]`. Type consistency was enforced through regex-based year extraction (`str.extract(r"(\d{4})")` and coercive numeric conversion (`pd.to_numeric(..., errors="coerce")`). This transformed the matrix of year-columns into a structured observation table with dimensions: `[country, series, year, value]`.

**Hierarchical Imputation Strategy**
Missing value handling employed a multi-stage approach: primary interpolation used `series.interpolate(method="linear", limit=1)` for internal gaps, followed by `ffill()`/`bfill()` for terminal missing values. The implementation maintained data provenance through boolean flags (`Imputed`) tracking original null positions that underwent modification, ensuring auditability in downstream economic modeling.

### 7. Employment and Unemployment Dataset

**Long-Format Transformation**
The dataset underwent similar structural transformation as other World Bank indicators via `melt()` operation. Critical data cleaning included replacement of placeholder strings (`".." → np.nan`) and type conversion of both year and value columns to machine-readable formats (integer and float respectively).

**Conservative Imputation Protocol**
Reflecting the sensitivity of labor market data, the imputation strategy was restricted to linear interpolation with `limit=1`, explicitly avoiding forward/backward filling for longer gaps. This preserved null patterns in periods of structural labor market shifts, preventing artificial smoothing of economically significant discontinuities.

### 8. Social and Welfare Indicators Dataset

**Indicator-Specific Imputation Rules**
The processing system implemented a rule-based framework through the `rules_for(series_code)` function, applying distinct interpolation policies based on indicator characteristics:
- **High-smoothness series** (e.g., `SP.DYN.LE00`): `limit=3` with edge filling
- **Survey-based metrics** (e.g., `SI.POV.GINI`): `limit=0` (no imputation)
- **Default policy**: `limit=1` without edge filling

**Provenance Tracking System**
The implementation advanced beyond binary imputation flags by capturing specific methods (`Impute_Method`) in {"original", "interp", "ffill", "bfill"}, providing granular metadata for sensitivity analysis in subsequent econometric modeling.

The code snippets used for preprocessing the dataset are given just for reference, you need not run that anywhere. 

# Model Selection and Training Approach

## Overview
Since trade effects unfold temporally, the model needed to capture temporal dependencies rather than treating data points as independent.

## Why LSTM
We selected **Long Short-Term Memory (LSTM)** networks because they are designed to handle sequence data. Trade and GDP indicators such as export ratios, import dependency, and growth volatility are not independent across years; instead, their past values influence future trends. Also, the final merged logical thinking of having 25 (Countries) x 25 (Years from 2000 - 2024) ~ 625, made it look like LSTM can be preferred as the deep learning model.  

Traditional models like Random Forests and Gradient Boosting Trees were tested. While they performed reasonably on static snapshots, they failed to capture the evolving, time-linked patterns. Similarly, classical econometric models (e.g., ARIMA, SARIMAX) were considered but showed limitations in handling the multivariate nature and nonlinear dependencies in trade features.  

**Kindly contact the contributors of this repo to know more abou the models that were tried in prior to LSTM.**

The **key reasons for choosing LSTM** were:
- Ability to **learn temporal dependencies** (e.g., how shocks propagate over years).
- Support for **multivariate input features**, not just univariate sequences.
- Better generalization to **counterfactual simulations**, where disruptions in one year affect future GDP outcomes.

## Dataset Split
We used a **70:30 train-test split**, ensuring enough data for training while keeping sufficient unseen samples for evaluation. This ratio balanced the relatively small dataset size with the need for reliable generalization testing. The year based split can be analysed in the code snippets provided.

## Hyperparameter Tuning
To ensure robust performance, we used **Optuna** for automated hyperparameter tuning. The tuning process systematically explored the search space to minimize mean squared error on validation sets. Parameters tuned included:
- **Sequence length** (years of historical data used as context).  
- **Hidden units** and **number of layers**, balancing model capacity with overfitting risk.  
- **Dropout rates** for regularization.  
- **Learning rate** of the optimizer, which strongly influenced convergence stability.  
- **Batch size**, impacting both speed and generalization.  

This process was essential. Short sequence lengths lost long-term information, while larger models often overfit given the dataset size (~600 rows). The tuning identified compact but effective architectures that achieved the best trade-off between accuracy and generalization. **Using Optuna nearly 20 trials were considered and out of the 20 the best trial was printed as the output** 

## Alternatives Considered
We experimented with several approaches before converging on LSTM:
- **Tree-based models** (Random Forest, Gradient Boosting): fast and interpretable, but unable to forecast sequential dynamics.  
- **Classical econometric models** (ARIMA, SARIMAX): well-suited to univariate forecasting, but insufficient for high-dimensional trade features. While trying with ARIMA, it was found that the data was not sequential. **The p-value for Canada turned out to be grater than 0.5 nearing 0.75. Although d was set to 1 to make it sequential. BUt LSTM suited better than a univariate approach.** 
- **Feed-forward neural networks**: handled static data well but performed poorly in simulating multi-year impacts.  

In summary, the **LSTM architecture, combined with careful hyperparameter tuning**, proved most suitable for modeling the complex, sequential, and interdependent nature of trade and GDP dynamics in this dataset.
