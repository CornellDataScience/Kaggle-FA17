# Corporacion Favorita Supermarket Challenge

<img src="https://upload.wikimedia.org/wikipedia/commons/0/0f/Corporaci%C3%B3n_Favorita_Logo.png" width="50%">

### Overview
We are given historical data for brick-and-mortar sales for Corporacion Favorita, an Ecuadorian supermarket chain. We want to predict new sales quantities for new dates, given a tuple of (item, store, date). Submissions are evaluated based on a modified RMSE method, where perishables are given slightly more weight.

### Exploratory Analysis

[Processing and looking at the data](./Jo/jo_eda.ipynb)

### Current Models
- [Weekday Mean Baseline](./weekday_sales_mean.py)
- [Weekday Adjusted Mean Baseline](./weekday_adjusted_mean_baseline.py)

### Planned Models
- LSTM based model
    - Per item?
    - Per store?
    - Singular model?
- ARIMA model
    - Per item (possible, kernel exists)

### Challenges
- Combinatoric nature of possible entries means that the training file is very large - need to find the most efficient way to group or process entries
- Finding relationships between the seemingly "extra" data - oil prices, holidays etc, and figuring out if any are actually useful for analysis

### Reference
- [Sales Prediction with Time Series Modeling](http://cs229.stanford.edu/proj2015/219_report.pdf)

