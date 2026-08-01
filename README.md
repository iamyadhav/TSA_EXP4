# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 28/7/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load Dataset
data = pd.read_csv("House_Price.csv")

# Average Price by YearBuilt
data = data.groupby("YearBuilt")["Price"].mean().reset_index()

X = data["Price"]

# ---------------- Original Data ----------------

plt.figure(figsize=(10,5))
plt.plot(X)
plt.title("Original Data")
plt.grid(True)
plt.show()

# =====================================================
# ARMA(1,1)
# =====================================================

ar1 = np.array([1, -0.6])
ma1 = np.array([1, 0.4])

arma11 = ArmaProcess(ar1, ma1)
sample1 = arma11.generate_sample(nsample=1000)

plt.figure(figsize=(10,5))
plt.plot(sample1)
plt.title("SIMULATED ARMA(1,1) PROCESS")
plt.grid(True)
plt.show()

# Partial Autocorrelation
plot_pacf(sample1, lags=30, method='ywm')
plt.title("Partial Autocorrelation")
plt.show()

# Autocorrelation
plot_acf(sample1, lags=30)
plt.title("Autocorrelation")
plt.show()

# =====================================================
# ARMA(2,2)
# =====================================================

ar2 = np.array([1, -0.75, 0.25])
ma2 = np.array([1, 0.65, 0.35])

arma22 = ArmaProcess(ar2, ma2)
sample2 = arma22.generate_sample(nsample=1000)

plt.figure(figsize=(10,5))
plt.plot(sample2)
plt.title("SIMULATED ARMA(2,2) PROCESS")
plt.grid(True)
plt.show()

# Partial Autocorrelation
plot_pacf(sample2, lags=30, method='ywm')
plt.title("Partial Autocorrelation")
plt.show()

# Autocorrelation
plot_acf(sample2, lags=30)
plt.title("Autocorrelation")
plt.show()
```
## OUTPUT:

Original data
<img width="905" height="418" alt="image" src="https://github.com/user-attachments/assets/88fd6c0d-aa08-461e-a8b3-f40e91e2ba04" />


SIMULATED ARMA(1,1) PROCESS:
<img width="831" height="423" alt="image" src="https://github.com/user-attachments/assets/8bb845e7-94f1-4cfc-8244-67e5abef6a8b" />





Partial Autocorrelation
<img width="1032" height="496" alt="image" src="https://github.com/user-attachments/assets/ec09430d-dbd4-4094-9aac-d4144a13d6eb" />

Autocorrelation
<img width="1070" height="495" alt="image" src="https://github.com/user-attachments/assets/ef9442ea-7875-4a85-ab32-e06a05ac4791" />



SIMULATED ARMA(2,2) PROCESS:
<img width="887" height="422" alt="image" src="https://github.com/user-attachments/assets/10d00b55-5470-470a-96e1-fdb9b733b891" />

Partial Autocorrelation

<img width="993" height="493" alt="image" src="https://github.com/user-attachments/assets/0926ef46-d302-4ef4-9fae-0a86e3b2506f" />


Autocorrelation
<img width="1109" height="504" alt="image" src="https://github.com/user-attachments/assets/2bc7ca8d-8380-447a-9eae-cc9f7ff0a720" />

### RESULT:


Thus, a python program is created to fir ARMA Model successfully.
