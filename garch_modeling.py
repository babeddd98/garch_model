import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from arch import arch_model
from scipy.stats import probplot
plt.rcParams.update({'font.size': 20})

def garch_data(x, p, q, vol="GARCH"):
    garch = arch_model(x, p = p, q = q, mean = 'constant', vol = vol, dist = 'normal') # construction du modèle
    garch_fit = garch.fit(update_freq = 4) # recherche des paramètres optimaux
    return garch_fit

def simulate_GARCH(data, omega, alpha, beta = 0):
    
    resid = np.zeros_like(data)
    variance = np.zeros_like(data)
    variance[0] = data[0]
    resid[0] = data[0]
    for i in range(1, len(data)):
        # Simulate the variance (sigma squared)
        variance[i] = omega + alpha * resid[i-1]**2 + beta * variance[i-1]
        # Simulate the residuals
        resid[i] = np.sqrt(variance[i]) * data[i]    
    
    return resid, variance

ticker1 = "^FCHI" # CAC40
ticker2 = "^VIX"

data = yf.download(ticker1, start=datetime(2010,1,1), end=datetime(2023,1,1))
# vix = yf.download(ticker2, start=datetime(2010,1,1), end=datetime(2023,1,1))

# data["Returns"] = 100 * np.abs(data["Close"].pct_change())
data["Returns"] = 100 * (data["Close"].pct_change())

x = data["Returns"].dropna()

# vix["Returns"] = 100 * np.abs(vix["Close"].pct_change())
# vix = vix["Returns"].dropna()

garch = garch_data(x, 1, 1)
garch1 = garch.conditional_volatility



# garch1 = garch.forecast(horizon=10)

    

# garch = garch_data(x, 1, 1,'ARCH')
# garch2 = garch.conditional_volatility

plt.plot(x, linewidth=0.7, color="grey", alpha=0.6, label="volatilité réalisée")
plt.plot(garch1, linewidth=1, color="red", label="volatilité estimée")
plt.plot(- garch1, linewidth=1, color="red")
# plt.plot(vix, linewidth=1, color="green", alpha=0.2, label="VIX")
# plt.plot(garch2, linewidth=0.5, color="blue", alpha=0.5)

plt.grid(axis='y', linestyle='dotted', color="blue", lw=1)
plt.xlabel("Année")
plt.ylabel("Volatilité")
# plt.title("Estimation de la volatilité du CAC40 par le modèle GARCH(1,1)")
plt.legend()

# probplot(x, plot=plt)
# probplot(garch1, plot=plt)

plt.show()