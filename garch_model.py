import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from arch import arch_model
plt.rcParams.update({'font.size': 20})

def garch_data(x, p, q, vol="GARCH"):
    garch = arch_model(x, p = p, q = q, 
                       mean = 'constant', 
                       vol = vol, 
                       dist = 'normal') # construction du modele
    garch_fit = garch.fit(update_freq = 4) # calibrage du modele 
    return garch_fit

ticker1 = "^FCHI" # CAC40

data = yf.download(ticker1, 
                   start=datetime(2010,1,1), 
                   end=datetime(2023,1,1))

data["Returns"] = 100 * (data["Close"].pct_change())
x = data["Returns"].dropna()

garch = garch_data(x, 1, 1)
garch1 = garch.conditional_volatility

plt.plot(x, linewidth=0.7, 
            color="grey", alpha=0.6, 
            label="volatilité réalisée")
plt.plot(garch1, linewidth=1, 
            color="red", 
            label="volatilité estimée")
plt.plot(- garch1, linewidth=1, color="red")
plt.grid(axis='y', linestyle='dotted', color="blue", lw=1)
plt.xlabel("Année")
plt.ylabel("Volatilité")
plt.legend()
plt.show()