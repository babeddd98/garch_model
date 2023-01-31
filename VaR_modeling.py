import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from arch import arch_model
import pandas as pd
import numpy as np

ticker1 = "^FCHI" # CAC40

data = yf.download(ticker1, 
                   start=datetime(2010,1,1), 
                   end=datetime(2023,1,1))

data["Returns"] = 100 * (data["Close"].pct_change())
x = data["Returns"].dropna()


am = arch_model(x , p = 1, q = 1, mean='constant', vol = 'EGARCH', dist = 't')
res = am.fit(update_freq=10)

forecasts = res.forecast(start='2019-01-01')
cond_mean = forecasts.mean['2019':]
cond_var = forecasts.variance['2019':]
q = am.distribution.ppf([0.01, 0.05], res.params[4])

value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * q[None, :]
value_at_risk = pd.DataFrame(value_at_risk, columns=['1%', '5%'], index=cond_var.index)

ax = value_at_risk.plot(legend=False, figsize=(12,6))
xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])

rets_2019 = x['2019':]


c = []
for idx in value_at_risk.index:
    if rets_2019[idx] > -value_at_risk.loc[idx, '5%']:
        c.append('#000000')
    elif rets_2019[idx] < -value_at_risk.loc[idx, '1%']:
        c.append('#BB0000')
    else:
        c.append('#BB00BB')
        
c = np.array(c, dtype='object')

labels = {
    
    '#BB0000': '1% Exceedence',
    '#BB00BB': '5% Exceedence',
    '#000000': 'No Exceedence'
}

markers = {'#BB0000': 'x', '#BB00BB': 's', '#000000': 'o'}

for color in np.unique(c):
    sel = c == color
    ax.scatter(
        rets_2019.index[sel],
        -rets_2019.loc[sel],
        marker=markers[color],
        c=c[sel],
        label=labels[color])
    
ax.set_title('Parametric VaR')
ax.legend(frameon=False, ncol=3)
ax.fill_between(value_at_risk.index, value_at_risk['1%'], value_at_risk['5%'], color="red", alpha=0.2)
plt.show()