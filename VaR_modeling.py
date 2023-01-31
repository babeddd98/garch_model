import yfinance as yf
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def garch_model(x, p, q, vol="GARCH"):
    garch = arch_model(x, p, q, mean="constant", vol=vol, dist="normal")
    garch_fit = garch.fit(update_freq=10)
    return garch_fit


ticker = "^FCHI"
data = yf.download(ticker, start=datetime(2010,1,1), end=datetime(2023,1,1))
data["Returns"] = 100*data["Close"].pct_change()
data = data.dropna()

garch = arch_model(data["Returns"], p=1, q=1, mean="constant", vol="EGARCH", dist="t")
garch_fit = garch.fit(update_freq=10)


f = garch_fit.forecast(start="2022-01-01")
mean_f = f.mean["2022":]
variance_f = f.variance["2022":]
quartile = garch.distribution.ppf([0.01, 0.05], garch_fit.params[4])
value_at_risk = -mean_f.values - np.sqrt(variance_f).values * quartile
value_at_risk = pd.DataFrame(value_at_risk, columns=['1%', '5%'], index=variance_f.index)
dates_1year = data.Returns["2022":]

colors = []
for i in value_at_risk.index:
    if dates_1year[i] > -value_at_risk.loc[i, "5%"]:
        colors.append("#000000")
    elif dates_1year[i] < -value_at_risk.loc[i, "1%"]:
        colors.append("#BB0000")
    else:
        colors.append("#BB00BB")

colors = np.array(colors)

labels = {

    '#BB0000': '1% Exceedence',
    '#BB00BB': '5% Exceedence',
    '#000000': 'No Exceedence'
}

markers = {'#BB0000': 'x', '#BB00BB': 's', '#000000': 'o'}

figure, [ax, value_at_risk_plot] = plt.subplots(2, 1)

value_at_risk_plot.plot(value_at_risk)

for color in np.unique(colors):
    sel = colors == color
    value_at_risk_plot.scatter(
        dates_1year.index[sel],
        -dates_1year.loc[sel],
        marker=markers[color],
        c=colors[sel],
        label=labels[color])

value_at_risk_plot.set_xlim([value_at_risk.index[0], value_at_risk.index[-1]])
value_at_risk_plot.set_title('Parametric value_at_risk')
value_at_risk_plot.legend()
value_at_risk_plot.fill_between(value_at_risk.index, value_at_risk['1%'], value_at_risk['5%'], color="red", alpha=0.2)

ax.plot(abs(data["Returns"]["2022":]), label="realised volatility", color="grey")
ax.plot(garch_fit.conditional_volatility, label="estimated volatility", color="red")
ax.set_xlim([value_at_risk.index[0], value_at_risk.index[-1]])
ax.set_title("CAC40 volatility estimation with GARCH model")
ax.legend()
plt.show()
