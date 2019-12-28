from sklearn.externals import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./regplot.csv', header=0)
sns.set(style="ticks", color_codes=True)

sns_plot = sns.regplot(x="PREDICTIONS", y="WAT", data=df)
fig = sns_plot.get_figure()
fig.savefig("reg_plot.png")

plt.show()