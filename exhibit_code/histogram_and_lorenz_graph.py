import pyarrow.dataset as ds
import pyarrow.fs as fs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

s3 = fs.S3FileSystem(anonymous=True)
with s3.open_input_file("cdcepi-flusight-forecast-hub/target-data/oracle-output.csv") as f:
    oracle = pd.read_csv(f)


oracle_mask = (oracle['target'] == 'wk inc flu hosp') & (oracle['location'] == 'US') & (oracle['horizon'] == 0)
# question: is it not log-normal?

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(oracle[oracle_mask]["oracle_value"], edgecolor='black', color='#EF3340', bins=25)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('Weekly Incidence of Flu Hospitalizations', size=16)
ax.set_ylabel('Frequency', size=16)
plt.savefig("./images/histogram.png", dpi=300, bbox_inches='tight')
plt.show()

data = oracle[oracle_mask].sort_values("oracle_value")["oracle_value"].values
lorenz = np.cumsum(data) / data.sum() #type:ignore
lorenz = np.insert(lorenz, 0, 0)  # start at (0,0)
x = np.linspace(0, 1, len(lorenz))

n = len(data)
G = (2 * np.sum(np.arange(1, n+1) * data)) / (n * data.sum()) - (n + 1) / n #type:ignore

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, lorenz, color='#8B0D21', linewidth=2)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')  # equality line
ax.set_xlabel('Cumulative Share of Weeks', size=15)
ax.set_ylabel('Cumulative Share of Flu Hospitalizations', size=15)
ax.text(0.57, 0.33, r'$\mathbf{\hat{G}ini} = $' + f'{G:.3f}',
        fontsize=15, ha='center', va='center', color='black')
ax.text(0.765, 0, r'$\mathbf{Gini} \coloneq \mathbb{E}_{Y, Y^* \sim G}|Y-Y^{*}|/(2\mathbb{E}[Y])$',
        fontsize=15, ha='center', va='center', color='black')
plt.savefig("./images/lorenz.png", dpi=300, bbox_inches='tight')
plt.show()