import matplotlib.pyplot as plt
import pandas as pd

result_file = 'result_test.csv'

result = pd.read_csv(result_file)
prob = result['score']
ale_unc = result['ale_unc']
epi_unc = result['epi_unc']
tot_unc = result['tot_unc']

f, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 3), constrained_layout=True)
ale_ax = axes[0]
ale_ax.scatter(prob, ale_unc, s=0.5, c='k')
ale_ax.set_xlabel('Predicted probability')
ale_ax.set_ylabel('Aleatoric uncertainty')
ale_ax.set_ylim([0, 0.3])

epi_ax = axes[1]
epi_ax.scatter(prob, epi_unc, s=0.5, c='k')
epi_ax.set_xlabel('Predicted probability')
epi_ax.set_ylabel('Epistemic uncertainty')
epi_ax.set_ylim([0, 0.3])

tot_ax = axes[2]
tot_ax.scatter(prob, tot_unc, s=0.5, c='k')
tot_ax.set_xlabel('Predicted probability')
tot_ax.set_ylabel('Total uncertainty')
tot_ax.set_ylim([0, 0.3])

plt.savefig('Uncertainty.png', dpi=300)
plt.show()
