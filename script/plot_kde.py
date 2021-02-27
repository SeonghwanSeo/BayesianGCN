import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

result_file = 'result_test.csv'

result = pd.read_csv(result_file)
probs = result['score']
pos_probs = result[result.Label == 1]['score']
neg_probs = result[result.Label == 0]['score']

plt.figure(figsize = (6,5))
sns.kdeplot(probs, label='ALL', lw=1, color='k', clip=(0.0,1.0))
sns.kdeplot(pos_probs, label='FDA', lw=1, color='r', clip=(0.0,1.0))
sns.kdeplot(neg_probs, label='ZINC', lw=1, color='b', clip=(0.0,1.0))
plt.xlabel('Predicted probability')
plt.legend()
plt.savefig('KDE.png', dpi=300)
plt.show()
