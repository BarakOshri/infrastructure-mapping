from __future__ import division

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_similarity_score


columns = ['eaelectricity', 'eapipedwater', 'easewage', 'earoad']

data = pd.read_csv('Afrobarometer_R6.csv')
data = data.fillna(0)
urban = data['urban'] # 0 = rural, 1 = urban

def main():
  corrs = {}
  for i in range(len(columns)):
    col = columns[i]
    corr = jaccard_similarity_score(urban, data[col])
    corrs['urban/rural : ' + col[2:]] = np.round(corr, 4)
    for j in range(i + 1, len(columns)):
      corr = jaccard_similarity_score(data[col], data[columns[j]])
      corrs[col[2:] + ' : ' + columns[j][2:]] = np.round(corr, 4)

  print corrs

if __name__ == "__main__":
  main()
