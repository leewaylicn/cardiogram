import os
import sys

#sys.path.append(
#    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

from detector_combo import ScollingDetectorAggregator

from rolling_method import RollingCount
from rolling_method import RollingSum
from rolling_method import RollingMean
from rolling_method import RollingMedian
from rolling_method import RollingVar
from rolling_method import RollingStd
from rolling_method import RollingCorr
from rolling_method import RollingCov
from rolling_method import RollingMAA
from rolling_method import RollingLogMAA

import pandas as pd

detectors = [RollingMAA(), RollingLogMAA()]
weights = [0.01, 0.99]

clf = ScollingDetectorAggregator(base_estimators=detectors, weights=weights)
clf_name = 'Aggregation by Averaging'

wave_df=pd.read_csv(file_path)

#x=wave_df['Unnamed: 0']
y=wave_df['value'].values.reshape(-1, 1)

clf.fit(y)

#y_test_pred = clf.predict(wave_df1)  # outlier labels (0 or 1)
#y_test_scores = clf.decision_function(wave_df1)  # outlier scores
y_test_pred = clf.predict(y)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(y)  # outlier scores

wave_df['score'] = y_test_scores
wave_df['result'] = y_test_pred

wave_df.to_csv(file_path+'result.csv')