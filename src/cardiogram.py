import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

from detector_combo import ScollingDetectorAggregator
#import awswrangler as wr

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
import boto3

from io import StringIO

class CardiogramDetector():

    def __init__(self, bucket,file_name):
        self.bucket = bucket
        self.file_name = file_name

    def detect(self):
        '''
        detectors = [RollingCount(), RollingSum(), RollingMean(), RollingMedian(), RollingVar(),
                    RollingStd(), RollingCorr(), RollingCov(), RollingMAA(), RollingLogMAA()]
        weights = [0, 0.001, 0.02, 0.03, 0,
                0, 0.02, 0, 0.04, 0.889]
        '''
        detectors = [RollingMAA(), RollingLogMAA()]
        weights = [0.01, 0.99]

        clf = ScollingDetectorAggregator(base_estimators=detectors, weights=weights)
        clf_name = 'Aggregation by Averaging'

        #wave_df=pd.read_csv(self.path, index_col=False)
        #wave_df=wave_df[0:5000]

        #x=wave_df['Unnamed: 0']
        s3 = boto3.client('s3')

        obj = s3.get_object(Bucket=self.bucket, Key=self.file_name)
        wave_df = pd.read_csv(obj['Body'])
        y=wave_df['value'].values.reshape(-1, 1)

        clf.fit(y)

        #y_test_pred = clf.predict(wave_df1)  # outlier labels (0 or 1)
        #y_test_scores = clf.decision_function(wave_df1)  # outlier scores
        y_test_pred = clf.predict(y)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(y)  # outlier scores

        wave_df['score'] = y_test_scores
        wave_df['result'] = y_test_pred

        csv_buf = StringIO()
        wave_df.to_csv(csv_buf, header=True, index=False)
        csv_buf.seek(0)
        s3.put_object(Bucket=self.bucket, Body=csv_buf.getvalue(), Key=self.file_name+"_result.csv")

        #wave_df.to_csv(self.path+'result.csv', index=False)
        """
        wave_df.to_csv('result.csv', index=False)

        s3 = boto3.client("s3")

        s3.put_object(
            Body=open("filename.csv").read(),
            Bucket="your-bucket",
            Key="your-key"
        )


        wr.s3.to_csv(
            df=wave_df,
            path=self.path+'result.csv'
        )
        """

        return True
