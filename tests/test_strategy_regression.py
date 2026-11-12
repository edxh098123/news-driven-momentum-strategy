"""Freeze both strategies' normal-data signals, target weights and daily ledger."""
import hashlib
import unittest
import numpy as np
import pandas as pd
from method.signals import price_signal, weights
from method.engine import run
from method.missing_returns import MissingReturnHandler

class StrategyRegressionTests(unittest.TestCase):
    def fingerprint(self, use_handler):
        calendar=pd.bdate_range('2018-01-01','2020-06-30'); ids=list(range(32))
        ii=np.arange(len(calendar))[:,None]; jj=np.arange(32)[None,:]
        returns=pd.DataFrame(.0001+np.sin(ii*.15+jj)*.003+jj*.00002,index=calendar,columns=ids)
        tradable=pd.DataFrame(True,index=calendar,columns=ids)
        members=pd.DataFrame({'permno':ids,'mbrstartdt':pd.Timestamp('2017-01-01'),'mbrenddt':pd.Timestamp('2021-12-31')})
        closes=pd.Series(pd.to_datetime([str(d.date())+' 21:00Z' for d in calendar]),index=calendar)
        news={'sentiment':pd.DataFrame(np.cos(ii*.09+jj)*15+50,index=calendar,columns=ids)}
        signal=price_signal(returns,calendar,pd.Timestamp('2020-02-28')); targets,_=weights(signal,set(ids),10)
        blobs=[signal.to_csv(float_format='%.15g'),targets.to_csv(float_format='%.15g')]
        for kind in ('price','sentiment'):
            options={'missing_handler':MissingReturnHandler(),'crsp_flags':{}} if use_handler else {}
            frame,diag=run(returns,tradable,members,news,calendar,closes,'2019-06-01','2020-06-30',kind=kind,**options)
            blobs.extend([frame.to_csv(float_format='%.15g'),diag.to_csv(index=False,float_format='%.15g')])
        return hashlib.sha256(''.join(blobs).encode()).hexdigest()
    def test_original_and_missing_handler_match_pre_migration_baseline(self):
        baseline='a613eb76e5dd7006ad03d85091ba5e0e4796d42a69d57c14a6beacc54706f26d'
        self.assertEqual(self.fingerprint(False),baseline)
        self.assertEqual(self.fingerprint(True),baseline)

if __name__=='__main__': unittest.main()
