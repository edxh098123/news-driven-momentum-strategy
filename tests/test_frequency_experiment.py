import unittest
import numpy as np
import pandas as pd
from method.frequency_experiment import schedules,matched_target
from method.engine import run
from method.monthly_long_only import long_weights

class FrequencyTests(unittest.TestCase):
    def test_schedules_and_annual_phases(self):
        cal=pd.bdate_range('2019-01-01','2022-12-31');s=schedules(cal)
        self.assertEqual(list(s['quarterly'].month.unique()),[3,6,9,12])
        for phase in (1,4,7,10):self.assertEqual(set(schedules(cal,phase)['annual'].month),{(phase-2)%12+1})
        self.assertIn(pd.Timestamp('2020-12-31'),s['annual'])
    def test_drift_cost_and_no_unscheduled_rebalance(self):
        cal=pd.bdate_range('2020-12-30','2021-02-05');cols=list(range(10));r=pd.DataFrame(0.,index=cal,columns=cols);r.loc['2021-01-05',0]=.5
        trad=pd.DataFrame(True,index=cal,columns=cols);m=pd.DataFrame({'permno':cols,'mbrstartdt':pd.Timestamp('2020-01-01'),'mbrenddt':pd.Timestamp('2022-01-01')})
        panel={'x':pd.DataFrame(np.tile(np.arange(10),(len(cal),1)),index=cal,columns=cols)}; observations={}
        def observe(d,b,a,p,n,prev):observations[d]=a/n
        result,_=run(r,trad,m,panel,cal,pd.Series(index=cal,dtype=float),'2020-12-31','2021-02-05',kind='x',cost=.001,borrow=0,weight_selector=long_weights,fully_funded=True,formation_schedule=[pd.Timestamp('2020-12-31')],observer=observe)
        self.assertAlmostEqual(result.loc['2021-01-01','turnover'],1/1.001)
        self.assertAlmostEqual(result.loc['2021-01-05','turnover'],0.)
        self.assertGreater(observations[pd.Timestamp('2021-01-05')][0],.1)
    def test_risk_scaling_uses_only_past_and_no_leverage(self):
        idx=pd.bdate_range('2020-01-01','2021-12-31');rng=np.random.default_rng(4);r=pd.DataFrame(rng.normal(0,.03,(len(idx),10)),index=idx,columns=range(10));b=pd.Series(rng.normal(0,.01,len(idx)),index=idx);w=pd.Series(.1,index=range(10));date=idx[300]
        v=matched_target(r,b)(date,w,[]);r2=r.copy();r2.loc[idx[301:]]=100;b2=b.copy();b2.loc[idx[301:]]=100
        pd.testing.assert_series_equal(v,matched_target(r2,b2)(date,w,[]));self.assertLessEqual(v.sum(),1)

if __name__=='__main__':unittest.main()
