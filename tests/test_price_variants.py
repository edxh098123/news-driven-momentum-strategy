import unittest
import numpy as np
import pandas as pd
from method.price_variants import continuous_signal,intermediate_signal,residual_momentum
from method.signals import weights

class PriceVariantsTests(unittest.TestCase):
    def setUp(self):
        self.calendar=pd.bdate_range('2015-01-01','2022-12-30')
        self.rng=np.random.default_rng(7)
        self.r=pd.DataFrame(self.rng.normal(.0002,.01,(len(self.calendar),60)),index=self.calendar,columns=range(60))
    def test_intermediate_ignores_recent_and_future_returns(self):
        date=pd.Timestamp('2021-12-31'); original=intermediate_signal(self.r,self.calendar,date)
        changed=self.r.copy(); changed.loc['2021-07-01':]=.5
        pd.testing.assert_series_equal(original,intermediate_signal(changed,self.calendar,date))
    def test_continuous_selects_extreme_pret_and_low_id(self):
        date=pd.Timestamp('2021-12-31'); tradable=pd.Series(True,index=self.r.columns)
        score=continuous_signal(self.r,self.calendar,date,set(self.r.columns),tradable)
        target,n=weights(score,set(self.r.columns),10)
        self.assertEqual(n,20); self.assertEqual((target>0).sum(),10); self.assertEqual((target<0).sum(),10)
        altered=self.r.copy(); altered.loc['2022-01-03':]=.9
        pd.testing.assert_series_equal(score,continuous_signal(altered,self.calendar,date,set(self.r.columns),tradable))
    def test_continuous_insufficient_extreme_quintile(self):
        score=continuous_signal(self.r,self.calendar,pd.Timestamp('2021-12-31'),set(range(45)),pd.Series(True,index=self.r.columns))
        self.assertTrue(score.empty)
    def test_residual_uses_only_prior_month_factors_and_returns(self):
        months=pd.date_range('2015-01-01','2022-12-01',freq='MS')
        factors=pd.DataFrame(self.rng.normal(0,.03,(len(months),4)),index=months,columns=['mktrf','smb','hml','rf'])
        first=residual_momentum(self.r,self.calendar,factors)
        changed=factors.copy();changed.loc['2021-12-01':]=99
        r=self.r.copy();r.loc['2021-12-01':]=.9
        second=residual_momentum(r,self.calendar,changed)
        pd.testing.assert_series_equal(first.loc['2021-12'],second.loc['2021-12'])
        self.assertGreater(first.loc['2021-12'].notna().sum(),0)

if __name__=='__main__': unittest.main()
