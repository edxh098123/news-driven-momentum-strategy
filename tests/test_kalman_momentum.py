import unittest
import numpy as np
import pandas as pd
from method.kalman_momentum import filter_log_returns,kalman_signal

class KalmanTests(unittest.TestCase):
    def setUp(self):
        self.index=pd.period_range('2017-01','2022-12',freq='M')
        self.values=pd.DataFrame({1:np.exp(.01+.02*np.sin(np.arange(len(self.index))))-1},index=self.index)
    def test_future_perturbation_does_not_change_past(self):
        first=filter_log_returns(self.values); changed=self.values.copy();changed.loc['2021-01':]=2.
        second=filter_log_returns(changed)
        pd.testing.assert_frame_equal(first.loc[:'2020-12'],second.loc[:'2020-12'])
    def test_fixed_prediction_and_update(self):
        x=pd.DataFrame({1:[.01,.02,.03]},index=pd.period_range('2020-01',periods=3,freq='M'))
        out=filter_log_returns(x,phi=.95,q_over_r=.05,init_months=2)
        a,b,c=np.log1p(x[1]); r=np.var([a,b],ddof=1); pred=.95*np.mean([a,b]); pv=(.95**2+.05)*r
        self.assertAlmostEqual(out.iloc[2,0],pred+pv/(pv+r)*(c-pred))
        self.assertTrue(out.iloc[:2,0].isna().all())
    def test_missing_prediction_only_and_invalid_log(self):
        x=self.values.copy();x.iloc[30,0]=np.nan;x.iloc[32,0]=-1
        y=filter_log_returns(x)
        self.assertTrue(np.isnan(y.iloc[30,0]));self.assertTrue(np.isnan(y.iloc[32,0]));self.assertTrue(np.isfinite(y.iloc[31,0]));self.assertTrue(np.isfinite(y.iloc[33,0]))
    def test_formation_window_requires_observations(self):
        calendar=pd.bdate_range('2017-01-01','2022-12-31'); m=self.values.copy(); f=filter_log_returns(m)
        date=pd.Timestamp('2021-12-31'); self.assertEqual(len(kalman_signal(f,m,calendar,date,'12_2')),1)
        m.loc['2021-03',1]=np.nan
        self.assertTrue(kalman_signal(f,m,calendar,date,'12_2').empty)
    def test_windows_match_original_when_filtered_equals_log_return(self):
        from method.price_variants import monthly_returns, intermediate_signal
        from method.signals import price_signal
        calendar=pd.bdate_range('2017-01-02','2022-12-30')
        days=pd.DataFrame({1:.001,2:-.0004},index=calendar)
        monthly=monthly_returns(days,calendar); filtered=np.log1p(monthly)
        for date in [pd.Timestamp('2021-12-31'),pd.Timestamp('2021-12-24')]:
            for window,ref in [('12_2',price_signal(days,calendar,date)),('12_7',intermediate_signal(days,calendar,date))]:
                pd.testing.assert_series_equal(kalman_signal(filtered,monthly,calendar,date,window),ref,check_exact=False,atol=1e-12,rtol=1e-12)

    def test_isolated_shock_is_attenuated_at_observation(self):
        baseline=self.values.copy(); shocked=baseline.copy(); date=self.index[40]
        shocked.loc[date,1]=np.expm1(np.log1p(shocked.loc[date,1])+.5)
        response=(filter_log_returns(shocked)-filter_log_returns(baseline)).loc[date,1]
        self.assertGreater(response,0.)
        self.assertLess(response,.5)

    def test_reject_invalid_fixed_parameters(self):
        with self.assertRaises(ValueError):filter_log_returns(self.values,q_over_r=0)

if __name__=='__main__':unittest.main()
