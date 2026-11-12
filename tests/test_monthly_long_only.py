import unittest
import numpy as np
import pandas as pd
from method.signals import price_signal
from method.monthly_long_only import long_weights,momentum_12_1,filtered_12_1,filtered_continuous
from method.kalman_momentum import filter_log_returns,kalman_signal
from method.price_variants import monthly_returns,residual_from_monthly
from method.engine import run

class MonthlyLongOnlyTests(unittest.TestCase):
    def test_distinct_month_windows_and_exact_compounding(self):
        cal=pd.bdate_range('2018-01-01','2022-12-30'); r=pd.DataFrame({1:.001,2:.002},index=cal)
        date=pd.Timestamp('2021-12-31');a=momentum_12_1(r,cal,date);b=price_signal(r,cal,date)
        marks=pd.Series(cal,index=cal).groupby(cal.to_period('M')).max();marks=marks[marks<=date]
        x=r.loc[(r.index>marks.iloc[-14])&(r.index<=marks.iloc[-2])]
        pd.testing.assert_series_equal(a,(1+x).prod()-1);self.assertTrue((a>b).all())
        monthly=monthly_returns(r,cal);logs=np.log1p(monthly)
        pd.testing.assert_series_equal(a,filtered_12_1(logs,monthly,cal,date),check_exact=False,rtol=1e-12,atol=1e-12)
        pd.testing.assert_series_equal(b,kalman_signal(logs,monthly,cal,date,'12_2'),check_exact=False,rtol=1e-12,atol=1e-12)
    def test_weights_and_insufficient(self):
        target,n=long_weights(pd.Series(range(15),index=range(15)),set(range(15)),10)
        self.assertEqual(n,15);self.assertAlmostEqual(target.sum(),1);self.assertTrue((target==.1).all());self.assertEqual(set(target.index),set(range(5,15)))
        self.assertTrue(long_weights(pd.Series(range(9)),set(range(9)),10)[0].empty)
    def test_month_end_then_next_close_and_funded_cost(self):
        calendar=pd.bdate_range('2021-01-27','2021-03-03'); ids=list(range(10));r=pd.DataFrame(.01,index=calendar,columns=ids)
        tradable=pd.DataFrame(True,index=calendar,columns=ids);member=pd.DataFrame({'permno':ids,'mbrstartdt':pd.Timestamp('2020-01-01'),'mbrenddt':pd.Timestamp('2022-01-01')})
        panels={'x':pd.DataFrame(np.tile(np.arange(10),(len(calendar),1)),index=calendar,columns=ids)}
        frame,diag=run(r,tradable,member,panels,calendar,pd.Series(index=calendar,dtype=float),'2021-01-27','2021-03-03',kind='x',frequency='monthly',borrow=0,weight_selector=long_weights,fully_funded=True)
        self.assertEqual(frame.loc['2021-01-29','turnover'],0)
        self.assertAlmostEqual(frame.loc['2021-02-01','turnover'],1/1.001)
        self.assertAlmostEqual(frame.loc['2021-02-01','gross_exposure'],1)
        self.assertAlmostEqual(frame.loc['2021-02-01','net'],-.001/1.001)
        self.assertAlmostEqual(frame.loc['2021-02-02','net'],.01)
        self.assertIn(pd.Timestamp('2021-02-26'),set(diag.formation))
    def test_year_start_uses_prior_month_formation(self):
        cal=pd.DatetimeIndex(['2020-12-30','2020-12-31','2021-01-04','2021-01-05']);ids=list(range(10))
        r=pd.DataFrame(0.,index=cal,columns=ids);tradable=pd.DataFrame(True,index=cal,columns=ids)
        members=pd.DataFrame({'permno':ids,'mbrstartdt':pd.Timestamp('2020-01-01'),'mbrenddt':pd.Timestamp('2022-01-01')})
        panels={'x':pd.DataFrame(np.tile(np.arange(10),(len(cal),1)),index=cal,columns=ids)}
        frame,_=run(r,tradable,members,panels,cal,pd.Series(index=cal,dtype=float),'2020-12-31','2021-01-05',kind='x',frequency='monthly',borrow=0,weight_selector=long_weights,fully_funded=True)
        self.assertEqual(frame.loc['2020-12-31','turnover'],0)
        self.assertGreater(frame.loc['2021-01-04','turnover'],0)

    def test_filtered_continuous_never_uses_future(self):
        cal=pd.bdate_range('2015-01-01','2022-12-30');rng=np.random.default_rng(19)
        r=pd.DataFrame(rng.normal(0,.01,(len(cal),100)),index=cal,columns=range(100));m=monthly_returns(r,cal);f=filter_log_returns(m);d=pd.Timestamp('2021-12-31')
        x=filtered_continuous(f,m,cal,d,set(range(100)),pd.Series(True,index=r.columns));self.assertEqual(len(x),10)
        altered=r.copy();altered.loc['2022-01-01':]=.5;mm=monthly_returns(altered,cal);ff=filter_log_returns(mm)
        pd.testing.assert_series_equal(x,filtered_continuous(ff,mm,cal,d,set(range(100)),pd.Series(True,index=r.columns)))
    def test_filtered_residual_is_future_causal(self):
        idx=pd.period_range('2014-01','2022-12',freq='M');rng=np.random.default_rng(11)
        m=pd.DataFrame(rng.normal(.01,.03,(len(idx),3)),index=idx);f=pd.DataFrame(rng.normal(0,.02,(len(idx),4)),index=idx.to_timestamp(),columns=['mktrf','smb','hml','rf'])
        def calc(monthly,factors):return residual_from_monthly(np.expm1(filter_log_returns(monthly)),factors)
        a=calc(m,f);m2=m.copy();m2.loc['2021-01':]=.8;f2=f.copy();f2.loc['2021-01-01':]=.8;b=calc(m2,f2)
        pd.testing.assert_frame_equal(a.loc[:'2020-12'],b.loc[:'2020-12'])

if __name__=='__main__':unittest.main()
