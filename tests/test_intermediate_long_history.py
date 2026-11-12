import unittest
import numpy as np
import pandas as pd
from method.volatility_target import target_15pct
from method.missing_returns import UnresolvedReturn
from experiments.intermediate_long_history import StrictObserved

class LongHistoryTests(unittest.TestCase):
    def test_no_stale_valuation(self):
        h=StrictObserved()
        o=h.process(1,pd.Timestamp('2015-05-12'),pd.Timestamp('2015-05-11'),np.nan,np.nan,'MP','',False)
        self.assertEqual(o.kind,'stale_valuation')
        self.assertIn(1,h.pending)
    def test_no_unknown_terminal_substitution(self):
        h=StrictObserved()
        with self.assertRaises(UnresolvedReturn):h.process(1,pd.Timestamp('2015-05-12'),pd.Timestamp('2015-05-11'),np.nan,np.nan,'DG','',True)
    def test_vol_target_is_causal_and_capped(self):
        idx=pd.bdate_range('2008-01-01',periods=200);rng=np.random.default_rng(1);r=pd.DataFrame(rng.normal(0,.1,(200,10)),index=idx,columns=range(10));w=pd.Series(.1,index=range(10));date=idx[150]
        a=target_15pct(r)(date,w,[]);r2=r.copy();r2.loc[idx[151:]]=1.;b=target_15pct(r2)(date,w,[])
        pd.testing.assert_series_equal(a,b);self.assertGreater(a.sum(),0);self.assertLessEqual(a.sum(),1)

if __name__=='__main__':unittest.main()

class BenchmarkHoldingsTests(unittest.TestCase):
    def test_historical_constituent_count_deduplicates_permno(self):
        from experiments.intermediate_long_history import benchmark_holdings
        m=pd.DataFrame({'permno':[1,1,2],'mbrstartdt':pd.to_datetime(['2020-01-01','2020-01-01','2020-01-02']),'mbrenddt':pd.to_datetime(['2020-01-03']*3)})
        self.assertEqual(benchmark_holdings(m,pd.to_datetime(['2020-01-01','2020-01-02'])),1.5)
