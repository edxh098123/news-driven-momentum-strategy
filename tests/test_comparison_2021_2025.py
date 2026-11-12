import sys, pathlib, unittest
import pandas as pd
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'experiments'))
from period_2021_2025 import annual_tables

class AnnualComparisonTests(unittest.TestCase):
 def test_partial_year_is_not_reported_as_full_and_legs_reconcile(self):
  dates=pd.bdate_range('2021-01-01','2022-12-31'); bench=pd.Series(.001,index=dates);rf=pd.Series(.0001,index=dates)
  def frame(index):
   n=len(index);return pd.DataFrame({'net':np.full(n,.001),'gross':np.full(n,.0012),'long':np.full(n,.002),'short':np.full(n,-.0008),'trade_cost':np.full(n,.0001),'borrow_cost':np.full(n,.0001),'turnover':0.,'gross_exposure':1.,'net_exposure':0.},index=index)
  partial=frame(dates[dates<='2022-03-31']);full=frame(dates)
  annual,legs=annual_tables({('price','strict'):partial,('sentiment','strict'):full},bench,rf,dates)
  row=annual[(annual.strategy=='price')&(annual.scenario=='strict')&(annual.year==2022)].iloc[0]
  self.assertEqual(row.status,'incomplete');self.assertTrue(pd.isna(row.annual_net_return));self.assertTrue(pd.notna(row.partial_net_return))
  row=legs[(legs.strategy=='price')&(legs.scenario=='strict')&(legs.year==2022)].iloc[0]
  self.assertAlmostEqual(row.long_contribution+row.short_contribution-row.transaction_cost-row.borrow_cost,row.net_return)
 def test_absent_path_remains_unavailable(self):
  dates=pd.bdate_range('2021-01-01','2021-12-31');a,_=annual_tables({},pd.Series(.001,index=dates),pd.Series(0.,index=dates),dates)
  row=a[(a.strategy=='price')&(a.scenario=='strict')&(a.year==2021)].iloc[0]
  self.assertEqual(row.status,'unavailable');self.assertEqual(row.days,0)
