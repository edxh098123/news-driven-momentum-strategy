import unittest
import pandas as pd
from method.missing_returns import MissingReturnHandler,UnresolvedReturn

class MissingReturnsTests(unittest.TestCase):
 def setUp(self):
  self.h=MissingReturnHandler(); self.d0=pd.Timestamp('2015-05-11'); self.d1=pd.Timestamp('2015-05-12'); self.d2=pd.Timestamp('2015-05-13')
 def test_missing_then_p1_recovery(self):
  first=self.h.process(1,self.d1,self.d0,None,None,'MP','MR',False)
  self.assertEqual(first.kind,'stale_valuation'); self.assertFalse(first.assumed)
  recovered=self.h.process(1,self.d2,self.d1,.011605,.011605,'NA','P1',False,self.d0)
  self.assertAlmostEqual(recovered.total,.011605); self.assertFalse(self.h.pending)
 def test_multi_day_and_expiry(self):
  h=MissingReturnHandler(max_stale_sessions=2)
  for n in range(2): h.process(1,self.d1+pd.Timedelta(days=n),self.d0+pd.Timedelta(days=n),None,None,'MP','MR',False)
  with self.assertRaises(UnresolvedReturn): h.process(1,self.d1+pd.Timedelta(days=2),self.d0,None,None,'MP','MR',False)
 def test_two_day_gap_p2(self):
  self.h.process(1,self.d1,self.d0,None,None,'MP','MR',False)
  self.h.process(1,self.d2,self.d1,None,None,'MP','MR',False)
  self.assertEqual(self.h.process(1,self.d2+pd.Timedelta(days=1),self.d2,.2,.2,'NA','P2',False,self.d0).total,.2)
 def test_invalid_recovery_refused(self):
  self.h.process(1,self.d1,self.d0,None,None,'MP','MR',False)
  with self.assertRaises(UnresolvedReturn): self.h.process(1,self.d2,self.d1,.1,.1,'NA','D1',False,self.d1)
 def test_observed_delisting(self):
  o=self.h.process(1,self.d1,self.d0,-.5,-.5,'NA','DD',True)
  self.assertTrue(o.settlement);self.assertFalse(o.assumed)
 def test_missing_delisting_scenarios_and_sides(self):
  with self.assertRaises(UnresolvedReturn): self.h.process(1,self.d1,self.d0,None,None,'DG','MR',True)
  for loss in (-.3,-.55,-1.):
   h=MissingReturnHandler(loss);o=h.process(1,self.d1,self.d0,None,None,'DG','MR',True)
   self.assertTrue(o.settlement);self.assertTrue(o.assumed)
   self.assertLess(.05*o.total,0);self.assertGreater(-.05*o.total,0)
 def test_settlement_requires_known_date(self):
  ontime=MissingReturnHandler();o=ontime.process(1,self.d1,self.d0,None,None,'DG','MR',True,settlement_return=-.4,settlement_known_at=self.d1)
  self.assertFalse(o.assumed);self.assertAlmostEqual(o.total,-.4)
  future=MissingReturnHandler()
  with self.assertRaises(UnresolvedReturn):future.process(1,self.d1,self.d0,None,None,'DG','MR',True,settlement_return=-.4,settlement_known_at=self.d2)
 def test_unknown_is_unresolved(self):
  with self.assertRaises(UnresolvedReturn): self.h.process(1,self.d1,self.d0,None,None,'NT','MR',False)

class EngineIntegrationTests(unittest.TestCase):
 def inputs(self):
  import numpy as np
  dates=pd.bdate_range('2024-01-01',periods=9); cols=list(range(20))
  ret=pd.DataFrame(0.,index=dates,columns=cols); trade=pd.DataFrame(True,index=dates,columns=cols)
  members=pd.DataFrame({'permno':cols,'mbrstartdt':pd.Timestamp('2000-01-01'),'mbrenddt':pd.Timestamp('2030-01-01')})
  closes=pd.Series(pd.to_datetime([str(d.date())+' 21:00Z' for d in dates]),index=dates)
  news=pd.DataFrame({'timestamp_utc':[closes.iloc[4]-pd.Timedelta(minutes=1)]*20,'permno':cols,'event_sentiment_score':np.arange(20,dtype=float),'relevance':50,'event_similarity_days':10})
  return dates,cols,ret,trade,members,closes,news
 def test_gap_held_and_cumulative_recovery(self):
  from method.engine import run
  dates,cols,ret,trade,members,closes,news=self.inputs();ret.loc[dates[6],19]=float('nan');ret.loc[dates[7],19]=.10;trade.loc[dates[6],19]=False
  flags={(dates[6],19):{'dlyretmissflg':'MP','dlyretdurflg':'MR','dlydelflg':'N'},(dates[7],19):{'dlyretmissflg':'NA','dlyretdurflg':'P1','dlydelflg':'N','dlyprevdt':dates[5]}}
  h=MissingReturnHandler();out,_=run(ret,trade,members,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment',cost=0,borrow=0,missing_handler=h,crsp_flags=flags)
  self.assertAlmostEqual(out.loc[dates[6],'gross'],0.);self.assertAlmostEqual(out.loc[dates[7],'long'],.005)
  self.assertEqual([e['kind'] for e in h.events],['stale_valuation','observed_multiperiod_recovery'])
 def test_delisting_short_gain_and_settlement(self):
  from method.engine import run,PathIncomplete
  dates,cols,ret,trade,members,closes,news=self.inputs();ret.loc[dates[6],0]=float('nan');trade.loc[dates[6],0]=False
  delisted=pd.DataFrame(False,index=dates,columns=cols);delisted.loc[dates[6],0]=True
  flags={(dates[6],0):{'dlyretmissflg':'DG','dlyretdurflg':'MR','dlydelflg':'Y'}}
  with self.assertRaises(PathIncomplete) as caught:run(ret,trade,members,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment',cost=0,borrow=0,missing_handler=MissingReturnHandler(),crsp_flags=flags,delisted=delisted)
  self.assertIsNotNone(caught.exception.checkpoint)
  h=MissingReturnHandler(-1.);out,_=run(ret,trade,members,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment',cost=0,borrow=0,missing_handler=h,crsp_flags=flags,delisted=delisted)
  resumed,_=run(ret,trade,members,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment',cost=0,borrow=0,missing_handler=MissingReturnHandler(-1.),crsp_flags=flags,delisted=delisted,checkpoint=caught.exception.checkpoint)
  pd.testing.assert_frame_equal(out,resumed)
  self.assertAlmostEqual(out.loc[dates[6],'short'],.05);self.assertTrue(h.events[0]['assumed'])
 def test_untradeable_stale_security_is_not_filled(self):
  from method.engine import run
  dates,cols,ret,trade,members,closes,_=self.inputs()
  rows=[]
  for permno in cols:
   rows.append({'timestamp_utc':closes.iloc[0]-pd.Timedelta(minutes=2),'permno':permno,'event_sentiment_score':permno*5.,'relevance':50.,'event_similarity_days':10.})
   for offset in range(3):rows.append({'timestamp_utc':closes.iloc[1]-pd.Timedelta(minutes=offset+1),'permno':permno,'event_sentiment_score':100.-permno*5.,'relevance':50.,'event_similarity_days':10.})
  news=pd.DataFrame(rows);ret.loc[dates[2],19]=float('nan');ret.loc[dates[3],19]=.10;trade.loc[dates[2],19]=False
  flags={(dates[2],19):{'dlyretmissflg':'MP','dlyretdurflg':'MR','dlydelflg':'N'},(dates[3],19):{'dlyretmissflg':'NA','dlyretdurflg':'P1','dlydelflg':'N','dlyprevdt':dates[1]}}
  h=MissingReturnHandler();out,diag=run(ret,trade,members,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment',frequency='daily',cost=0,borrow=0,missing_handler=h,crsp_flags=flags)
  self.assertAlmostEqual(out.loc[dates[2],'gross_exposure'],.05)
  self.assertTrue(diag.reason.fillna('').str.contains('untradeable orders').any())
  self.assertAlmostEqual(out.loc[dates[3],'long'],.005)

class DataFlagsTests(unittest.TestCase):
 def test_delisting_metadata_uses_daily_date_and_latest_source_date(self):
  from data.collect import build_crsp_flags
  day=pd.Timestamp('2023-03-13');later=pd.Timestamp('2023-03-28')
  prices=pd.DataFrame([dict(permno=1,dlycaldt=day,dlyretmissflg='DG',dlyretdurflg='MR',dlydelflg='Y',dlyprevdt=pd.Timestamp('2023-03-10'))])
  links=pd.DataFrame([dict(permno=1,deldlydt=day,delret=-.7,delretmisstype='NA',delnextdt=later,delamtdt=day)])
  info=build_crsp_flags(prices,links)[(day,1)]
  self.assertEqual(info['settlement_known_at'],later)
  with self.assertRaises(UnresolvedReturn):MissingReturnHandler().process(1,day,pd.Timestamp('2023-03-10'),None,None,'DG','MR',True,settlement_return=info['settlement_return'],settlement_known_at=info['settlement_known_at'])
