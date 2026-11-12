import pathlib,sys,unittest
import pandas as pd
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from method.signals import weights,news_signal
from method.engine import run, PathIncomplete
from method.report import metrics

class Tests(unittest.TestCase):
 def test_insufficient_and_neutral(self):
  s=pd.Series(range(20),index=range(20)); self.assertTrue(weights(s,set(range(19)),10)[0].empty)
  w,n=weights(s,set(range(20)),10); self.assertEqual(n,20); self.assertAlmostEqual(w.sum(),0); self.assertAlmostEqual(w.abs().sum(),1)
 def test_news_cutoff_and_missing(self):
  cal=pd.bdate_range('2024-01-01',periods=6); closes=pd.Series(pd.to_datetime([str(d.date())+' 21:00Z' for d in cal]),index=cal)
  news=pd.DataFrame({'timestamp_utc':pd.to_datetime(['2024-01-08 20:59Z','2024-01-08 21:01Z']),'event_sentiment_score':[80.,0.],'permno':[1,1],'relevance':[50,50],'event_similarity_days':[10,10]})
  self.assertEqual(news_signal(news,cal,closes,cal[-1]).loc[1],80)
 def test_next_close_and_cost(self):
  dates=pd.bdate_range('2024-01-01',periods=8); columns=list(range(20)); ret=pd.DataFrame(0.,index=dates,columns=columns); ret.loc[dates[5],19]=.2; ret.loc[dates[6],19]=.1
  tradable=pd.DataFrame(True,index=dates,columns=columns); members=pd.DataFrame({'permno':columns,'mbrstartdt':pd.Timestamp('2000-01-01'),'mbrenddt':pd.Timestamp('2030-01-01')})
  closes=pd.Series(pd.to_datetime([str(d.date())+' 21:00Z' for d in dates]),index=dates)
  news=pd.DataFrame({'timestamp_utc':[closes.iloc[4]-pd.Timedelta(minutes=1)]*20,'permno':columns,'event_sentiment_score':np.arange(20,dtype=float),'relevance':50,'event_similarity_days':10})
  out,_=run(ret,tradable,members,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment')
  self.assertAlmostEqual(out.loc[dates[5],'trade_cost'],.001); self.assertAlmostEqual(out.loc[dates[5],'gross'],0); self.assertGreater(out.loc[dates[6],'long'],0)
 def test_dividend_is_cash_not_extra_shares(self):
  dates=pd.bdate_range('2024-01-01',periods=8); cols=list(range(20)); total=pd.DataFrame(0.,index=dates,columns=cols); total.loc[dates[6],19]=.10; total.loc[dates[7],19]=.10
  ex=total.copy(); ex.loc[dates[6],19]=0.
  tr=pd.DataFrame(True,index=dates,columns=cols); m=pd.DataFrame({'permno':cols,'mbrstartdt':pd.Timestamp('2000-01-01'),'mbrenddt':pd.Timestamp('2030-01-01')})
  closes=pd.Series(pd.to_datetime([str(d.date())+' 21:00Z' for d in dates]),index=dates)
  news=pd.DataFrame({'timestamp_utc':[closes.iloc[4]-pd.Timedelta(minutes=1)]*20,'permno':cols,'event_sentiment_score':np.arange(20,dtype=float),'relevance':50,'event_similarity_days':10})
  result,_=run(total,tr,m,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment',price_returns=ex,cost=0,borrow=0)
  self.assertAlmostEqual(result.loc[dates[7],'long'],.005/1.005)
  total.loc[dates[6],0]=.1; ex.loc[dates[6],0]=0.
  result,_=run(total,tr,m,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment',price_returns=ex,cost=0,borrow=0)
  self.assertAlmostEqual(result.loc[dates[6],'short'],-.005)
 def test_missing_held_return_aborts(self):
  dates=pd.bdate_range('2024-01-01',periods=8); cols=list(range(20)); r=pd.DataFrame(0.,index=dates,columns=cols); r.loc[dates[6],19]=float('nan')
  tr=pd.DataFrame(True,index=dates,columns=cols); m=pd.DataFrame({'permno':cols,'mbrstartdt':pd.Timestamp('2000-01-01'),'mbrenddt':pd.Timestamp('2030-01-01')})
  closes=pd.Series(pd.to_datetime([str(d.date())+' 21:00Z' for d in dates]),index=dates)
  news=pd.DataFrame({'timestamp_utc':[closes.iloc[4]-pd.Timedelta(minutes=1)]*20,'permno':cols,'event_sentiment_score':np.arange(20,dtype=float),'relevance':50,'event_similarity_days':10})
  with self.assertRaisesRegex(PathIncomplete,'missing held returns') as caught: run(r,tr,m,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment')
  self.assertEqual(caught.exception.frame.index.max(),dates[5])
  r.loc[dates[5],19]=.02; delisted=pd.DataFrame(False,index=dates,columns=cols); delisted.loc[dates[5],19]=True
  # The terminal flag on an entry date must not enable trading a delisted security.
  tr.loc[dates[5],19]=False
  out,_=run(r,tr,m,news,dates,closes,str(dates[0].date()),str(dates[-1].date()),kind='sentiment',delisted=delisted)
  self.assertEqual(out.gross_exposure.sum(),0.)
 def test_metrics_drawdown_from_initial_wealth(self):
  dates=pd.date_range('2024-01-01',periods=2); f=pd.DataFrame({'net':[-.1,0], 'gross':[-.1,0], 'turnover':[0,0], 'trade_cost':[0,0], 'borrow_cost':[0,0], 'gross_exposure':[0,0], 'net_exposure':[0,0]},index=dates)
  self.assertAlmostEqual(metrics(f,pd.Series([0,.01],index=dates),pd.Series(0.,index=dates))['max_drawdown'],-.1)
if __name__=='__main__':unittest.main()
