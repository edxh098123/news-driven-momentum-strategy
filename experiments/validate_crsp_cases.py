"""Validate the general handler on specified CRSP cases without writing source data."""
import pathlib,sys,json
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.collect import DataAccess
from method.missing_returns import MissingReturnHandler,UnresolvedReturn

def feed(h,row,previous):
 return h.process(int(row.permno),pd.Timestamp(row.dlycaldt),pd.Timestamp(previous),row.dlyret,row.dlyretx,row.dlyretmissflg,row.dlyretdurflg,row.dlydelflg=='Y',row.dlyprevdt)

def main():
 cfg=json.loads((ROOT/'config/settings.json').read_text())
 db=DataAccess(cfg['wrds_user']);report={}
 try:
  d=db.prices('2015-05-11','2015-05-13','2008-09-01')
  h=MissingReturnHandler();a=feed(h,d[(d.permno==76744)&(d.dlycaldt==pd.Timestamp('2015-05-12'))].iloc[0],'2015-05-11');b=feed(h,d[(d.permno==76744)&(d.dlycaldt==pd.Timestamp('2015-05-13'))].iloc[0],'2015-05-12')
  assert a.kind=='stale_valuation' and b.kind=='observed_multiperiod_recovery' and abs(b.total-.011605)<1e-8
  report['VRTX']={'stale_class':a.kind,'recovery_class':b.kind,'recovery_return':b.total,'passed':True}
  d=db.prices('2023-03-13','2023-03-13','2008-09-01');row=d[d.permno.eq(90090)].iloc[0]
  try:feed(MissingReturnHandler(),row,'2023-03-10');raise AssertionError('strict must stop')
  except UnresolvedReturn:pass
  outcomes={}
  for loss in (-.3,-.55,-1.):
   h=MissingReturnHandler(loss);o=feed(h,row,'2023-03-10');assert o.assumed and o.settlement and o.total==loss;outcomes[str(loss)]=o.total
  report['SBNY']={'strict':'unresolved','scenarios':outcomes,'passed':True};report['wrds_queries']=len(db.query_stats)
 finally: db.close()
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
