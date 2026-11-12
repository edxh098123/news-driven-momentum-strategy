"""Two-year WRDS experiments; inputs remain in memory, outputs are aggregates."""
import json, logging, os, pathlib, sys, time
import pandas as pd
import numpy as np
import exchange_calendars as xc
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.collect import DataAccess,link_entities,match_news,build_crsp_flags
from method.signals import aggregate_news,rolling_news
from method.engine import run,PathIncomplete
from method.missing_returns import MissingReturnHandler
from method.report import metrics

def execute(start,end,reporter=None):
 began=time.monotonic(); label=f'{start[:4]}_{end[:4]}'
 out=ROOT/'result/experiments'/label
 if out.exists() and any(out.iterdir()): out=out/f'run_{time.time_ns()}'
 out.mkdir(parents=True,exist_ok=True)
 log_path=ROOT/'logs'/f'experiment_{label}_{time.strftime("%Y%m%d_%H%M%S")}.log'
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(log_path)])
 config=json.loads((ROOT/'config/settings.json').read_text()); warm=(pd.Timestamp(start)-pd.DateOffset(months=13)).strftime('%Y-%m-%d')
 nyse=xc.get_calendar('XNYS',start=warm,end=end); schedule=nyse.schedule
 calendar=pd.DatetimeIndex(schedule.index).tz_localize(None); closes=pd.Series(pd.to_datetime(schedule.close,utc=True).values,index=calendar).dt.tz_localize('UTC')
 os.environ.setdefault('PGOPTIONS','-c statement_timeout=180000');db=DataAccess(config['wrds_user'])
 try:
  logging.info('Loading membership and historical mappings')
  membership=db.membership(warm,end); names=db.names(warm,end)
  mapping=db.mappings(warm,end,names.cusip.dropna().str[:8].unique().tolist())
  links=link_entities(names,mapping,membership,warm,end);del names,mapping
  prices=db.prices(warm,end,warm)
  ids=membership.permno.dropna().astype(int).unique().tolist()
  delistings=db.delistings(warm,end,ids)
  benchmark=db.benchmark(warm,end);rf=db.riskfree(start,end)
  returns=prices.pivot(index='dlycaldt',columns='permno',values='dlyret').reindex(calendar).astype(float)
  price_returns=prices.pivot(index='dlycaldt',columns='permno',values='dlyretx').reindex(calendar).astype(float)
  delisted=prices.assign(flag=prices.dlydelflg.eq('Y')).pivot(index='dlycaldt',columns='permno',values='flag').reindex(calendar).fillna(False).astype(bool)
  prices['can_trade']=prices.dlyprc.notna() & (prices.dlyprc.astype(float)>0) & prices.tradingstatusflg.eq('A') & prices.dlydelflg.ne('Y')
  tradable=prices.pivot(index='dlycaldt',columns='permno',values='can_trade').reindex(calendar).fillna(False).astype(bool)
  flags=build_crsp_flags(prices,delistings);del prices
  parts=[];news_start=(pd.Timestamp(start)-pd.Timedelta(days=14)).strftime('%Y-%m-%d')
  for period in pd.period_range(news_start,end,freq='Q'):
   a=max(pd.Timestamp(news_start),period.start_time).strftime('%Y-%m-%d');b=min(pd.Timestamp(end),period.end_time).strftime('%Y-%m-%d')
   logging.info('RavenPack %s to %s',a,b);raw=db.news(period.year,links.rp_entity_id.dropna().unique(),a,b);mapped=match_news(raw,links);del raw
   logging.info('Matched %d records',len(mapped));parts.append(aggregate_news(mapped,calendar,closes));del mapped
  if not parts or not any(not p.empty for p in parts): raise ValueError('No mapped news')
  news=rolling_news(parts,calendar); del parts
 finally:
  db.close()
 levels=benchmark.assign(datadate=pd.to_datetime(benchmark.datadate)).set_index('datadate').prccddiv.astype(float).sort_index();bench=levels.pct_change(fill_method=None).loc[start:end]
 risk=rf.assign(date=pd.to_datetime(rf.date)).set_index('date').rf.astype(float)
 required=calendar[(calendar>=pd.Timestamp(start))&(calendar<=pd.Timestamp(end))]
 if bench.reindex(required).isna().any() or risk.reindex(required).isna().any(): raise ValueError('Benchmark or risk-free coverage incomplete')
 scenarios=[('strict',None)]+[(f'loss_{int(-100*loss)}',float(loss)) for loss in config.get('missing_returns',{}).get('terminal_loss_scenarios',[-.30,-.55,-1.])];all_metrics=[];all_events=[];paths={};failure=[];price_cache={};replays=0
 for kind in ('price','sentiment'):
  strict_result=None; checkpoint=None
  for label_s,loss in scenarios:
   handler=MissingReturnHandler(loss,max_stale_sessions=config.get('missing_returns',{}).get('max_stale_sessions',10)); start_run=time.monotonic()
   if label_s!='strict' and strict_result is not None and not any(e['kind']=='unresolved_delisting' for e in strict_result[3]):
    frame,diag,status,events,error=strict_result; events=[{**e,'scenario':loss} for e in events]
    if status=='complete':status='scenario_complete'
    logging.info('%s %s reused unaffected strict path',kind,label_s)
   else:
    if label_s!='strict':replays+=1
    logging.info('Backtest %s %s',kind,label_s);error=''
    try:
     frame,diag=run(returns,tradable,membership,news,calendar,closes,start,end,kind=kind,price_returns=price_returns,delisted=delisted,price_cache=price_cache,missing_handler=handler,crsp_flags=flags,checkpoint=checkpoint if label_s!='strict' else None)
     status='complete' if label_s=='strict' else 'scenario_complete'
    except PathIncomplete as exc:
     frame=exc.frame;diag=exc.diagnostics;status='incomplete';error=str(exc)
     if label_s=='strict':checkpoint=exc.checkpoint
     failure.append({'strategy':kind,'scenario':label_s,'error':error})
    events=handler.events
   if label_s=='strict':strict_result=(frame,diag,status,events,error)
   for event in events: all_events.append({'strategy':kind,'scenario_name':label_s,**event})
   if len(frame)>1:
    paths[(kind,label_s)]=frame
    all_metrics.append({'strategy':kind,'scenario':label_s,'status':status,'sample_start':str(frame.index.min().date()),'sample_end':str(frame.index.max().date()),'seconds':round(time.monotonic()-start_run,2),'error':error,**metrics(frame,bench,risk)})
   else: all_metrics.append({'strategy':kind,'scenario':label_s,'status':'incomplete','error':error})
 bf=pd.DataFrame({'net':bench,'gross':bench,'turnover':0.,'trade_cost':0.,'borrow_cost':0.,'gross_exposure':1.,'net_exposure':1.})
 all_metrics.append({'strategy':'sp500_total_return','scenario':'observed','status':'complete','sample_start':str(bf.index.min().date()),'sample_end':str(bf.index.max().date()),**metrics(bf,bench,risk)})
 events=pd.DataFrame(all_events)
 if not events.empty and not delistings.empty:
  delistings=delistings.rename(columns={'deldlydt':'date'});delistings['date']=pd.to_datetime(delistings.date)
  events=events.merge(delistings,on=['permno','date'],how='left')
 events.to_csv(out/'missing_events.csv',index=False)
 if not events.empty: events.groupby(['strategy','scenario_name','kind'],dropna=False).size().rename('count').reset_index().to_csv(out/'missing_summary.csv',index=False)
 else: pd.DataFrame(columns=['strategy','scenario_name','kind','count']).to_csv(out/'missing_summary.csv',index=False)
 pd.DataFrame(all_metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(failure,columns=['strategy','scenario','error']).to_csv(out/'unresolved.csv',index=False)
 legs=[]
 for (kind,scenario),frame in paths.items():
  if kind!='sentiment': continue
  for year,g in frame.groupby(frame.index.year):
   prior=(1+g.net).cumprod().shift(1,fill_value=1.)
   legs.append({'year':year,'scenario':scenario,'long_contribution':(prior*g['long']).sum(),'short_contribution':(prior*g['short']).sum(),'transaction_cost':(prior*g.trade_cost).sum(),'borrow_cost':(prior*g.borrow_cost).sum(),'net_return':(1+g.net).prod()-1})
 pd.DataFrame(legs).to_csv(out/'news_legs.csv',index=False)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(10,5));(1+bench).cumprod().plot(ax=ax,label='S&P 500 total return')
 for (kind,scenario),frame in paths.items():
  (1+frame.net).cumprod().plot(ax=ax,label=f'{kind} {scenario}',alpha=.8)
 ax.legend(fontsize=7);ax.set_title(f'{start} to {end}: incomplete paths end early');fig.tight_layout();fig.savefig(out/'cumulative_returns.png');plt.close(fig)
 if reporter is not None:
  reporter(out,paths,bench,risk,required,start,end)
 stats={'start':start,'end':end,'elapsed_seconds':round(time.monotonic()-began,2),'wrds_query_count':len(db.query_stats),'wrds_queries':db.query_stats,'slow_queries':sum(x['seconds']>=10 for x in db.query_stats),'scenario_checkpoint_replays':replays,'log_file':str(log_path),'note':'Affected scenarios resume at the missing-return date from a saved in-memory portfolio checkpoint; inputs and signal caches are shared. Assumed delist settlement at close, no borrow availability data.'}
 (out/'run_summary.json').write_text(json.dumps(stats,indent=2,default=str));print(json.dumps({'output':str(out),**stats},default=str),flush=True)
