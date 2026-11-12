"""Intermediate long-only 2010–2020, strict observed paths and 15% vol target."""
import json,logging,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.collect import DataAccess,build_crsp_flags
from method.price_variants import intermediate_signal
from method.signals import formation_dates
from method.monthly_long_only import long_weights
from method.engine import run,PathIncomplete
from method.missing_returns import MissingReturnHandler,UnresolvedReturn
from method.volatility_target import target_15pct
from method.report import metrics

START='2010-01-01';END='2020-12-31';WARM='2008-01-01'

class StrictObserved(MissingReturnHandler):
    """Observed P1 recovery with explicit action audit and provisional NAV."""
    require_action_audit=True

def benchmark_holdings(members,dates):
    """Count distinct historical constituent PERMNOs at each trading date."""
    values=[]
    for d in dates:
        x=members.loc[(members.mbrstartdt<=d)&(members.mbrenddt>=d),'permno']
        values.append(x.nunique())
    return float(np.mean(values)) if values else np.nan

def run_experiment():
    t=time.monotonic();out=ROOT/'result/experiments/intermediate_long_history'
    if out.exists() and any(out.iterdir()):out=out/f'run_{time.time_ns()}'
    out.mkdir(parents=True,exist_ok=True);log=ROOT/'logs'/f'intermediate_history_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(log)])
    config=json.loads((ROOT/'config/settings.json').read_text());os.environ.setdefault('PGOPTIONS','-c statement_timeout=300000')
    cal=xc.get_calendar('XNYS',start=WARM,end=END).schedule;calendar=pd.DatetimeIndex(cal.index).tz_localize(None);closes=pd.Series(pd.to_datetime(cal.close,utc=True).values,index=calendar).dt.tz_localize('UTC')
    db=DataAccess(config['wrds_user'])
    try:
        logging.info('Querying historical universe and daily CRSP data')
        members=db.membership(WARM,END);prices=db.prices(WARM,END,WARM,audit_actions=True);dl=db.delistings(WARM,END,members.permno.dropna().astype(int).unique().tolist());benchmark=db.benchmark(WARM,END);rf=db.riskfree(START,END)
    finally:db.close()
    returns=prices.pivot(index='dlycaldt',columns='permno',values='dlyret').reindex(calendar).astype(float)
    retx=prices.pivot(index='dlycaldt',columns='permno',values='dlyretx').reindex(calendar).astype(float)
    delisted=prices.assign(flag=prices.dlydelflg.eq('Y')).pivot(index='dlycaldt',columns='permno',values='flag').reindex(calendar).fillna(False).astype(bool)
    prices['can_trade']=prices.dlyprc.notna()&(prices.dlyprc.astype(float)>0)&prices.tradingstatusflg.eq('A')&prices.dlydelflg.ne('Y')
    tradable=prices.pivot(index='dlycaldt',columns='permno',values='can_trade').reindex(calendar).fillna(False).astype(bool);flags=build_crsp_flags(prices,dl);del prices
    levels=benchmark.assign(datadate=pd.to_datetime(benchmark.datadate)).set_index('datadate').prccddiv.astype(float).sort_index();bench=levels.pct_change(fill_method=None).loc[START:END]
    risk=rf.assign(date=pd.to_datetime(rf.date)).set_index('date').rf.astype(float);required=calendar[(calendar>=pd.Timestamp(START))&(calendar<=pd.Timestamp(END))]
    if bench.reindex(required).isna().any() or risk.reindex(required).isna().any():raise ValueError('Incomplete benchmark or risk-free data')
    marks=formation_dates(calendar,'monthly');first=calendar[calendar<pd.Timestamp(START)][-1];dates=marks[(marks>=first)&(marks<=pd.Timestamp(END))]
    logging.info('Computing shared Intermediate scores')
    panels={'intermediate':pd.DataFrame.from_dict({d:intermediate_signal(returns,calendar,d) for d in dates},orient='index').reindex(calendar)}
    paths={};events=[];diagnostics=[];status={};holdings={}
    for freq in ('monthly',):
        sched=dates if freq=='monthly' else dates[dates.month==12]
        for mode in ('full','vol15'):
            logging.info('Running %s %s',freq,mode);handler=StrictObserved(max_stale_sessions=config['missing_returns']['max_stale_sessions']);counts={};error=''
            def observe(date,before,after,pnl,nav,prev):counts[date]=len(after)
            try:
                frame,diag=run(returns,tradable,members,panels,calendar,closes,str(first.date()),END,kind='intermediate',frequency='monthly',cost=.001,borrow=0.,price_returns=retx,delisted=delisted,missing_handler=handler,crsp_flags=flags,weight_selector=long_weights,fully_funded=True,formation_schedule=sched,target_transform=target_15pct(returns) if mode=='vol15' else None,observer=observe)
                state='complete'
            except PathIncomplete as exc:
                frame,diag,state,error=exc.frame,exc.diagnostics,'incomplete',str(exc)
            frame=frame.loc[START:END] if len(frame) else frame;key=(freq,mode);paths[key]=frame;status[key]=(state,error);holdings[key]=pd.Series(counts,dtype=float)
            events.extend(dict(frequency=freq,mode=mode,**e) for e in handler.events)
            if len(diag):diagnostics.extend(diag.assign(frequency=freq,mode=mode).to_dict('records'))
    rows=[];annual=[];nav=[]
    bf=pd.DataFrame({'pending_valuation':False,'net':bench,'gross':bench,'turnover':0.,'trade_cost':0.,'borrow_cost':0.,'gross_exposure':1.,'net_exposure':1.});paths[('sp500','benchmark')]=bf;status[('sp500','benchmark')]=('complete','')
    for key,frame in paths.items():
        freq,mode=key;state,error=status[key];is_complete=frame.index.equals(required);entry=dict(frequency=freq,mode=mode,status=state,error=error,provisional_valuation_days=int(frame.pending_valuation.sum()) if len(frame) else 0,valuation_basis='includes provisional prior-mark valuation' if len(frame) and frame.pending_valuation.any() else 'observed',sample_end=str(frame.index.max().date()) if len(frame) else None)
        if len(frame)>1:
            x=metrics(frame,bench,risk);entry['partial_cumulative_return']=x['cumulative_return']
            if is_complete:
                years=(pd.Timestamp(END)-pd.Timestamp(START)).days/365.2425;entry.update(x);entry['cagr']=(1+x['cumulative_return'])**(1/years)-1;entry['calmar']=entry['cagr']/abs(x['max_drawdown']) if x['max_drawdown']<0 else np.nan;entry['annualized_turnover']=x['turnover']/years;entry['average_holdings']=holdings[key].reindex(frame.index).mean() if key in holdings else benchmark_holdings(members,frame.index)
            nav.append(pd.DataFrame(dict(date=frame.index,frequency=freq,mode=mode,pending_valuation=frame.pending_valuation.to_numpy(),nav=100*(1+frame.net).cumprod().to_numpy())))
        rows.append(entry)
        for year in range(2010,2021):
            dates_y=required[required.year==year];g=frame.loc[frame.index.year==year] if len(frame) else frame;complete=len(g)>0 and g.index.equals(dates_y)
            row=dict(year=year,frequency=freq,mode=mode,provisional_valuation_days=int(g.pending_valuation.sum()) if len(g) else 0,status='complete' if complete else ('incomplete' if len(g) else 'unavailable'),days=len(g))
            if complete and len(g)>1:
                m=metrics(g,bench,risk);row.update({k:m[k] for k in ('cumulative_return','volatility','sharpe','max_drawdown','alpha','beta','turnover','trade_cost','average_gross')})
            annual.append(row)
    pd.DataFrame(rows).to_csv(out/'period_metrics.csv',index=False);pd.DataFrame(annual).to_csv(out/'annual_metrics.csv',index=False);pd.DataFrame(events).to_csv(out/'missing_events.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/'execution_diagnostics.csv',index=False)
    if nav:pd.concat(nav).to_csv(out/'derived_daily_nav.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,5))
    for (freq,mode),frame in paths.items():
        if len(frame): (100*(1+frame.net).cumprod()).plot(ax=ax,label=f'{freq} {mode}'+(' incomplete' if status[(freq,mode)][0]!='complete' else ''))
    ax.set_title('2010–2020 paths: prior-mark valuations are provisional');ax.set_ylabel('NAV, initial 100');ax.legend();fig.tight_layout();fig.savefig(out/'nav_strict.png',dpi=140);plt.close(fig)
    summarize(out)
    stats=dict(start=START,end=END,elapsed_seconds=round(time.monotonic()-t,2),wrds_query_count=len(db.query_stats),wrds_queries=db.query_stats,log_file=str(log),vol_target='15% annualized, 63 prior sessions, minimum 42 complete, selected basket, no leverage, cash earns zero',strict='temporary prior-mark valuation is provisional; P1 requires corporate action audit; no assumed delisting settlement')
    (out/'run_summary.json').write_text(json.dumps(stats,indent=2,default=str));print(json.dumps({'output':str(out),**stats},default=str),flush=True)

def summarize(out):
    annual=pd.read_csv(out/'annual_metrics.csv');benchmark=annual[(annual.frequency=='sp500')&(annual['mode']=='benchmark')][['year','cumulative_return']].rename(columns={'cumulative_return':'market_return'})
    a=annual[annual.frequency!='sp500'].merge(benchmark,on='year');a['market_environment']=np.where(a.market_return<0,'negative_year','nonnegative_year');a['excess_return']=a.cumulative_return-a.market_return
    a.to_csv(out/'annual_market_comparison.csv',index=False)
    valid=a[a.status=='complete'];stats=valid.groupby(['frequency','mode','market_environment']).agg(years=('year','count'),mean_strategy_return=('cumulative_return','mean'),median_strategy_return=('cumulative_return','median'),mean_market_return=('market_return','mean'),mean_excess_return=('excess_return','mean'),positive_year_fraction=('cumulative_return',lambda x:(x>0).mean())).reset_index();stats.to_csv(out/'market_environment_summary.csv',index=False)
    distribution=valid.groupby(['frequency','mode']).agg(years=('year','count'),mean_return=('cumulative_return','mean'),median_return=('cumulative_return','median'),std_return=('cumulative_return','std'),min_return=('cumulative_return','min'),max_return=('cumulative_return','max'),positive_year_fraction=('cumulative_return',lambda x:(x>0).mean())).reset_index();distribution.to_csv(out/'annual_return_distribution.csv',index=False)
    years=np.arange(2010,2021);fig,ax=plt.subplots(figsize=(13,5))
    for i,(freq,mode,label) in enumerate([('monthly','full','Monthly full'),('monthly','vol15','Monthly 15% target'),('annual','full','Annual full'),('annual','vol15','Annual 15% target'),('sp500','benchmark','S&P 500 TR')]):
        x=annual[(annual.frequency==freq)&(annual['mode']==mode)].set_index('year').reindex(years).cumulative_return
        ax.bar(np.arange(len(years))+(i-2)*.16,100*x,.16,label=label)
    ax.set_xticks(np.arange(len(years)),years,rotation=45);ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Annual net return (%)');ax.set_title('Strict paths; missing bars are incomplete or unavailable');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'annual_returns.png',dpi=140);plt.close(fig)

if __name__=='__main__':run_experiment()
