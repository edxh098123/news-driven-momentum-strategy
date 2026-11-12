"""Ten monthly long-only momentum variants; WRDS input remains in memory."""
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
from method.monthly_long_only import build_signals,long_weights
from method.engine import run,PathIncomplete
from method.missing_returns import MissingReturnHandler
from method.report import metrics

START='2021-01-01';END='2025-12-31';WARM='2015-01-01'
KINDS=('mom_12_1','mom_12_2','intermediate','residual','continuous','kalman_12_1','kalman_12_2','kalman_12_7','kalman_residual','kalman_continuous')
PAIRS=list(zip(KINDS[:5],KINDS[5:]));SCENARIOS=(('strict',None),('loss_30',-.30),('loss_55',-.55),('loss_100',-1.))

def main():
    began=time.monotonic();out=ROOT/'result/experiments/monthly_long_only_comparison'
    if out.exists() and any(out.iterdir()):out=out/f'run_{time.time_ns()}'
    out.mkdir(parents=True,exist_ok=True)
    log=ROOT/'logs'/f'monthly_long_only_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(log)])
    cfg=json.loads((ROOT/'config/settings.json').read_text());os.environ.setdefault('PGOPTIONS','-c statement_timeout=300000')
    cal=xc.get_calendar('XNYS',start=WARM,end=END).schedule;calendar=pd.DatetimeIndex(cal.index).tz_localize(None)
    closes=pd.Series(pd.to_datetime(cal.close,utc=True).values,index=calendar).dt.tz_localize('UTC')
    db=DataAccess(cfg['wrds_user'])
    try:
        logging.info('Querying historical universe, CRSP, benchmark and FF3')
        members=db.membership(WARM,END); prices=db.prices(WARM,END,WARM)
        dl=db.delistings(WARM,END,members.permno.dropna().astype(int).unique().tolist())
        benchmark=db.benchmark(WARM,END);rf=db.riskfree(START,END)
        factors=db.query('select date,mktrf,smb,hml,rf from ff.factors_monthly where date between %(start)s and %(end)s order by date',dict(start=WARM,end=END))
    finally:db.close()
    factors=factors.set_index(pd.to_datetime(factors.pop('date')))
    returns=prices.pivot(index='dlycaldt',columns='permno',values='dlyret').reindex(calendar).astype(float)
    retx=prices.pivot(index='dlycaldt',columns='permno',values='dlyretx').reindex(calendar).astype(float)
    delisted=prices.assign(flag=prices.dlydelflg.eq('Y')).pivot(index='dlycaldt',columns='permno',values='flag').reindex(calendar).fillna(False).astype(bool)
    prices['can_trade']=prices.dlyprc.notna()&(prices.dlyprc.astype(float)>0)&prices.tradingstatusflg.eq('A')&prices.dlydelflg.ne('Y')
    tradable=prices.pivot(index='dlycaldt',columns='permno',values='can_trade').reindex(calendar).fillna(False).astype(bool)
    flags=build_crsp_flags(prices,dl);del prices
    levels=benchmark.assign(datadate=pd.to_datetime(benchmark.datadate)).set_index('datadate').prccddiv.astype(float).sort_index()
    bench=levels.pct_change(fill_method=None).loc[START:END];risk=rf.assign(date=pd.to_datetime(rf.date)).set_index('date').rf.astype(float)
    required=calendar[(calendar>=pd.Timestamp(START))&(calendar<=pd.Timestamp(END))]
    if bench.reindex(required).isna().any() or risk.reindex(required).isna().any():raise ValueError('Incomplete benchmark or risk-free series')
    logging.info('Building ten formation signals once')
    formation_start=calendar[calendar<pd.Timestamp(START)][-1].strftime('%Y-%m-%d')
    panels=build_signals(returns,calendar,members,tradable,factors,formation_start,END)
    paths={};summary=[];events=[];diagnostics=[]
    for kind in KINDS:
        strict=None;checkpoint=None
        for scenario,loss in SCENARIOS:
            logging.info('Running %s %s',kind,scenario);t=time.monotonic();handler=MissingReturnHandler(loss,max_stale_sessions=cfg['missing_returns']['max_stale_sessions']);error=''
            if scenario!='strict' and strict is not None and strict[1]=='complete':
                frame,status,ev,diag,error=strict;status='scenario_complete';ev=[{**e,'scenario':loss} for e in ev]
            else:
                try:
                    frame,diag=run(returns,tradable,members,panels,calendar,closes,formation_start,END,kind=kind,count=10,frequency='monthly',cost=.001,borrow=0.,price_returns=retx,delisted=delisted,missing_handler=handler,crsp_flags=flags,checkpoint=checkpoint if scenario!='strict' else None,weight_selector=long_weights,fully_funded=True)
                    status='complete' if scenario=='strict' else 'scenario_complete'
                except PathIncomplete as ex:
                    frame,diag,status,error=ex.frame,ex.diagnostics,'incomplete',str(ex)
                    if scenario=='strict':checkpoint=ex.checkpoint
                ev=handler.events
                if len(frame): frame=frame.loc[START:END]
            if scenario=='strict':strict=(frame,status,ev,diag,error)
            events.extend({'strategy':kind,'scenario_name':scenario,**e} for e in ev)
            if not diag.empty:diagnostics.extend(diag.assign(strategy=kind,scenario=scenario).to_dict('records'))
            row=dict(strategy=kind,scenario=scenario,status=status,error=error,seconds=round(time.monotonic()-t,2),sample_end=str(frame.index.max().date()) if len(frame) else None)
            if len(frame)>1:paths[(kind,scenario)]=frame;row.update(metrics(frame,bench,risk))
            summary.append(row)
    bf=pd.DataFrame({'net':bench,'gross':bench,'turnover':0.,'trade_cost':0.,'borrow_cost':0.,'gross_exposure':1.,'net_exposure':1.})
    paths[('sp500','observed')]=bf;summary.append(dict(strategy='sp500',scenario='observed',status='complete',sample_end=END,**metrics(bf,bench,risk)))
    pd.DataFrame(events).to_csv(out/'missing_events.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/'formation_diagnostics.csv',index=False)
    write_results(out,paths,pd.DataFrame(summary),bench,risk,required)
    stats={'start':START,'end':END,'warm_start':WARM,'first_formation':formation_start,'elapsed_seconds':round(time.monotonic()-began,2),'wrds_query_count':len(db.query_stats),'wrds_queries':db.query_stats,'log_file':str(log),'note':'Strict incomplete paths stop. Scenario returns are assumptions; factor vintage may be revised.'}
    (out/'run_summary.json').write_text(json.dumps(stats,indent=2,default=str));logging.info('Output %s elapsed %.2fs',out,stats['elapsed_seconds'])

def write_results(out,paths,summary,bench,risk,required):
    summary.to_csv(out/'metrics_partial_and_full.csv',index=False)
    annual=[];daily=[]
    for (kind,scenario),frame in paths.items():
        daily.append(pd.DataFrame({'date':frame.index,'strategy':kind,'scenario':scenario,'nav':100*(1+frame.net).cumprod().to_numpy()}))
        for year in range(2021,2026):
            expected=required[required.year==year];g=frame.loc[frame.index.year==year];complete=g.index.equals(expected)
            status=('scenario_complete' if scenario.startswith('loss_') else 'complete') if complete else ('incomplete' if len(g) else 'unavailable')
            row=dict(strategy=kind,scenario=scenario,year=year,status=status,days=len(g))
            if len(g)>1 and complete:
                stat=metrics(g,bench,risk); b=(1+bench.reindex(expected)).prod()-1
                row.update({k:stat[k] for k in ('cumulative_return','cumulative_gross_return','volatility','sharpe','max_drawdown','turnover','trade_cost','alpha','beta')});row['excess_return']=stat['cumulative_return']-b
            annual.append(row)
    annual=pd.DataFrame(annual);annual.to_csv(out/'annual_performance.csv',index=False);pd.concat(daily).to_csv(out/'derived_daily_nav.csv',index=False)
    comp=summary.copy();complete=comp.status.isin(['complete','scenario_complete'])&comp.sample_end.eq(END)
    comp['partial_cumulative_return']=comp.cumulative_return.where(~complete)
    for c in ('cumulative_return','cumulative_gross_return','annualized_return','volatility','sharpe','max_drawdown','turnover','trade_cost','alpha','beta'):comp.loc[~complete,c]=np.nan
    comp['excess_cumulative_return']=comp.cumulative_return-((1+bench).prod()-1)
    comp.to_csv(out/'five_year_comparison.csv',index=False)
    pairs=[]
    for scenario,_ in SCENARIOS:
        for raw,kalman in PAIRS:
            a=comp[(comp.strategy==raw)&(comp.scenario==scenario)].iloc[0];b=comp[(comp.strategy==kalman)&(comp.scenario==scenario)].iloc[0]
            pairs.append(dict(scenario=scenario,raw=raw,kalman=kalman,raw_status=a.status,kalman_status=b.status,net_return_raw=a.cumulative_return,net_return_kalman=b.cumulative_return,net_return_difference=b.cumulative_return-a.cumulative_return,turnover_raw=a.turnover,turnover_kalman=b.turnover,turnover_difference=b.turnover-a.turnover))
    pd.DataFrame(pairs).to_csv(out/'kalman_pairs.csv',index=False)
    ranking=comp[comp.strategy.ne('sp500')].copy();ranking['rank']=ranking.groupby('scenario').cumulative_return.rank(ascending=False,method='min');ranking['note']='exploratory in-sample; incomplete paths not ranked';ranking.sort_values(['scenario','rank']).to_csv(out/'exploratory_ranking.csv',index=False)
    make_plots(out,paths,annual)

def make_plots(out,paths,annual):
    for scenario,_ in SCENARIOS:
        for group,items in [('original',KINDS[:5]),('kalman',KINDS[5:])]+[(f'pair_{a}',(a,b)) for a,b in PAIRS]:
            fig,ax=plt.subplots(figsize=(10,5))
            for kind in (*items,'sp500'):
                sc='observed' if kind=='sp500' else scenario;frame=paths.get((kind,sc))
                if frame is None or frame.empty:continue
                nav=100*(1+frame.net).cumprod();nav=pd.concat([pd.Series([100.],index=[pd.Timestamp('2020-12-31')]),nav]);nav.plot(ax=ax,label=kind+(' incomplete' if frame.index.max()<pd.Timestamp(END) else ''))
            ax.set_title(f'{group}: {scenario}');ax.set_ylabel('NAV (initial 100)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'nav_{group}_{scenario}.png',dpi=130);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True);idx=np.arange(5);years=np.arange(2021,2026)
        for ax,items in zip(axes,[KINDS[:5],KINDS[5:]]):
            for i,kind in enumerate((*items,'sp500')):
                sc='observed' if kind=='sp500' else scenario;v=annual[(annual.strategy==kind)&(annual.scenario==sc)].set_index('year').reindex(years).cumulative_return
                ax.bar(idx+(i-2.5)*.15,100*v,.15,label=kind)
            ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Annual net return (%)');ax.legend(fontsize=7,ncol=3)
        axes[-1].set_xticks(idx,years);fig.suptitle(f'Annual net returns: {scenario}; blank = incomplete');fig.tight_layout();fig.savefig(out/f'annual_net_{scenario}.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()
