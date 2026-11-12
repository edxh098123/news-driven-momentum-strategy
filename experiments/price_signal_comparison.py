"""2021–2025 price-only experiment; raw WRDS inputs remain in memory."""
import json, logging, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from data.collect import DataAccess,build_crsp_flags
from method.engine import run,PathIncomplete
from method.missing_returns import MissingReturnHandler
from method.price_variants import signal_panels
from method.report import metrics

START='2021-01-01'; END='2025-12-31'; WARM='2016-12-01'
KINDS=('price','residual','intermediate','continuous'); SCENARIOS=(('strict',None),('loss_30',-.30),('loss_55',-.55),('loss_100',-1.))

def main(experiment='price_signal_comparison', kinds=KINDS, builder=signal_panels, warm=WARM, use_factors=True, extra_report=None):
    began=time.monotonic(); out=ROOT/'result/experiments'/experiment
    if out.exists() and any(out.iterdir()): out=out/f'run_{time.time_ns()}'
    out.mkdir(parents=True,exist_ok=True)
    log=ROOT/'logs'/f'{experiment}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(log)])
    cfg=json.loads((ROOT/'config/settings.json').read_text()); os.environ.setdefault('PGOPTIONS','-c statement_timeout=300000')
    nyse=xc.get_calendar('XNYS',start=warm,end=END); schedule=nyse.schedule
    calendar=pd.DatetimeIndex(schedule.index).tz_localize(None); closes=pd.Series(pd.to_datetime(schedule.close,utc=True).values,index=calendar).dt.tz_localize('UTC')
    db=DataAccess(cfg['wrds_user'])
    try:
        logging.info('Querying membership, prices, delistings, benchmark and FF3 factors')
        members=db.membership(warm,END); prices=db.prices(warm,END,warm)
        dl=db.delistings(warm,END,members.permno.dropna().astype(int).unique().tolist())
        benchmark=db.benchmark(warm,END); rf=db.riskfree(START,END)
        factors=db.query('select date,mktrf,smb,hml,rf from ff.factors_monthly where date between %(start)s and %(end)s order by date',dict(start=warm,end=END)) if use_factors else None
    finally: db.close()
    if factors is not None: factors=factors.set_index(pd.to_datetime(factors.pop('date')))
    returns=prices.pivot(index='dlycaldt',columns='permno',values='dlyret').reindex(calendar).astype(float)
    retx=prices.pivot(index='dlycaldt',columns='permno',values='dlyretx').reindex(calendar).astype(float)
    delisted=prices.assign(flag=prices.dlydelflg.eq('Y')).pivot(index='dlycaldt',columns='permno',values='flag').reindex(calendar).fillna(False).astype(bool)
    prices['can_trade']=prices.dlyprc.notna()&(prices.dlyprc.astype(float)>0)&prices.tradingstatusflg.eq('A')&prices.dlydelflg.ne('Y')
    tradable=prices.pivot(index='dlycaldt',columns='permno',values='can_trade').reindex(calendar).fillna(False).astype(bool)
    flags=build_crsp_flags(prices,dl); del prices
    levels=benchmark.assign(datadate=pd.to_datetime(benchmark.datadate)).set_index('datadate').prccddiv.astype(float).sort_index()
    bench=levels.pct_change(fill_method=None).loc[START:END]; risk=rf.assign(date=pd.to_datetime(rf.date)).set_index('date').rf.astype(float)
    required=calendar[(calendar>=pd.Timestamp(START))&(calendar<=pd.Timestamp(END))]
    if bench.reindex(required).isna().any() or risk.reindex(required).isna().any(): raise ValueError('Benchmark/risk-free coverage incomplete')
    logging.info('Building point-in-time signals')
    panels=builder(returns,calendar,members,tradable,factors,START,END)
    if extra_report is not None: extra_report(out,panels,calendar,members,tradable,START,END)
    paths={}; result=[]; events=[]; diagrows=[]; price_cache={}
    for kind in kinds:
        checkpoint=None; strict=None
        for name,loss in SCENARIOS:
            logging.info('Running %s %s',kind,name); t=time.monotonic(); handler=MissingReturnHandler(loss,max_stale_sessions=cfg['missing_returns']['max_stale_sessions']); error=''
            if name!='strict' and strict is not None and not any(e['kind']=='unresolved_delisting' for e in strict[2]):
                frame,status,ev,diagnostics,error=strict; ev=[{**e,'scenario':loss} for e in ev]
                if status=='complete': status='scenario_complete'
            else:
                try:
                    frame,diagnostics=run(returns,tradable,members,panels,calendar,closes,START,END,kind=kind,price_returns=retx,delisted=delisted,price_cache=price_cache,missing_handler=handler,crsp_flags=flags,checkpoint=checkpoint if name!='strict' else None)
                    status='complete' if name=='strict' else 'scenario_complete'
                except PathIncomplete as ex:
                    frame,diagnostics,status,error=ex.frame,ex.diagnostics,'incomplete',str(ex)
                    if name=='strict': checkpoint=ex.checkpoint
                ev=handler.events
            if name=='strict': strict=(frame,status,ev,diagnostics,error)
            events.extend({'strategy':kind,'scenario_name':name,**e} for e in ev)
            if not diagnostics.empty: diagrows.extend(diagnostics.assign(strategy=kind,scenario=name).to_dict('records'))
            row=dict(strategy=kind,scenario=name,status=status,error=error,seconds=round(time.monotonic()-t,2),sample_end=str(frame.index.max().date()) if len(frame) else None)
            if len(frame)>1:
                paths[(kind,name)]=frame; row.update(metrics(frame,bench,risk))
            result.append(row)
    bf=pd.DataFrame({'net':bench,'gross':bench,'turnover':0.,'trade_cost':0.,'borrow_cost':0.,'gross_exposure':1.,'net_exposure':1.})
    paths[('sp500','observed')]=bf; result.append(dict(strategy='sp500',scenario='observed',status='complete',sample_end=END,**metrics(bf,bench,risk)))
    pd.DataFrame(result).to_csv(out/'metrics.csv',index=False); pd.DataFrame(events).to_csv(out/'missing_events.csv',index=False); pd.DataFrame(diagrows).to_csv(out/'formation_diagnostics.csv',index=False)
    annual=[]; legs=[]
    for (kind,scenario),frame in paths.items():
        for year in range(2021,2026):
            expected=required[required.year==year]; g=frame.loc[frame.index.year==year]; complete=g.index.equals(expected)
            status=('scenario_complete' if scenario.startswith('loss_') else 'complete') if complete else ('incomplete' if len(g) else 'unavailable')
            row=dict(strategy=kind,scenario=scenario,year=year,status=status,days=len(g))
            if len(g)>1:
                stats=metrics(g,bench,risk); row.update({k:(stats[k] if complete else np.nan) for k in ('cumulative_return','cumulative_gross_return','sharpe','volatility','max_drawdown','turnover','trade_cost','borrow_cost')})
                if kind!='sp500':
                    prior=(1+g.net).cumprod().shift(1,fill_value=1.)
                    legs.append(dict(strategy=kind,scenario=scenario,year=year,status=status,long_contribution=(prior*g['long']).sum() if complete else np.nan,short_contribution=(prior*g['short']).sum() if complete else np.nan,net_return=stats['cumulative_return'] if complete else np.nan))
            annual.append(row)
    annual=pd.DataFrame(annual); legs=pd.DataFrame(legs); annual.to_csv(out/'annual_performance.csv',index=False); legs.to_csv(out/'annual_leg_attribution.csv',index=False)
    five_year_table(out)
    plot(out,paths,annual,legs,kinds)
    summary={'start':START,'end':END,'warm_start':warm,'elapsed_seconds':round(time.monotonic()-began,2),'wrds_query_count':len(db.query_stats),'wrds_queries':db.query_stats,'log':str(log),'factor_vintage':'WRDS current historical series; monthly publication vintages unavailable' if use_factors else None,'note':'Strict incomplete paths stop; loss scenarios are hypothetical.'}
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2,default=str)); logging.info('Output %s; elapsed %.1f seconds',out,summary['elapsed_seconds'])

def five_year_table(out):
    """Suppress full-period figures for incomplete paths and link annual legs."""
    m=pd.read_csv(out/'metrics.csv'); a=pd.read_csv(out/'annual_performance.csv'); legs=pd.read_csv(out/'annual_leg_attribution.csv')
    rows=[]
    for row in m.to_dict('records'):
        kind=row['strategy']; scenario=row['scenario']; complete=row['status'] in ('complete','scenario_complete') and str(row['sample_end'])==END
        item={k:row.get(k) for k in ('strategy','scenario','status','sample_end','cumulative_return','cumulative_gross_return','annualized_return','volatility','sharpe','max_drawdown','turnover','trade_cost','borrow_cost')}
        if not complete:
            item['partial_net_return']=item['cumulative_return']
            for k in ('cumulative_return','cumulative_gross_return','annualized_return','volatility','sharpe','max_drawdown','turnover','trade_cost','borrow_cost'): item[k]=np.nan
        if complete and kind!='sp500':
            ar=a[(a.strategy==kind)&(a.scenario==scenario)].sort_values('year'); lg=legs[(legs.strategy==kind)&(legs.scenario==scenario)].sort_values('year')
            if len(ar)==len(lg)==5:
                scale=(1+ar.cumulative_return).cumprod().shift(1,fill_value=1.).to_numpy()
                item['long_contribution']=(scale*lg.long_contribution.to_numpy()).sum()
                item['short_contribution']=(scale*lg.short_contribution.to_numpy()).sum()
        rows.append(item)
    pd.DataFrame(rows).to_csv(out/'five_year_comparison.csv',index=False)

def plot(out,paths,annual,legs,kinds):
    names={'price':'Baseline','residual':'Residual','intermediate':'Intermediate','continuous':'Continuous','kalman_12_2':'Kalman 12–2','kalman_12_7':'Kalman 12–7','sp500':'S&P 500 TR'}; years=np.arange(2021,2026); idx=np.arange(5)
    for scenario in ['strict','loss_30','loss_55','loss_100']:
        fig,ax=plt.subplots(figsize=(11,5))
        for i,kind in enumerate((*kinds,'sp500')):
            s='observed' if kind=='sp500' else scenario
            vals=annual[(annual.strategy==kind)&(annual.scenario==s)].set_index('year').reindex(years).cumulative_return
            ax.bar(idx+(i-2)*.16,100*vals,width=.16,label=names[kind])
        ax.axhline(0,color='black',lw=.6);ax.set_xticks(idx,years);ax.set_ylabel('Annual net return (%)');ax.set_title(f'Annual net returns — {scenario} (missing bars: incomplete)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'annual_net_{scenario}.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,5))
        for kind in (*kinds,'sp500'):
            s='observed' if kind=='sp500' else scenario; frame=paths.get((kind,s))
            if frame is None or frame.empty:continue
            v=100*(1+frame.net).cumprod();v=pd.concat([pd.Series([100.],index=[pd.Timestamp('2020-12-31')]),v]); v.plot(ax=ax,label=names[kind]+(' (incomplete)' if frame.index.max()<pd.Timestamp(END) else ''))
        ax.set_title(f'Net asset value — {scenario}');ax.set_ylabel('Initial NAV = 100');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'net_value_{scenario}.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
        for ax,kind in zip(axes.flat,kinds):
            g=legs[(legs.strategy==kind)&(legs.scenario==scenario)].set_index('year').reindex(years)
            for off,col,label in [(-.2,'long_contribution','Long'),(.2,'short_contribution','Short')]:ax.bar(idx+off,100*g[col],.4,label=label)
            ax.axhline(0,color='black',lw=.6);ax.set_xticks(idx,years);ax.set_title(names[kind]);ax.set_ylabel('Contribution to initial annual NAV (%)')
        axes.flat[0].legend();fig.suptitle(f'Gross leg contributions — {scenario}');fig.tight_layout();fig.savefig(out/f'legs_{scenario}.png',dpi=140);plt.close(fig)

if __name__=='__main__': main()
