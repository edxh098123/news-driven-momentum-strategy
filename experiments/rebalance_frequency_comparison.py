"""Frequency-only experiment, formal signals/engine; all WRDS inputs in memory."""
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
from method.frequency_experiment import schedules,matched_target
from method.engine import run,PathIncomplete
from method.missing_returns import MissingReturnHandler
from method.report import metrics
from experiments.monthly_long_only_comparison import KINDS

START='2021-01-01';END='2025-12-31';WARM='2015-01-01';FREQ=('weekly','monthly','quarterly','annual');FOCUS=('intermediate','kalman_12_2','mom_12_1')

class Attribution:
    def __init__(self):self.contrib={};self.duration={};self.runs=[];self.daily=[];self.previous=set();self.start_nav=None
    def __call__(self,date,before,after,pnl,nav,prev):
        if date<pd.Timestamp(START):return
        if self.start_nav is None:self.start_nav=prev
        for p,v in pnl.items():self.contrib[(int(p),date.year)]=self.contrib.get((int(p),date.year),0.)+float(v)/self.start_nav
        present=set(after.index);weights=after/nav
        for p in self.previous-present:self.runs.append((int(p),self.duration.pop(p),False))
        for p in present:self.duration[p]=self.duration.get(p,0)+1
        self.previous=present
        self.daily.append({'date':date,'hhi':float((weights**2).sum()),'max_weight':float(weights.max()) if len(weights) else 0.,'top3_weight':float(weights.nlargest(3).sum()),'cash_weight':1-float(weights.sum()),'names':len(weights)})
    def finish(self):
        return self.runs+[(int(p),d,True) for p,d in self.duration.items()]

def main():
    began=time.monotonic();out=ROOT/'result/experiments/rebalance_frequency_comparison'
    if out.exists() and any(out.iterdir()):out=out/f'run_{time.time_ns()}'
    out.mkdir(parents=True,exist_ok=True);log=ROOT/'logs'/f'frequency_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(log)])
    cfg=json.loads((ROOT/'config/settings.json').read_text());os.environ.setdefault('PGOPTIONS','-c statement_timeout=300000')
    schedule=xc.get_calendar('XNYS',start=WARM,end=END).schedule;calendar=pd.DatetimeIndex(schedule.index).tz_localize(None);closes=pd.Series(pd.to_datetime(schedule.close,utc=True).values,index=calendar).dt.tz_localize('UTC')
    db=DataAccess(cfg['wrds_user'])
    try:
        members=db.membership(WARM,END);prices=db.prices(WARM,END,WARM);dl=db.delistings(WARM,END,members.permno.dropna().astype(int).unique().tolist());benchmark=db.benchmark(WARM,END);rf=db.riskfree(START,END)
        names=db.query('select permno,ticker,issuernm,namedt,nameenddt from crsp.stocknames_v2 where permno=any(%(ids)s) and namedt<=%(end)s and nameenddt >= %(start)s',dict(ids=members.permno.dropna().astype(int).unique().tolist(),start='2020-01-01',end=END))
        factors=db.query('select date,mktrf,smb,hml,rf from ff.factors_monthly where date between %(start)s and %(end)s order by date',dict(start=WARM,end=END))
    finally:db.close()
    factors=factors.set_index(pd.to_datetime(factors.pop('date')))
    returns=prices.pivot(index='dlycaldt',columns='permno',values='dlyret').reindex(calendar).astype(float);retx=prices.pivot(index='dlycaldt',columns='permno',values='dlyretx').reindex(calendar).astype(float)
    delisted=prices.assign(flag=prices.dlydelflg.eq('Y')).pivot(index='dlycaldt',columns='permno',values='flag').reindex(calendar).fillna(False).astype(bool)
    prices['can_trade']=prices.dlyprc.notna()&(prices.dlyprc.astype(float)>0)&prices.tradingstatusflg.eq('A')&prices.dlydelflg.ne('Y')
    tradable=prices.pivot(index='dlycaldt',columns='permno',values='can_trade').reindex(calendar).fillna(False).astype(bool);flags=build_crsp_flags(prices,dl);del prices
    levels=benchmark.assign(datadate=pd.to_datetime(benchmark.datadate)).set_index('datadate').prccddiv.astype(float).sort_index();allbench=levels.pct_change(fill_method=None).reindex(calendar);bench=allbench.loc[START:END]
    risk=rf.assign(date=pd.to_datetime(rf.date)).set_index('date').rf.astype(float);required=calendar[(calendar>=pd.Timestamp(START))&(calendar<=pd.Timestamp(END))]
    if bench.reindex(required).isna().any() or risk.reindex(required).isna().any():raise ValueError('Benchmark or risk-free coverage incomplete')
    schedules_main=schedules(calendar);forms=set().union(*[set(schedules_main[f]) for f in FREQ]);forms={x for x in forms if pd.Timestamp('2020-01-01')<=x<=pd.Timestamp(END)}
    logging.info('Building shared weekly/monthly signals for %d formation dates',len(forms))
    panels=build_signals(returns,calendar,members,tradable,factors,'2020-01-01',END,formation_schedule=sorted(forms))
    paths={};rows=[];events=[];details=[];attributions={};diag_all=[]
    def execute(kind,freq,phase=1,scaled=False):
        sched=schedules_main[freq] if freq!='annual' or phase==1 else schedules(calendar,phase)['annual']
        first=calendar[calendar<pd.Timestamp(START)][-1]
        sched=pd.DatetimeIndex([first]).union(sched[(sched>=pd.Timestamp(START))&(sched<=pd.Timestamp(END))])
        observer=Attribution() if kind in FOCUS and not scaled and phase==1 else None
        transform=matched_target(returns,allbench) if scaled else None
        checkpoint=None;strict=None
        for label,loss in [('strict',None),('loss_30',-.3),('loss_55',-.55),('loss_100',-1.)]:
            handler=MissingReturnHandler(loss,max_stale_sessions=cfg['missing_returns']['max_stale_sessions']);error='';t=time.monotonic()
            if label!='strict' and strict is not None and strict[1]=='complete':
                frame,status,ev,diag,error=strict;status='scenario_complete';ev=[{**e,'scenario':loss} for e in ev]
            else:
                try:
                    frame,diag=run(returns,tradable,members,panels,calendar,closes,str(first.date()),END,kind=kind,count=10,frequency='monthly',cost=.001,borrow=0.,price_returns=retx,delisted=delisted,missing_handler=handler,crsp_flags=flags,checkpoint=checkpoint if label!='strict' else None,weight_selector=long_weights,fully_funded=True,formation_schedule=sched,target_transform=transform,observer=observer if label=='strict' else None)
                    status='complete' if label=='strict' else 'scenario_complete'
                except PathIncomplete as ex:
                    frame,diag,status,error=ex.frame,ex.diagnostics,'incomplete',str(ex)
                    if label=='strict':checkpoint=ex.checkpoint
                ev=handler.events
                if len(frame):frame=frame.loc[START:END]
            if label=='strict':strict=(frame,status,ev,diag,error)
            key=(kind,freq,phase,'risk_matched' if scaled else 'base',label)
            events.extend({'strategy':kind,'frequency':freq,'phase':phase,'mode':key[3],'scenario_name':label,**e} for e in ev)
            if not diag.empty:diag_all.extend(diag.assign(strategy=kind,frequency=freq,phase=phase,mode=key[3],scenario=label).to_dict('records'))
            record=dict(strategy=kind,frequency=freq,phase=phase,mode=key[3],scenario=label,status=status,error=error,first_formation=str(first.date()),sample_end=str(frame.index.max().date()) if len(frame) else None,seconds=round(time.monotonic()-t,3))
            if len(frame)>1:paths[key]=frame;record.update(metrics(frame,bench,risk))
            rows.append(record)
        if observer is not None and strict and strict[1]=='complete':attributions[(kind,freq)]=observer
    for freq in FREQ:
        for kind in KINDS:
            logging.info('Base %s %s',freq,kind);execute(kind,freq)
    for phase in (4,7,10):
        for kind in KINDS:
            logging.info('Annual phase %s %s',phase,kind);execute(kind,'annual',phase)
    for freq in FREQ:
        for kind in FOCUS:
            logging.info('Risk matched %s %s',freq,kind);execute(kind,freq,scaled=True)
    pd.DataFrame(events).to_csv(out/'missing_events.csv',index=False);pd.DataFrame(diag_all).to_csv(out/'formation_diagnostics.csv',index=False)
    report(out,paths,pd.DataFrame(rows),bench,risk,required,attributions,names)
    stats=dict(start=START,end=END,warm_start=WARM,elapsed_seconds=round(time.monotonic()-began,2),wrds_query_count=len(db.query_stats),wrds_queries=db.query_stats,log_file=str(log),risk_matching='63 past sessions, selected basket historical equal-weight returns, min 42, benchmark vol target, 1x cap, zero-rate cash',factor_vintage='WRDS historical FF3 factors can be revised; release vintages unavailable')
    (out/'run_summary.json').write_text(json.dumps(stats,indent=2,default=str));logging.info('Output %s elapsed %.1fs',out,stats['elapsed_seconds'])

def report(out,paths,summary,bench,risk,required,attributions,names):
    summary.to_csv(out/'metrics_partial_and_full.csv',index=False)
    comp=summary.copy();complete=comp.status.isin(['complete','scenario_complete'])&comp.sample_end.eq(END)
    comp['partial_cumulative_return']=comp.cumulative_return.where(~complete)
    for col in ('cumulative_return','cumulative_gross_return','annualized_return','volatility','sharpe','max_drawdown','turnover','trade_cost','alpha','beta'):comp.loc[~complete,col]=np.nan
    comp['excess_cumulative_return']=comp.cumulative_return-((1+bench).prod()-1);comp.to_csv(out/'five_year_metrics.csv',index=False)
    comp[(comp['mode']=='base')&(comp.phase==1)].assign(rank=lambda x:x.groupby(['scenario','frequency']).cumulative_return.rank(ascending=False,method='min'),note='Exploratory in-sample ranking').sort_values(['scenario','frequency','rank']).to_csv(out/'frequency_rankings.csv',index=False)
    comp[(comp['mode']=='base')&(comp.frequency=='annual')].to_csv(out/'annual_phase_sensitivity.csv',index=False)
    annual=[];nav=[];monthrows=[]
    for key,frame in paths.items():
        kind,freq,phase,mode,scenario=key;nav.append(pd.DataFrame(dict(date=frame.index,strategy=kind,frequency=freq,phase=phase,mode=mode,scenario=scenario,nav=100*(1+frame.net).cumprod().to_numpy())))
        for year in range(2021,2026):
            expected=required[required.year==year];g=frame.loc[frame.index.year==year];ok=g.index.equals(expected)
            row=dict(strategy=kind,frequency=freq,phase=phase,mode=mode,scenario=scenario,year=year,status=('scenario_complete' if scenario!='strict' else 'complete') if ok else ('incomplete' if len(g) else 'unavailable'),days=len(g))
            if ok and len(g)>1:
                x=metrics(g,bench,risk);row.update({k:x[k] for k in ('cumulative_return','cumulative_gross_return','volatility','sharpe','max_drawdown','turnover','trade_cost','alpha','beta')});row['excess_return']=x['cumulative_return']-((1+bench.reindex(expected)).prod()-1)
                if kind in FOCUS and mode=='base' and phase==1 and scenario=='strict':
                    months=g.net.groupby(g.index.to_period('M')).apply(lambda z:(1+z).prod()-1);prior=(1+months).cumprod().shift(1,fill_value=1.);linked=months*prior;positive=linked[linked>0]
                    for m,v in linked.items():monthrows.append(dict(strategy=kind,frequency=freq,year=year,month=str(m),contribution=float(v),top3_positive_share=float(positive.nlargest(3).sum()/positive.sum()) if positive.sum()>0 else np.nan))
            annual.append(row)
    pd.DataFrame(annual).to_csv(out/'annual_performance.csv',index=False);pd.concat(nav).to_csv(out/'derived_daily_nav.csv',index=False);pd.DataFrame(monthrows).to_csv(out/'monthly_contributions.csv',index=False)
    stock=[];holding=[];concentration=[]
    names=names.copy();names['namedt']=pd.to_datetime(names.namedt);names['nameenddt']=pd.to_datetime(names.nameenddt)
    for (kind,freq),obs in attributions.items():
        for (permno,year),value in obs.contrib.items():
            known=names[(names.permno==permno)&(names.namedt<=pd.Timestamp(year,12,31))].sort_values('namedt')
            name=known.iloc[-1] if len(known) else None
            stock.append(dict(strategy=kind,frequency=freq,year=year,permno=permno,ticker=name.ticker if name is not None else None,issuer=name.issuernm if name is not None else None,contribution_to_initial_nav=value))
        for permno,duration,is_open in obs.finish():holding.append(dict(strategy=kind,frequency=freq,permno=permno,holding_sessions=duration,open_at_end=is_open))
        concentration.extend(dict(strategy=kind,frequency=freq,**x) for x in obs.daily)
    pd.DataFrame(stock).to_csv(out/'stock_contributions.csv',index=False);pd.DataFrame(holding).to_csv(out/'holding_periods.csv',index=False);pd.DataFrame(concentration).to_csv(out/'daily_concentration.csv',index=False)
    old=pd.read_csv(ROOT/'result/experiments/monthly_long_only_comparison/run_1794460543886121000/five_year_comparison.csv')
    new=comp[(comp.frequency=='monthly')&(comp.phase==1)&(comp['mode']=='base')];old=old[old.strategy.isin(KINDS)]
    check=new.merge(old,on=['strategy','scenario'],suffixes=('_new','_previous'))
    for col in ('cumulative_return','cumulative_gross_return','turnover','sharpe','max_drawdown'):
        check[col+'_difference']=check[col+'_new']-check[col+'_previous']
    check[['strategy','scenario','status_new','status_previous']+[c+'_difference' for c in ('cumulative_return','cumulative_gross_return','turnover','sharpe','max_drawdown')]].to_csv(out/'monthly_regression_check.csv',index=False)
    make_plots(out,paths,comp,pd.DataFrame(annual),bench)
    summarize_concentration(out)
    extra_plots(out)
    export_verified_benchmark(out)

def make_plots(out,paths,comp,annual,bench):
    subset=comp[(comp.scenario=='strict')&(comp.phase==1)&(comp['mode']=='base')]
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,freq in zip(axes.flat,FREQ):
        x=subset[subset.frequency==freq].sort_values('cumulative_return');ax.barh(x.strategy,100*x.cumulative_return);ax.axvline(100*((1+bench).prod()-1),color='red',label='S&P 500');ax.set_title(freq);ax.set_xlabel('Five-year net return (%)')
    axes.flat[0].legend();fig.tight_layout();fig.savefig(out/'rankings_by_frequency.png',dpi=140);plt.close(fig)
    for kind in FOCUS:
        fig,ax=plt.subplots(figsize=(10,5));(100*(1+bench).cumprod()).plot(ax=ax,label='S&P 500',color='black')
        for freq in FREQ:
            frame=paths.get((kind,freq,1,'base','strict'))
            if frame is not None:(100*(1+frame.net).cumprod()).plot(ax=ax,label=freq+(' incomplete' if frame.index.max()<pd.Timestamp(END) else ''))
        ax.set_title(kind);ax.set_ylabel('NAV (initial 100)');ax.legend();fig.tight_layout();fig.savefig(out/f'nav_{kind}.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,col,title in zip(axes,['cumulative_return','sharpe','max_drawdown'],['Net return','Sharpe','Maximum drawdown']):
        for j,kind in enumerate(FOCUS):
            x=subset[subset.strategy==kind].set_index('frequency').reindex(FREQ);ax.bar(np.arange(4)+(j-1)*.25,x[col],.25,label=kind)
        ax.set_xticks(range(4),FREQ,rotation=20);ax.set_title(title)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'risk_return_by_frequency.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for freq in FREQ:
        x=subset[subset.frequency==freq];ax.scatter(x.turnover,x.cumulative_return,label=freq)
        for r in x.itertuples():ax.annotate(r.strategy,(r.turnover,r.cumulative_return),fontsize=6)
    ax.set_xlabel('Five-year turnover');ax.set_ylabel('Five-year net return');ax.legend();fig.tight_layout();fig.savefig(out/'turnover_vs_return.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for ax,kind in zip(axes,FOCUS):
        for j,freq in enumerate(FREQ):
            x=annual[(annual.strategy==kind)&(annual.frequency==freq)&(annual.phase==1)&(annual['mode']=='base')&(annual.scenario=='strict')].set_index('year').reindex(range(2021,2026));ax.bar(np.arange(5)+(j-1.5)*.2,x.cumulative_return,.2,label=freq)
        ax.set_title(kind);ax.axhline(0,color='black',lw=.6)
    axes[0].legend(ncol=4);axes[-1].set_xticks(range(5),range(2021,2026));fig.suptitle('Annual net returns');fig.tight_layout();fig.savefig(out/'annual_returns_by_frequency.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    for ax,kind in zip(axes,FOCUS):
        (100*(1+bench).cumprod()).plot(ax=ax,label='S&P 500',color='black')
        for freq in FREQ:
            frame=paths.get((kind,freq,1,'risk_matched','strict'))
            if frame is not None:(100*(1+frame.net).cumprod()).plot(ax=ax,label=freq)
        ax.set_title(kind);ax.set_ylabel('NAV')
    axes[0].legend(ncol=5,fontsize=8);fig.suptitle('Causal capped risk matching vs S&P 500');fig.tight_layout();fig.savefig(out/'risk_matched_nav.png',dpi=140);plt.close(fig)

def summarize_concentration(out):
    stocks=pd.read_csv(out/'stock_contributions.csv');hold=pd.read_csv(out/'holding_periods.csv');conc=pd.read_csv(out/'daily_concentration.csv');months=pd.read_csv(out/'monthly_contributions.csv')
    annual_stock=[]
    for (kind,freq,year),g in stocks.groupby(['strategy','frequency','year']):
        positive=g[g.contribution_to_initial_nav>0].sort_values('contribution_to_initial_nav',ascending=False)
        denominator=positive.contribution_to_initial_nav.sum()
        annual_stock.append(dict(strategy=kind,frequency=freq,year=year,top_stock=positive.iloc[0].ticker if len(positive) else None,top_stock_positive_pnl_share=positive.contribution_to_initial_nav.iloc[0]/denominator if denominator>0 else np.nan,top3_positive_pnl_share=positive.contribution_to_initial_nav.head(3).sum()/denominator if denominator>0 else np.nan))
    pd.DataFrame(annual_stock).to_csv(out/'annual_stock_concentration.csv',index=False)
    total=stocks.groupby(['strategy','frequency','permno'],as_index=False).contribution_to_initial_nav.sum()
    names=stocks.sort_values('year').drop_duplicates(['strategy','frequency','permno'],keep='last')[['strategy','frequency','permno','ticker','issuer']]
    total=total.merge(names,on=['strategy','frequency','permno'],how='left');total['rank']=total.groupby(['strategy','frequency']).contribution_to_initial_nav.rank(ascending=False,method='first')
    total.sort_values(['strategy','frequency','rank']).to_csv(out/'stock_contribution_ranking.csv',index=False)
    hg=hold.groupby(['strategy','frequency']).holding_sessions.agg(['count','mean','median','max']).reset_index().rename(columns={'count':'holding_spells','mean':'mean_holding_sessions','median':'median_holding_sessions','max':'max_holding_sessions'})
    cg=conc.groupby(['strategy','frequency'])[['hhi','max_weight','top3_weight','cash_weight','names']].mean().reset_index()
    concentration=cg.merge(hg,on=['strategy','frequency'],how='left');concentration.to_csv(out/'concentration_holding_summary.csv',index=False)
    months.groupby(['strategy','frequency','year'],as_index=False).agg(top3_positive_month_share=('top3_positive_share','first')).to_csv(out/'monthly_concentration_summary.csv',index=False)

def extra_plots(out):
    comp=pd.read_csv(out/'five_year_metrics.csv');annual=pd.read_csv(out/'annual_performance.csv')
    c=comp[(comp.scenario=='strict')&(comp.phase==1)&(comp['mode']=='base')]
    fig,axes=plt.subplots(1,3,figsize=(16,7))
    for ax,col,title in zip(axes,['cumulative_return','sharpe','max_drawdown'],['Net return','Sharpe','Maximum drawdown']):
        m=c.pivot(index='strategy',columns='frequency',values=col).reindex(index=KINDS,columns=FREQ)
        im=ax.imshow(m.to_numpy(float),aspect='auto',cmap='RdYlGn')
        ax.set_xticks(range(4),FREQ,rotation=35);ax.set_yticks(range(len(KINDS)),KINDS);ax.set_title(title)
        for i in range(len(KINDS)):
            for j in range(4):
                v=m.iloc[i,j];ax.text(j,i,'N/A' if pd.isna(v) else f'{v:.2f}',ha='center',va='center',fontsize=7)
        fig.colorbar(im,ax=ax,shrink=.6)
    fig.tight_layout();fig.savefig(out/'all_signals_frequency_heatmaps.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,18),sharex=True)
    for ax,kind in zip(axes.flat,KINDS):
        for j,freq in enumerate(FREQ):
            g=annual[(annual.strategy==kind)&(annual.frequency==freq)&(annual.phase==1)&(annual['mode']=='base')&(annual.scenario=='strict')].set_index('year').reindex(range(2021,2026))
            ax.bar(np.arange(5)+(j-1.5)*.2,100*g.cumulative_return,.2,label=freq)
        ax.set_title(kind);ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Net return (%)');ax.set_xticks(range(5),range(2021,2026))
    axes.flat[0].legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(out/'all_signals_annual_frequency.png',dpi=140);plt.close(fig)

def export_verified_benchmark(out):
    old=ROOT/'result/experiments/monthly_long_only_comparison/run_1794460543886121000'
    prev_annual=pd.read_csv(old/'annual_performance.csv');prev_annual=prev_annual[(prev_annual.strategy=='sp500')&(prev_annual.scenario=='observed')].sort_values('year')
    annual=pd.read_csv(out/'annual_performance.csv');sample=annual[(annual.strategy=='mom_12_1')&(annual.frequency=='monthly')&(annual.phase==1)&(annual['mode']=='base')&(annual.scenario=='strict')].sort_values('year')
    implied=sample.cumulative_return.to_numpy()-sample.excess_return.to_numpy()
    if len(implied)!=5 or not np.allclose(implied,prev_annual.cumulative_return.to_numpy(),rtol=1e-10,atol=1e-10):raise ValueError('Benchmark differs from previously recorded WRDS index; cannot reuse benchmark risk metrics')
    prev_annual.to_csv(out/'benchmark_annual.csv',index=False)
    prev=pd.read_csv(old/'five_year_comparison.csv');prev[(prev.strategy=='sp500')&(prev.scenario=='observed')].to_csv(out/'benchmark_five_year.csv',index=False)

if __name__=='__main__':main()
