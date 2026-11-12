"""2021–2025 strategy comparison using the existing in-memory runner."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from method.report import metrics
from local_runner import execute

START='2021-01-01'
END='2025-12-31'
YEARS=range(2021,2026)
SCENARIOS=('loss_30','loss_55','loss_100')

def benchmark_frame(benchmark):
    return pd.DataFrame({'net':benchmark,'gross':benchmark,'turnover':0.,'trade_cost':0.,'borrow_cost':0.,'gross_exposure':1.,'net_exposure':1.})

def annual_tables(paths,bench,risk,required):
    all_paths={('sp500_total_return','observed'):benchmark_frame(bench),**paths}
    # Include all requested combinations even if a path could not produce rows.
    wanted=[('sp500_total_return','observed')]+[(k,s) for k in ('price','sentiment') for s in ('strict',*SCENARIOS)]
    rows=[];legs=[]
    for kind,scenario in wanted:
        frame=all_paths.get((kind,scenario))
        for year in YEARS:
            expected=required[required.year==year]
            g=frame.loc[frame.index.year==year] if frame is not None else pd.DataFrame()
            complete=len(expected)>0 and len(g)==len(expected) and g.index.equals(expected)
            row={'year':year,'strategy':kind,'scenario':scenario,'status':'complete' if complete else ('incomplete' if len(g) else 'unavailable'),'days':len(g),'sample_start':str(g.index.min().date()) if len(g) else None,'sample_end':str(g.index.max().date()) if len(g) else None}
            if len(g)>1:
                stat=metrics(g,bench,risk)
                row.update({'partial_net_return':stat['cumulative_return'],'annual_net_return':stat['cumulative_return'] if complete else np.nan,'annual_gross_return':stat['cumulative_gross_return'] if complete else np.nan,'volatility':stat['volatility'] if complete else np.nan,'sharpe':stat['sharpe'] if complete else np.nan,'max_drawdown':stat['max_drawdown'] if complete else np.nan,'transaction_cost':stat['trade_cost'],'borrow_cost':stat['borrow_cost']})
                if kind!='sp500_total_return':
                    prior=(1+g.net).cumprod().shift(1,fill_value=1.)
                    vals={'long_contribution':(prior*g['long']).sum(),'short_contribution':(prior*g['short']).sum(),'transaction_cost':(prior*g.trade_cost).sum(),'borrow_cost':(prior*g.borrow_cost).sum(),'net_return':(1+g.net).prod()-1}
                    legs.append({'year':year,'strategy':kind,'scenario':scenario,'status':row['status'],**vals})
            rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(legs)

def report(out,paths,bench,risk,required,start,end):
    out=Path(out); annual,legs=annual_tables(paths,bench,risk,required)
    annual.to_csv(out/'annual_performance.csv',index=False)
    legs.to_csv(out/'annual_leg_attribution.csv',index=False)
    full=[]
    for kind,scenario in [('sp500_total_return','observed')]+[(k,s) for k in ('price','sentiment') for s in ('strict',*SCENARIOS)]:
        frame=benchmark_frame(bench) if kind=='sp500_total_return' else paths.get((kind,scenario))
        complete=frame is not None and frame.index.equals(required)
        row={'strategy':kind,'scenario':scenario,'status':'complete' if complete and scenario in ('strict','observed') else ('scenario_complete' if complete else 'incomplete'),'sample_end':str(frame.index.max().date()) if frame is not None and len(frame) else None}
        if frame is not None and len(frame)>1:
            stat=metrics(frame,bench,risk)
            row.update({'five_year_cumulative_return':stat['cumulative_return'] if complete else np.nan,'partial_cumulative_return':stat['cumulative_return'],'annualized_return':stat['annualized_return'] if complete else np.nan,'volatility':stat['volatility'] if complete else np.nan,'sharpe':stat['sharpe'] if complete else np.nan,'max_drawdown':stat['max_drawdown'] if complete else np.nan})
        full.append(row)
    pd.DataFrame(full).to_csv(out/'five_year_comparison.csv',index=False)
    labels={'sp500_total_return':'S&P 500 TR','price':'Price','sentiment':'News'}
    idx=np.arange(len(YEARS));fig,ax=plt.subplots(figsize=(10,5))
    for offset,(kind,scenario) in zip((-.25,0,.25),(('sp500_total_return','observed'),('price','strict'),('sentiment','strict'))):
        v=annual[(annual.strategy==kind)&(annual.scenario==scenario)].set_index('year').reindex(YEARS).annual_net_return
        bars=ax.bar(idx+offset,v*100,width=.25,label=labels[kind])
        for x,value in zip(idx+offset,v):
            if pd.isna(value): ax.annotate('N/A',(x,0),rotation=90,ha='center',va='bottom',fontsize=8)
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(idx,list(YEARS));ax.set_ylabel('Net return (%)');ax.set_title('Annual returns: strict observed/valuation paths');ax.legend();fig.tight_layout();fig.savefig(out/'annual_returns_grouped.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    selections=[('sp500_total_return','observed','S&P 500 TR'),('sentiment','strict','News'),('price','loss_30','Price loss 30%'),('price','loss_55','Price loss 55%'),('price','loss_100','Price loss 100%')]
    for i,(kind,scenario,label) in enumerate(selections):
        values=annual[(annual.strategy==kind)&(annual.scenario==scenario)].set_index('year').reindex(YEARS).annual_net_return
        ax.bar(idx+(i-2)*.16,values*100,width=.16,label=label)
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(idx,list(YEARS));ax.set_ylabel('Net return (%)');ax.set_title('Annual sensitivity: assumed terminal losses');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'annual_returns_scenarios.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));
    for (kind,scenario),style in [ (('sp500_total_return','observed'),'-'),(('price','strict'),'-'),(('sentiment','strict'),'-') ]:
        frame=benchmark_frame(bench) if kind=='sp500_total_return' else paths.get((kind,scenario))
        if frame is None or frame.empty: continue
        series=100*(1+frame.net).cumprod();series=pd.concat([pd.Series([100.],index=[pd.Timestamp(start)-pd.Timedelta(days=1)]),series])
        suffix=f' (stopped {frame.index.max().date()})' if not frame.index.equals(required) else ''
        series.plot(ax=ax,label=labels[kind]+suffix,linestyle=style)
    ax.set_ylabel('Net asset value (initial 100)');ax.set_title('Strict paths; incomplete series stop');ax.legend();fig.tight_layout();fig.savefig(out/'cumulative_net_value_100.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for (kind,scenario),frame in paths.items():
        if kind!='price' or scenario not in SCENARIOS:continue
        series=100*(1+frame.net).cumprod();series=pd.concat([pd.Series([100.],index=[pd.Timestamp(start)-pd.Timedelta(days=1)]),series])
        suffix=' (incomplete)' if not frame.index.equals(required) else ''
        series.plot(ax=ax,label=f'Price {scenario}{suffix}')
    ax.set_ylabel('Net asset value (initial 100)');ax.set_title('Assumed terminal-loss scenarios');ax.legend();fig.tight_layout();fig.savefig(out/'price_sensitivity_net_value_100.png',dpi=150);plt.close(fig)
    for kind in ('price','sentiment'):
        fig,ax=plt.subplots(figsize=(10,5)); g=legs[(legs.strategy==kind)&(legs.scenario=='strict')].set_index('year').reindex(YEARS)
        if 'status' in g: g.loc[g.status!='complete',['long_contribution','short_contribution']]=np.nan
        for off,col,title in [(-.2,'long_contribution','Long contribution'),(.2,'short_contribution','Short contribution')]:
            values=g[col] if col in g else pd.Series(np.nan,index=YEARS)
            ax.bar(idx+off,values*100,width=.4,label=title)
            for x,value in zip(idx+off,values):
                if pd.isna(value):ax.annotate('N/A',(x,0),rotation=90,ha='center',va='bottom',fontsize=8)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(idx,list(YEARS));ax.set_ylabel('Contribution to annual starting NAV (%)');ax.set_title(f'{labels[kind]} legs: gross contributions, strict path');ax.legend();fig.tight_layout();fig.savefig(out/f'{kind}_annual_legs.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,scenario in zip(axes,SCENARIOS):
        g=legs[(legs.strategy=='price')&(legs.scenario==scenario)].set_index('year').reindex(YEARS)
        if 'status' in g:g.loc[g.status!='complete',['long_contribution','short_contribution']]=np.nan
        for off,col,title in [(-.2,'long_contribution','Long'),(.2,'short_contribution','Short')]:
            values=g[col] if col in g else pd.Series(np.nan,index=YEARS);ax.bar(idx+off,values*100,width=.4,label=title)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(idx,list(YEARS),rotation=45);ax.set_title(scenario)
    axes[0].set_ylabel('Contribution to annual starting NAV (%)');axes[0].legend();fig.suptitle('Price legs: assumed terminal-loss scenarios');fig.tight_layout();fig.savefig(out/'price_annual_legs_scenarios.png',dpi=150);plt.close(fig)

if __name__=='__main__': execute(START,END,reporter=report)
