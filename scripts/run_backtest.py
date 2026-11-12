"""Direct WRDS research runner. Never writes raw input observations to disk."""
import argparse, json, logging, os, pathlib, sys
import numpy as np
import pandas as pd
import exchange_calendars as xc
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.collect import DataAccess,link_entities,match_news,build_crsp_flags
from method.engine import run, PathIncomplete
from method.missing_returns import MissingReturnHandler
from method.report import metrics
from method.signals import aggregate_news, rolling_news

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--start'); parser.add_argument('--end'); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    cfg=json.loads((ROOT/'config/settings.json').read_text()); start=args.start or cfg['start_date']; end=args.end or cfg['end_date']
    if pd.Timestamp(start)>pd.Timestamp(end): raise ValueError('start must not exceed end')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s'); log=logging.getLogger('backtest')
    warm=(pd.Timestamp(start)-pd.DateOffset(months=16)).strftime('%Y-%m-%d')
    nyse=xc.get_calendar('XNYS',start=warm,end=end); schedule=nyse.schedule
    calendar=pd.DatetimeIndex(schedule.index).tz_localize(None); closes=pd.Series(pd.to_datetime(schedule.close,utc=True).values,index=calendar).dt.tz_localize('UTC')
    os.environ.setdefault('PGOPTIONS','-c statement_timeout=180000')
    db=DataAccess(cfg['wrds_user'])
    try:
        log.info('Query membership, prices, mappings, benchmark and risk-free rates')
        membership=db.membership(warm,end); names=db.names(warm,end); mapping=db.mappings(warm,end,names.cusip.dropna().str[:8].unique().tolist())
        links=link_entities(names,mapping,membership,warm,end); del names,mapping
        prices=db.prices(warm,end,warm); delistings=db.delistings(warm,end,membership.permno.dropna().astype(int).unique().tolist()); benchmark=db.benchmark(warm,end); rf=db.riskfree(start,end)
        returns=prices.pivot(index='dlycaldt',columns='permno',values='dlyret').reindex(calendar).astype(float)
        price_returns=prices.pivot(index='dlycaldt',columns='permno',values='dlyretx').reindex(calendar).astype(float)
        delisted=prices.assign(is_delisted=prices.dlydelflg.eq('Y')).pivot(index='dlycaldt',columns='permno',values='is_delisted').reindex(calendar).fillna(False).astype(bool)
        prices['can_trade']=prices.dlyprc.notna() & (prices.dlyprc.astype(float)>0) & prices.tradingstatusflg.eq('A')
        tradable=prices.pivot(index='dlycaldt',columns='permno',values='can_trade').reindex(calendar).fillna(False).astype(bool); crsp_flags=build_crsp_flags(prices,delistings); del prices,delistings
        news=[]
        news_start=(pd.Timestamp(start)-pd.Timedelta(days=14)).strftime('%Y-%m-%d')
        for period in pd.period_range(news_start,end,freq='Q'):
            a=max(pd.Timestamp(news_start),period.start_time).strftime('%Y-%m-%d')
            b=min(pd.Timestamp(end),period.end_time).strftime('%Y-%m-%d')
            log.info('Query RavenPack %s to %s; in-memory only',a,b)
            raw=db.news(period.year,links.rp_entity_id.dropna().unique(),a,b)
            mapped=match_news(raw,links); del raw
            log.info('Matched %s news records',len(mapped)); news.append(aggregate_news(mapped,calendar,closes)); del mapped
        news=rolling_news(news,calendar) if news and any(not p.empty for p in news) else None
    finally: db.close()
    if news is None: raise ValueError('No mapped news records; refusing to report a news strategy')
    covered=news['sentiment'].notna().any(axis=1)
    available=covered.loc[start:end]
    if not available.any(): raise ValueError('No scored news in requested period')
    start=max(pd.Timestamp(start),available[available].index.min()).strftime('%Y-%m-%d')
    end=min(pd.Timestamp(end),available[available].index.max()).strftime('%Y-%m-%d')
    log.info('Common scored-news and requested coverage: %s to %s',start,end)
    # Compustat S&P 500 index 000003: prccddiv is total-return index level.
    levels=benchmark.assign(datadate=pd.to_datetime(benchmark.datadate)).set_index('datadate').prccddiv.astype(float).sort_index()
    if levels.isna().any() or levels.le(0).any() or levels.index.has_duplicates: raise ValueError('Invalid S&P 500 total-return index levels')
    bench=levels.pct_change(fill_method=None).loc[start:end]
    risk=rf.assign(date=pd.to_datetime(rf.date)).set_index('date').rf.astype(float)
    required=calendar[(calendar>=pd.Timestamp(start)) & (calendar<=pd.Timestamp(end))]
    if bench.reindex(required).isna().any(): raise ValueError('Incomplete S&P 500 total-return trading-day coverage')
    if risk.reindex(required).isna().any(): raise ValueError('Incomplete daily risk-free-rate coverage')
    variants=[('price','price'),('news','sentiment'),('news_relevance','relevance'),('news_novelty_similarity_gap','novelty'),('price_news_coverage','covered_price')]
    specs=[(name,kind,10,'weekly') for name,kind in variants]
    if not args.smoke:
        specs += [(f'{kind}_{n}_{frequency}',kind,n,frequency) for kind in ('price','sentiment') for n in (10,20,50) for frequency in ('daily','weekly','monthly') if (n,frequency)!=(10,'weekly')]
    results={}; failures=[]; metric_rows=[]; diagnostics=[]; price_cache={}
    for name,kind,n,freq in specs:
        try:
            log.info('Backtest %s',name); frame,diag=run(returns,tradable,membership,news,calendar,closes,start,end,kind,n,freq,price_returns=price_returns,delisted=delisted,price_cache=price_cache,missing_handler=MissingReturnHandler(max_stale_sessions=cfg.get('missing_returns',{}).get('max_stale_sessions',10)),crsp_flags=crsp_flags)
            results[name]=frame; metric_rows.append(dict(strategy=name,status='complete',sample_start=str(frame.index.min().date()),sample_end=str(frame.index.max().date()),**metrics(frame,bench,risk)))
            diagnostics.append(dict(strategy=name,formation_count=int(diag.formation.nunique()),insufficient_count=int(diag.reason.fillna('').str.startswith('insufficient').sum()),untradeable_count=int(diag.reason.fillna('').str.startswith('untradeable').sum()),terminal_settlement_count=int(diag.reason.fillna('').str.startswith('CRSP terminal').sum())))
        except PathIncomplete as exc:
            log.error('Backtest %s stopped: %s',name,exc); failures.append(dict(strategy=name,error=str(exc)))
            frame=exc.frame; diag=exc.diagnostics
            if not diag.empty:
                diagnostics.append(dict(strategy=name,formation_count=int(diag.formation.nunique()),insufficient_count=int(diag.reason.fillna('').str.startswith('insufficient').sum()),untradeable_count=int(diag.reason.fillna('').str.startswith('untradeable').sum()),terminal_settlement_count=int(diag.reason.fillna('').str.startswith('CRSP terminal').sum())))
            if len(frame)>1:
                results[name]=frame
                metric_rows.append(dict(strategy=name,status='incomplete',sample_start=str(frame.index.min().date()),sample_end=str(frame.index.max().date()),**metrics(frame,bench,risk)))
        except Exception as exc:
            log.exception('Backtest %s failed',name); failures.append(dict(strategy=name,error=str(exc)))
    out=ROOT/'result/tables'; fig=ROOT/'result/figures'; out.mkdir(parents=True,exist_ok=True); fig.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(failures,columns=['strategy','error']).to_csv(out/'failed_runs.csv',index=False)
    if metric_rows:
        bf=pd.DataFrame({'net':bench,'gross':bench,'turnover':0.,'trade_cost':0.,'borrow_cost':0.,'gross_exposure':1.,'net_exposure':1.})
        metric_rows.append(dict(strategy='sp500_total_return',status='complete',sample_start=str(bf.index.min().date()),sample_end=str(bf.index.max().date()),**metrics(bf,bench,risk)))
        pd.DataFrame(metric_rows).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/'diagnostics_summary.csv',index=False)
        annual=[]
        for name,frame in results.items():
            for year,g in frame.groupby(frame.index.year): annual.append(dict(strategy=name,year=year,sample_start=str(g.index.min().date()),sample_end=str(g.index.max().date()),complete_year=len(g)==int((bench.index.year==year).sum()),gross_return=(1+g.gross).prod()-1,net_return=(1+g.net).prod()-1,trade_cost=g.trade_cost.sum(),borrow_cost=g.borrow_cost.sum(),turnover=g.turnover.sum()))
        pd.DataFrame(annual).to_csv(out/'annual_gross_net_costs.csv',index=False)
    if {'price','news'}.issubset(results):
        import matplotlib.pyplot as plt
        fig1,ax=plt.subplots(); ((1+bench).cumprod()).plot(ax=ax,label='S&P 500 Total Return (Compustat)')
        for name in ('price','news'):
            frame=results[name]; suffix=f" (stopped {frame.index.max().date()})" if frame.index.max()<pd.Timestamp(end) else ''
            (1+frame.net).cumprod().plot(ax=ax,label=name+suffix)
        ax.legend(); ax.set_title('Cumulative returns'); fig1.tight_layout(); fig1.savefig(fig/'cumulative_returns.png'); plt.close(fig1)
        years=pd.DataFrame({k:(1+v.net).groupby(v.index.year).prod()-1 for k,v in results.items()}); years['sp500_total_return']=(1+bench).groupby(bench.index.year).prod()-1
        for name,frame in results.items():
            for year,g in frame.groupby(frame.index.year):
                if len(g)!=int((bench.index.year==year).sum()): years.loc[year,name]=np.nan
        years.to_csv(out/'annual_returns.csv')
        legs=[]
        for year,g in results['news'].groupby(results['news'].index.year):
            prior=(1+g.net).cumprod().shift(1,fill_value=1.)
            legs.append(dict(year=year,sample_start=str(g.index.min().date()),sample_end=str(g.index.max().date()),long_contribution=(prior*g['long']).sum(),short_contribution=(prior*g['short']).sum(),transaction_cost=(prior*g.trade_cost).sum(),borrow_cost=(prior*g.borrow_cost).sum(),net_return=(1+g.net).prod()-1))
        pd.DataFrame(legs).to_csv(out/'news_legs.csv',index=False)
    log.warning('Benchmark: Compustat S&P 500 total-return index gvkeyx 000003. Borrow availability is assumed. Similarity-gap weights are a novelty proxy. CRSP terminal returns are settled to cash on delisting dates.')
    if failures: raise RuntimeError(f'{len(failures)} backtest paths failed; see failed_runs.csv')

if __name__=='__main__': main()
