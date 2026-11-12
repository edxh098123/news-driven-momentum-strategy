"""Metrics from daily net returns; all annualization assumptions are explicit."""
import numpy as np
import pandas as pd

def metrics(frame, benchmark, rf):
    x=frame.join(benchmark.rename('benchmark'),how='inner').join(rf.rename('rf'),how='inner').dropna(subset=['net','benchmark','rf'])
    if len(x)<2: raise ValueError('Insufficient overlapping benchmark and risk-free data')
    r=x.net; excess=r-x.rf; market=x.benchmark-x.rf
    n=len(r); total=(1+r).prod(); vol=r.std(ddof=1)*np.sqrt(252)
    down=np.sqrt(np.mean(np.minimum(excess,0)**2))*np.sqrt(252)
    beta=excess.cov(market)/market.var(ddof=1) if market.var(ddof=1)>0 else np.nan
    wealth=(1+r).cumprod(); peak=wealth.cummax().clip(lower=1.)
    return dict(days=n,cumulative_return=total-1,cumulative_gross_return=(1+x.gross).prod()-1,annualized_gross_return=(1+x.gross).prod()**(252/n)-1,annualized_return=total**(252/n)-1,volatility=vol,sharpe=excess.mean()/excess.std(ddof=1)*np.sqrt(252) if excess.std(ddof=1)>0 else np.nan,sortino=excess.mean()*252/down if down else np.nan,max_drawdown=(wealth/peak-1).min(),alpha=(excess.mean()-beta*market.mean())*252,beta=beta,turnover=x.turnover.sum(),trade_cost=x.trade_cost.sum(),borrow_cost=x.borrow_cost.sum(),average_gross=x.gross_exposure.mean(),average_net=x.net_exposure.mean(),max_gross=x.gross_exposure.max(),min_net=x.net_exposure.min(),max_net=x.net_exposure.max())
