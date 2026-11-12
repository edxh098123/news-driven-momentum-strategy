"""Schedules and causal exposure sizing for frequency experiments."""
import numpy as np
import pandas as pd
from .signals import formation_dates


def schedules(calendar, phase=1):
    if phase not in (1,4,7,10):raise ValueError('annual phase must be 1,4,7,10')
    month=formation_dates(calendar,'monthly')
    return {'weekly':formation_dates(calendar,'weekly'),'monthly':month,'quarterly':month[month.month.isin([3,6,9,12])], 'annual':month[month.month==((phase-2)%12+1)]}


def matched_target(stock_returns, benchmark, lookback=63, min_obs=42):
    """Past equal-weight candidate-basket volatility vs benchmark; cap at 1x.

    Re-estimated only at formation; cash earns zero and no debt is permitted.
    """
    def transform(date,target,rows):
        if target.empty:return target
        stock=stock_returns.loc[:date,target.index].tail(lookback)
        basket=stock.mean(axis=1).where(stock.notna().all(axis=1))
        aligned=pd.concat([basket.rename('p'),benchmark.loc[:date].rename('b')],axis=1).dropna().tail(lookback)
        if len(aligned)<min_obs:return target*0.
        sigma=aligned.p.std(ddof=1);vol=aligned.b.std(ddof=1)
        scale=min(1.,vol/sigma) if sigma>0 else 0.
        return target*scale
    return transform
