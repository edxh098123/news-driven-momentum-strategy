"""Fixed, causal volatility targeting of a selected long-only basket."""
import numpy as np
import pandas as pd

def target_15pct(stock_returns,lookback=63,min_obs=42):
    """Annualized daily volatility, capped at 100% exposure; cash earns zero."""
    def transform(date,target,rows):
        if target.empty:return target
        x=stock_returns.loc[:date,target.index].tail(lookback)
        basket=x.mean(axis=1).where(x.notna().all(axis=1)).dropna()
        if len(basket)<min_obs:return target*0.
        vol=basket.std(ddof=1)*np.sqrt(252)
        return target*(min(1.,.15/vol) if np.isfinite(vol) and vol>0 else 0.)
    return transform
