"""Causal one-dimensional Kalman filtering of monthly total log returns."""
import numpy as np
import pandas as pd
from .price_variants import monthly_returns, intermediate_signal
from .signals import formation_dates,price_signal


def filter_log_returns(monthly, phi=.95, q_over_r=.05, init_months=24):
    """Per stock, initialize on first 24 consecutive observed months, then recurse.

    Initialization mean/variance use only prior observations and remain fixed.
    Missing observations use the prediction step and never emit an observation.
    A monthly total return <= -100% has no finite log and is treated as missing.
    """
    if not (0 <= phi <= 1) or q_over_r <= 0 or init_months < 2:
        raise ValueError('Invalid fixed Kalman parameters')
    obs=np.log1p(monthly.where(monthly > -1)).to_numpy(float)
    result=np.full(obs.shape,np.nan)
    for j in range(obs.shape[1]):
        x=obs[:,j]; history=[]; mu=var=r=None
        for t,value in enumerate(x):
            if r is None:
                if np.isfinite(value): history.append(value)
                else: history=[]
                if len(history)==init_months:
                    mu=float(np.mean(history)); r=max(float(np.var(history,ddof=1)),1e-12); var=r
                continue
            pred=phi*mu; predvar=phi*phi*var+q_over_r*r
            if np.isfinite(value):
                gain=predvar/(predvar+r); mu=pred+gain*(value-pred); var=(1-gain)*predvar
                result[t,j]=mu
            else:
                mu,var=pred,predvar
    return pd.DataFrame(result,index=monthly.index,columns=monthly.columns)


def kalman_signal(filtered, monthly, calendar, date, window):
    """Sum filtered monthly log returns over exactly the existing month-end marks."""
    if window not in ('12_2','12_7'): raise ValueError('Unknown momentum window')
    marks=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max()
    completed=marks[marks<=date]
    if len(completed)<13: return pd.Series(dtype=float)
    left=completed.iloc[-13].to_period('M'); right=completed.iloc[-2 if window=='12_2' else -7].to_period('M')
    vals=filtered.loc[(filtered.index>left)&(filtered.index<=right)]
    actual=monthly.reindex(vals.index)
    valid=vals.notna().all() & actual.notna().all() & (actual > -1).all()
    return np.expm1(vals.loc[:,valid].sum())


def build_panels(returns, calendar, members, tradable, start, end):
    monthly=monthly_returns(returns,calendar)
    filtered=filter_log_returns(monthly)
    dates=formation_dates(calendar,'weekly'); dates=dates[(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))]
    signals={'price':{},'intermediate':{},'kalman_12_2':{},'kalman_12_7':{}}
    for date in dates:
        signals['price'][date]=price_signal(returns,calendar,date)
        signals['intermediate'][date]=intermediate_signal(returns,calendar,date)
        for w in ('12_2','12_7'): signals['kalman_'+w][date]=kalman_signal(filtered,monthly,calendar,date,w)
    return {k:pd.DataFrame.from_dict(v,orient='index').reindex(calendar) for k,v in signals.items()}


def stability_table(panels,calendar,members,tradable,start,end):
    """Compare successive rankings on the identical common point-in-time pool."""
    dates=formation_dates(calendar,'weekly'); dates=dates[(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))]
    records=[]
    for raw,filt in [('price','kalman_12_2'),('intermediate','kalman_12_7')]:
        previous=None
        for d in dates:
            eligible=members.loc[(members.mbrstartdt<=d)&(members.mbrenddt>=d),'permno']
            x=pd.concat([panels[raw].loc[d].rename('raw'),panels[filt].loc[d].rename('filtered')],axis=1).dropna()
            x=x[x.index.isin(eligible) & tradable.loc[d].reindex(x.index,fill_value=False)]
            if previous is not None:
                old,prev=previous; common=x.index.intersection(old.index)
                if len(common)>=20:
                    record={'window':raw,'date':d,'previous_date':prev,'common_count':len(common)}
                    for col in ('raw','filtered'):
                        a=old.loc[common,col]; b=x.loc[common,col]
                        ar=a.rank(pct=True,method='average'); br=b.rank(pct=True,method='average')
                        record[col+'_rank_spearman']=ar.corr(br,method='spearman')
                        record[col+'_mean_abs_rank_change']=(ar-br).abs().mean()
                        record[col+'_mean_abs_log_score_change']=(np.log1p(a)-np.log1p(b)).abs().mean()
                        for side,ascending in [('short',True),('long',False)]:
                            aa=set(a.sort_index().sort_values(ascending=ascending,kind='stable').index[:10]); bb=set(b.sort_index().sort_values(ascending=ascending,kind='stable').index[:10])
                            record[col+'_'+side+'_retention']=len(aa & bb)/10
                    records.append(record)
            previous=(x,d)
    return pd.DataFrame(records)
