"""Independent point-in-time price momentum variants for the formal engine."""
import numpy as np
import pandas as pd
from .signals import formation_dates


def monthly_returns(returns, calendar):
    """Compound only complete calendar months, retaining missing observations."""
    r=returns.reindex(calendar)
    return (1+r).groupby(calendar.to_period('M')).prod(min_count=1).sub(1).where(r.notna().groupby(calendar.to_period('M')).all())


def residual_momentum(returns, calendar, factors):
    """FF3 one-step residuals from 36 prior months, then standardized 11-month sum.

    The signal month t uses residuals t-11..t-1; final factor month is t-1.
    Betas for each residual are estimated using only its preceding 36 months.
    """
    return residual_from_monthly(monthly_returns(returns, calendar), factors)


def residual_from_monthly(monthly, factors):
    """Shared FF3 residual calculation for observed or filtered monthly returns."""
    f=factors.copy(); f.index=pd.DatetimeIndex(f.index).to_period('M'); f=f.reindex(monthly.index)
    values=monthly.to_numpy(dtype=float); fac=f[['mktrf','smb','hml','rf']].to_numpy(dtype=float)
    residual=np.full(values.shape,np.nan)
    for i in range(36,len(monthly)):
        hist=values[i-36:i]; xfac=fac[i-36:i]
        if not np.isfinite(xfac).all() or not np.isfinite(fac[i]).all(): continue
        valid=np.isfinite(hist).all(axis=0)&np.isfinite(values[i])
        if not valid.any(): continue
        x=np.column_stack((np.ones(36),xfac[:,:3])); y=hist[:,valid]-xfac[:,3,None]
        if np.linalg.matrix_rank(x)<4: continue
        beta=np.linalg.lstsq(x,y,rcond=None)[0]
        residual[i,valid]=values[i,valid]-fac[i,3]-np.r_[1.,fac[i,:3]]@beta
    residual=pd.DataFrame(residual,index=monthly.index,columns=monthly.columns)
    lag=residual.shift(1)
    sd=lag.rolling(11,min_periods=11).std(ddof=1)
    return lag.rolling(11,min_periods=11).sum().div(sd.where(sd>0))


def intermediate_signal(returns,calendar,date):
    marks=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max(); completed=marks[marks<=date]
    if len(completed)<13: return pd.Series(dtype=float)
    left,right=completed.iloc[-13],completed.iloc[-7]
    x=returns.loc[(returns.index>left)&(returns.index<=right)]; valid=x.notna().all()
    return (1+x.loc[:,valid]).prod()-1


def continuous_signal(returns,calendar,date,eligible,tradable,count=10):
    """Paper's PRET quintiles followed by low-ID sort in both extreme quintiles."""
    marks=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max(); completed=marks[marks<=date]
    if len(completed)<13: return pd.Series(dtype=float)
    x=returns.loc[(returns.index>completed.iloc[-13])&(returns.index<=completed.iloc[-2])]
    cols=x.columns[x.columns.isin(eligible)&tradable.reindex(x.columns,fill_value=False).to_numpy()]
    x=x[cols]; x=x.loc[:,x.notna().all()]
    n=len(x.columns)//5
    if n<count: return pd.Series(dtype=float)
    pret=(1+x).prod()-1
    discr=np.sign(pret)*((x<0).mean()-(x>0).mean())
    ordered=pret.sort_index().sort_values(kind='stable'); losers=ordered.index[:n]; winners=ordered.index[-n:]
    low=discr.loc[losers].sort_index().sort_values(kind='stable').index[:count]
    high=discr.loc[winners].sort_index().sort_values(kind='stable').index[:count]
    # Ordinals encode selection only; no blending or weighting of PRET and ID.
    return pd.Series(np.r_[np.arange(count),np.arange(count,2*count)],index=low.append(high),dtype=float)


def signal_panels(returns,calendar,members,tradable,factors,start,end,count=10):
    forms=formation_dates(calendar,'weekly'); forms=forms[(forms>=pd.Timestamp(start))&(forms<=pd.Timestamp(end))]
    residual=residual_momentum(returns,calendar,factors); intermediate={}; continuous={}; rp={}
    for d in forms:
        intermediate[d]=intermediate_signal(returns,calendar,d)
        eligible=set(members.loc[(members.mbrstartdt<=d)&(members.mbrenddt>=d),'permno'])
        continuous[d]=continuous_signal(returns,calendar,d,eligible,tradable.loc[d],count)
        # Match baseline's completed-month convention, including month-end formation.
        completed=calendar[calendar<=d]; m=completed[-1].to_period('M')
        if d not in set(formation_dates(calendar,'monthly')): m=m-1
        if m in residual.index: rp[d]=residual.loc[m].dropna()
    return {k:pd.DataFrame.from_dict(v,orient='index').reindex(calendar) for k,v in [('residual',rp),('intermediate',intermediate),('continuous',continuous)]}
