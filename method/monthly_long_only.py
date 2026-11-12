"""Independent monthly long-only momentum signals and target weights."""
import numpy as np
import pandas as pd
from .signals import formation_dates,price_signal
from .price_variants import monthly_returns,intermediate_signal,residual_momentum,residual_from_monthly,continuous_signal
from .kalman_momentum import filter_log_returns,kalman_signal


def long_weights(signal,members,count):
    s=signal[signal.index.isin(members)].dropna().sort_index().sort_values(kind='stable')
    if len(s)<count:return pd.Series(dtype=float),len(s)
    return pd.Series(1./count,index=s.index[-count:]),len(s)


def momentum_12_1(returns,calendar,date):
    marks=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max(); marks=marks[marks<=date]
    if len(marks)<14:return pd.Series(dtype=float)
    x=returns.loc[(returns.index>marks.iloc[-14])&(returns.index<=marks.iloc[-2])]
    return (1+x.loc[:,x.notna().all()]).prod()-1


def filtered_12_1(filtered,monthly,calendar,date):
    marks=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max(); marks=marks[marks<=date]
    if len(marks)<14:return pd.Series(dtype=float)
    left,right=marks.iloc[-14].to_period('M'),marks.iloc[-2].to_period('M')
    x=filtered.loc[(filtered.index>left)&(filtered.index<=right)]; observed=monthly.reindex(x.index)
    valid=x.notna().all()&observed.notna().all()&(observed>-1).all()
    return np.expm1(x.loc[:,valid].sum())


def filtered_continuous(filtered,monthly,calendar,date,members,tradable,count=10):
    marks=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max();marks=marks[marks<=date]
    if len(marks)<13:return pd.Series(dtype=float)
    left,right=marks.iloc[-13].to_period('M'),marks.iloc[-2].to_period('M')
    x=filtered.loc[(filtered.index>left)&(filtered.index<=right)]
    obs=monthly.reindex(x.index); valid=x.notna().all()&obs.notna().all()&(obs>-1).all()
    x=x.loc[:,valid]; x=x.loc[:,x.columns.isin(members)&tradable.reindex(x.columns,fill_value=False).to_numpy()]
    n=len(x.columns)//5
    if n<count:return pd.Series(dtype=float)
    pret=np.expm1(x.sum()); discr=np.sign(pret)*((x<0).mean()-(x>0).mean())
    winners=pret.sort_index().sort_values(kind='stable').index[-n:]
    selected=discr.loc[winners].sort_index().sort_values(kind='stable').index[:count]
    return pd.Series(np.arange(count,dtype=float),index=selected)


def build_signals(returns,calendar,members,tradable,factors,start,end,formation_schedule=None):
    monthly=monthly_returns(returns,calendar); filtered=filter_log_returns(monthly)
    residual=residual_momentum(returns,calendar,factors)
    kresidual=residual_from_monthly(np.expm1(filtered),factors)
    forms=formation_dates(calendar,'monthly') if formation_schedule is None else pd.DatetimeIndex(formation_schedule)
    forms=forms[(forms>=pd.Timestamp(start))&(forms<=pd.Timestamp(end))]
    month_ends=set(formation_dates(calendar,'monthly'))
    names=('mom_12_1','mom_12_2','intermediate','residual','continuous','kalman_12_1','kalman_12_2','kalman_12_7','kalman_residual','kalman_continuous')
    series={k:{} for k in names}
    for d in forms:
        elig=set(members.loc[(members.mbrstartdt<=d)&(members.mbrenddt>=d),'permno'])
        series['mom_12_1'][d]=momentum_12_1(returns,calendar,d);series['mom_12_2'][d]=price_signal(returns,calendar,d)
        series['intermediate'][d]=intermediate_signal(returns,calendar,d)
        series['continuous'][d]=continuous_signal(returns,calendar,d,elig,tradable.loc[d])
        series['kalman_12_1'][d]=filtered_12_1(filtered,monthly,calendar,d)
        series['kalman_12_2'][d]=kalman_signal(filtered,monthly,calendar,d,'12_2')
        series['kalman_12_7'][d]=kalman_signal(filtered,monthly,calendar,d,'12_7')
        series['kalman_continuous'][d]=filtered_continuous(filtered,monthly,calendar,d,elig,tradable.loc[d])
        period=d.to_period('M') if d in month_ends else d.to_period('M')-1
        if period in residual.index:series['residual'][d]=residual.loc[period].dropna();series['kalman_residual'][d]=kresidual.loc[period].dropna()
    return {k:pd.DataFrame.from_dict(v,orient='index').reindex(calendar) for k,v in series.items()}
