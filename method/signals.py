"""Point-in-time signals and formation schedule."""
import numpy as np
import pandas as pd

def formation_dates(calendar, frequency):
    s=pd.Series(calendar,index=calendar)
    return pd.DatetimeIndex(s.groupby(calendar.to_period({'daily':'D','weekly':'W-FRI','monthly':'M'}[frequency])).max().values)

def price_signal(returns, calendar, date):
    # Select completed month-end marks and require every daily return in the window.
    marks=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max()
    completed=marks[marks<=date]
    if len(completed)<13: return pd.Series(dtype=float)
    left,right=completed.iloc[-13],completed.iloc[-2]
    sample=returns.loc[(returns.index>left)&(returns.index<=right)]
    valid=sample.notna().all(axis=0)
    return (1+sample.loc[:,valid]).prod()-1

def news_signal(news, calendar, closes, date, variant='sentiment', window=5):
    i=calendar.get_loc(date); begin=calendar[max(0,i-window+1)]
    if i-window>=0: lower=closes.loc[calendar[i-window]]
    else: lower=pd.Timestamp(begin,tz='UTC')-pd.Timedelta(days=7)
    x=news[(news.timestamp_utc>lower)&(news.timestamp_utc<=closes.loc[date])].copy()
    x=x[x.event_sentiment_score.between(0,100)]
    if variant=='sentiment': return x.groupby('permno').event_sentiment_score.mean()
    col={'relevance':'relevance','novelty':'event_similarity_days'}[variant]
    x=x[x[col].notna() & (x[col]>0)].copy()
    if variant=='relevance': x=x[x[col].between(0,100)]
    else: x[col]=x[col].clip(upper=90)
    if x.empty: return pd.Series(dtype=float)
    x['weighted']=x.event_sentiment_score*x[col]
    return x.groupby('permno').weighted.sum()/x.groupby('permno')[col].sum()

def weights(signal, members, count):
    s=signal[signal.index.isin(members)].dropna().sort_index().sort_values(kind='stable')
    if len(s)<2*count: return pd.Series(dtype=float),len(s)
    return pd.concat([pd.Series(-.5/count,index=s.index[:count]),pd.Series(.5/count,index=s.index[-count:])]),len(s)

def aggregate_news(news, calendar, closes):
    """Accumulate event numerator/denominator by next close and security."""
    x=news[news.event_sentiment_score.between(0,100)].copy()
    locations=closes.searchsorted(x.timestamp_utc,side='left')
    ok=locations<len(calendar); x=x.loc[ok].copy(); locations=locations[ok]
    x['session']=calendar.take(locations).values
    for name,col in [('relevance','relevance'),('novelty','event_similarity_days')]:
        weight=pd.to_numeric(x[col],errors='coerce').where(lambda z:z>0)
        weight=weight.where(weight<=100) if name=='relevance' else weight.clip(upper=90)
        x[name+'_weight']=weight.fillna(0.); x[name+'_weighted']=x.event_sentiment_score*x[name+'_weight']
    x['count']=1
    return x.groupby(['session','permno'],sort=False)[['event_sentiment_score','count','relevance_weight','relevance_weighted','novelty_weight','novelty_weighted']].sum()

def rolling_news(parts, calendar):
    combined=pd.concat(parts).groupby(level=[0,1]).sum()
    mats={k:combined[k].unstack(fill_value=0).reindex(calendar,fill_value=0).rolling(5,min_periods=1).sum() for k in combined.columns}
    return {'sentiment':mats['event_sentiment_score'].div(mats['count'].replace(0,np.nan)), 'relevance':mats['relevance_weighted'].div(mats['relevance_weight'].replace(0,np.nan)), 'novelty':mats['novelty_weighted'].div(mats['novelty_weight'].replace(0,np.nan))}
