"""WRDS-only data access. All downloaded observations remain in memory."""
import pandas as pd
import time
import logging
import wrds

class DataAccess:
    def __init__(self, username='edxh'):
        self.db = wrds.Connection(wrds_username=username)
        self.query_stats=[]
    def close(self):
        self.db.close()
    def query(self, sql, params):
        started=time.monotonic()
        try: return self.db.raw_sql(sql, params=params)
        finally:
            seconds=time.monotonic()-started
            self.query_stats.append({'seconds':seconds,'query':sql[:100]})
            logging.info('WRDS query %d completed in %.2fs%s',len(self.query_stats),seconds,' (slow)' if seconds>=10 else '')
    def membership(self, start, end):
        x = self.query('select permno,mbrstartdt,mbrenddt from crsp_a_indexes.dsp500list_v2 where mbrstartdt <= %(end)s and mbrenddt >= %(start)s', locals())
        for c in ['mbrstartdt','mbrenddt']: x[c]=pd.to_datetime(x[c])
        return x.drop_duplicates()
    def names(self, start, end):
        x=self.query('select permno,cusip,namedt,nameenddt from crsp.stocknames_v2 where namedt <= %(end)s and nameenddt >= %(start)s and permno in (select permno from crsp_a_indexes.dsp500list_v2 where mbrstartdt<=%(end)s and mbrenddt >= %(start)s)', locals())
        for c in ['namedt','nameenddt']: x[c]=pd.to_datetime(x[c])
        return x.drop_duplicates()
    def mappings(self, start, end, cusips=None):
        
        if cusips is None:
            cusips=self.names(start,end).cusip.dropna().str[:8].unique().tolist()
        x=self.query("select rp_entity_id,cusip,range_start,range_end from ravenpack_common.wrds_rpa_company_mappings_hist where range_start<=%(end)s and (range_end is null or range_end >= %(start)s) and left(cusip,8)=any(%(cusips)s)",locals())
        for c in ['range_start','range_end']: x[c]=pd.to_datetime(x[c],errors='coerce')
        return x.drop_duplicates()
    def prices(self, start, end, universe_start, audit_actions=False):
        fields=',d.dlyorddivamt,d.dlynonorddivamt,d.dlyfacprc,d.dlydistretflg' if audit_actions else ''
        x=self.query('select d.permno,d.dlycaldt,d.dlyprc,d.dlyret,d.dlyretx,d.dlydelflg,d.tradingstatusflg,d.dlyretmissflg,d.dlyretdurflg,d.dlyprcflg,d.dlyprevdt'+fields+' from crsp.dsf_v2 d where d.dlycaldt between %(start)s and %(end)s and exists (select 1 from crsp_a_indexes.dsp500list_v2 m where m.permno=d.permno and m.mbrstartdt <= %(end)s and m.mbrenddt >= %(universe_start)s)',locals())
        x['dlycaldt']=pd.to_datetime(x.dlycaldt)
        if x.duplicated(['permno','dlycaldt']).any(): raise ValueError('Duplicate CRSP PERMNO/date')
        return x
    def delistings(self,start,end,permnos):
        return self.query('select permno,delistingdt,deldlydt,deldtprc,deldtprcflg,delactiontype,delstatustype,delreasontype,delpaymenttype,delret,delretmisstype,delnextdt,delnextprc,delamtdt,deldivamt from crsp.stkdelists where permno=any(%(permnos)s) and delistingdt between %(start)s and %(end)s',locals())
    def benchmark(self,start,end):
        return self.query("select datadate,prccddiv from comp.idx_daily where gvkeyx='000003' and datadate between %(start)s and %(end)s order by datadate",locals())
    def riskfree(self,start,end):
        return self.query('select date,rf from ff.factors_daily where date between %(start)s and %(end)s order by date',locals())
    def news(self, year, entities, start, end):
        # Identifier is derived exclusively from an integer, not user SQL.
        table=f'ravenpack_dj.rpa_djpr_equities_{int(year)}'
        query=f'''select timestamp_utc,rp_story_id,rp_story_event_index,rp_entity_id,event_sentiment_score,relevance,nip,event_similarity_days,category from {table} where rpa_date_utc between %(start)s and %(end)s and rp_entity_id = any(%(entities)s)'''
        return self.query(query,dict(start=start,end=end,entities=list(entities)))

def link_entities(names, mappings, membership, start, end):
    names=names.copy(); mappings=mappings.copy()
    names['key']=names.cusip.fillna('').str[:8]; mappings['key']=mappings.cusip.fillna('').str[:8]
    names=names[names.key.str.len()==8]; mappings=mappings[mappings.key.str.len()==8]
    x=mappings.merge(names,on='key').merge(membership,on='permno')
    x['begin']=x[['range_start','namedt','mbrstartdt']].max(axis=1).clip(lower=pd.Timestamp(start))
    x['end']=x[['range_end','nameenddt','mbrenddt']].min(axis=1).clip(upper=pd.Timestamp(end))
    return x.loc[x.begin<=x.end,['rp_entity_id','permno','begin','end']].drop_duplicates()

def match_news(news, links):
    news=news.copy(); news['timestamp_utc']=pd.to_datetime(news.timestamp_utc,utc=True,errors='coerce')
    news=news.dropna(subset=['timestamp_utc','rp_entity_id','rp_story_id','event_sentiment_score'])
    news=news.drop_duplicates(subset=['timestamp_utc','rp_story_id','rp_story_event_index','rp_entity_id','event_sentiment_score','relevance','nip'])
    news['_id']=range(len(news)); match=news[['_id','rp_entity_id','timestamp_utc']].merge(links,on='rp_entity_id')
    day=match.timestamp_utc.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    match=match[(day>=match.begin)&(day<=match.end)][['_id','permno']].drop_duplicates()
    counts=match.groupby('_id').permno.nunique(); match=match[match._id.isin(counts[counts==1].index)]
    return news.merge(match,on='_id',how='inner').drop(columns='_id')

def build_crsp_flags(prices, delistings):
    """Build in-memory abnormal-record lookups; never use postdated settlements."""
    abnormal=prices[(prices.dlyretmissflg.notna() & prices.dlyretmissflg.ne('NA')) | prices.dlyretdurflg.fillna('').str.startswith('P') | prices.dlydelflg.eq('Y')]
    flags={(row.dlycaldt,int(row.permno)):dict(dlyretmissflg=row.dlyretmissflg,dlyretdurflg=row.dlyretdurflg,dlydelflg=row.dlydelflg,dlyprevdt=row.dlyprevdt) for row in abnormal.itertuples()}
    action_cols=['dlyorddivamt','dlynonorddivamt','dlyfacprc','dlydistretflg']
    if all(c in abnormal.columns for c in action_cols):
        for row in abnormal.itertuples():
            flags[(row.dlycaldt,int(row.permno))]['action_audit']={c:getattr(row,c) for c in action_cols}
    for row in delistings.itertuples():
        day=pd.to_datetime(row.deldlydt,errors='coerce')
        dates=[pd.Timestamp(x) for x in (row.delnextdt,row.delamtdt) if pd.notna(x)]
        if pd.notna(day) and pd.notna(row.delret) and row.delretmisstype=='NA' and dates:
            info=flags.get((day,int(row.permno)))
            if info is not None: info.update(settlement_return=float(row.delret),settlement_known_at=max(dates))
    return flags
