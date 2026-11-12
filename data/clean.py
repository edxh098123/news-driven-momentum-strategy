"""Clean collected records without altering originals; report missingness and ambiguous links."""
import json, logging, pathlib
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
logging.basicConfig(filename=ROOT/'logs/cleaning.log',level=logging.INFO,format='%(asctime)s %(message)s')
links=pd.read_parquet(ROOT/'data/metadata/ravenpack_permno_mapping.parquet') if (ROOT/'data/metadata/ravenpack_permno_mapping.parquet').exists() else pd.DataFrame()
report=[]
for f in sorted((ROOT/'data/raw').rglob('*.parquet')):
 df=pd.read_parquet(f); n=len(df); df=df.drop_duplicates(); removed=n-len(df); extra={}
 if f.stem.startswith('ravenpack_') and f.stem[-4:].isdigit() and not links.empty:
  df['publication_timestamp_utc']=pd.to_datetime(df['timestamp_utc'],utc=True,errors='coerce')
  df['_row']=range(len(df)); mapped=df[['_row','rp_entity_id','rpa_date_utc']].merge(links,on='rp_entity_id',how='left')
  dt=pd.to_datetime(mapped.rpa_date_utc); mapped=mapped[(dt>=pd.to_datetime(mapped.start_date))&(dt<=pd.to_datetime(mapped.end_date))]
  mapped=mapped[['_row','permno']].drop_duplicates(); counts=mapped.groupby('_row').permno.nunique(); valid=counts[counts==1].index
  df=df.merge(mapped[mapped._row.isin(valid)],on='_row',how='left').drop(columns='_row')
  extra={'ambiguous_rows':int((counts>1).sum()),'unmapped_rows':int(df.permno.isna().sum())}
 out=ROOT/'data/processed'/f.relative_to(ROOT/'data/raw'); out.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(out,index=False,compression='zstd')
 report.append({'file':str(f.relative_to(ROOT)), 'raw_rows':n,'clean_rows':len(df),'exact_duplicates_removed':removed,'missing_by_column':{k:int(v) for k,v in df.isna().sum().items()},**extra})
 logging.info('%s rows=%s duplicates=%s',f,len(df),removed)
(ROOT/'data/metadata/coverage.json').write_text(json.dumps(report,indent=2))
