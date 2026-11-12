"""Summarize collected Parquet schemas and coverage without changing source data."""
import json,pathlib
import pandas as pd
import pyarrow.parquet as pq
root=pathlib.Path(__file__).resolve().parents[1]
dictionary={}; summary={}
for path in sorted((root/'data/raw').rglob('*.parquet')):
 key=str(path.relative_to(root)); pf=pq.ParquetFile(path)
 dictionary[key]=[{'name':f.name,'type':str(f.type)} for f in pf.schema_arrow]
 item={'rows':pf.metadata.num_rows}
 if path.name.startswith('daily_'):
  d=pd.read_parquet(path,columns=['permno','dlycaldt']); item.update(first_date=str(d.dlycaldt.min()),last_date=str(d.dlycaldt.max()),duplicate_permno_dates=int(d.duplicated(['permno','dlycaldt']).sum()),permnos=int(d.permno.nunique()))
 if path.name=='membership.parquet':
  d=pd.read_parquet(path); item.update(first_membership=str(d.mbrstartdt.min()),last_membership=str(d.mbrenddt.max()),permnos=int(d.permno.nunique()))
 summary[key]=item
(root/'data/metadata/data_dictionary.json').write_text(json.dumps(dictionary,indent=2))
(root/'data/metadata/validation.json').write_text(json.dumps(summary,indent=2))
