import psycopg2
c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',user='edxh',connect_timeout=15)
tables=[('ravenpack_dj','rpa_djpr_equities_2010'),('ravenpack_common','wrds_rpa_company_mappings_hist'),('ravenpack_common','rpa_company_mappings'),('ciq_keydev','wrds_keydev'),('ciq_keydev','ciqkeydev'),('ciq_keydev','ciqkeydevtoobjecttoeventtype'),('wrdsapps_link_crsp_ibes','ibcrsphist'),('ibes','actu_epsus'),('crsp','stocknames_v2'),('crsp','dsf_v2')]
with c.cursor() as q:
 for schema,table in tables:
  q.execute('select column_name,data_type from information_schema.columns where table_schema=%s and table_name=%s order by ordinal_position',(schema,table)); print(schema,table,q.fetchall(),flush=True)
c.close()
