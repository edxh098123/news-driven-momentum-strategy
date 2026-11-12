import psycopg2
c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',user='edxh',connect_timeout=15)
with c.cursor() as q:
 q.execute("select n.nspname,c.relname from pg_class c join pg_namespace n on n.oid=c.relnamespace where c.relname in ('ccmxpf_linktable','ccmxpf_lnkhist','ibcrsphist') and has_schema_privilege(n.oid,'usage') and has_table_privilege(c.oid,'select')")
 for s,t in q.fetchall():
  q.execute('select column_name,data_type from information_schema.columns where table_schema=%s and table_name=%s order by ordinal_position',(s,t));print(s,t,q.fetchall())
