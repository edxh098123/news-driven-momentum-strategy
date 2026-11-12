import psycopg2
c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',user='edxh',connect_timeout=15)
with c.cursor() as q:
 q.execute("select schema_name from information_schema.schemata where schema_name ~* '(raven|rpa|refinitiv|reuters|lseg|ciq|capital|ibes|news)' order by 1")
 print('SCHEMAS',q.fetchall())
 q.execute("select n.nspname,c.relname from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid=c.relnamespace where n.nspname ~* '(raven|rpa|refinitiv|reuters|lseg|ciq|capital|ibes|news)' and c.relkind in ('r','v','m','p') and has_schema_privilege(n.oid,'usage') and has_table_privilege(c.oid,'select') order by 1,2")
 for row in q.fetchall(): print('.'.join(row))
c.close()
