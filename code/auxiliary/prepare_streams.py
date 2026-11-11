#!/usr/bin/env python3
import csv, gzip, hashlib, json, os, struct, zipfile, collections, resource, time
from pathlib import Path
root=Path(__file__).resolve().parent
archive=root/'covertype.zip'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def save(name,obj): (root/name).write_text(json.dumps(obj,indent=2)+'\n')
starts=[0,110000,220000,330000,440000]; ids=[101,211,307,401,503]
parts={'X':(0,1024),'Y':(2048,1024),'replacement':(4096,8192)}
streams=[]; outputs={}; counts=collections.Counter(); mins=[None]*54; maxs=[None]*54
(root/'selected').mkdir(exist_ok=True)
for seed,start in zip(ids,starts):
 s={'stream_id':seed,'seed':seed,'region_start':start,'arrays':{}}
 for part,(offset,n) in parts.items():
  p=root/'selected'/f'{seed}_{part}.i32'; outputs[(seed,part)]=(p.open('wb'),hashlib.sha256(),0)
  s['arrays'][part]={'path':str(p.relative_to(root)),'shape':[n,54],'dtype':'<i4','source_row_start':start+offset,'source_row_stop_exclusive':start+offset+n}
 streams.append(s)
t0=time.monotonic(); datahash=hashlib.sha256(); size=0; rows=0
with zipfile.ZipFile(archive) as z:
 entries=[{'name':i.filename,'uncompressed_size':i.file_size,'compressed_size':i.compress_size,'crc32':f'{i.CRC:08x}'} for i in z.infolist()]
 candidates=[i for i in z.infolist() if i.filename=='covtype.data.gz']
 if len(candidates)!=1: raise ValueError('expected one data member')
 with z.open(candidates[0]) as inner, gzip.GzipFile(fileobj=inner) as data:
  for raw in data:
   datahash.update(raw); size+=len(raw)
   try:
    fields=next(csv.reader([raw.decode('ascii').rstrip('\r\n')],strict=True))
    if len(fields)!=55 or any(not x or x.strip()!=x for x in fields): raise ValueError('field count or missing')
    vals=[int(x) for x in fields]
    if any(str(v)!=x for v,x in zip(vals,fields)): raise ValueError('noncanonical integer')
    if not 1<=vals[54]<=7: raise ValueError('target out of range')
    if any(v < -2147483648 or v > 2147483647 for v in vals[:54]): raise ValueError('int32 overflow')
   except Exception as e: raise ValueError(f'row {rows}: {e}') from e
   counts[str(vals[54])]+=1
   for j,v in enumerate(vals[:54]):
    mins[j]=v if mins[j] is None else min(mins[j],v); maxs[j]=v if maxs[j] is None else max(maxs[j],v)
   for s in streams:
    for part,a in s['arrays'].items():
     if a['source_row_start']<=rows<a['source_row_stop_exclusive']:
      key=(s['seed'],part); f,h,n=outputs[key]; f.write(struct.pack('<54i',*vals[:54])); h.update(raw); outputs[key]=(f,h,n+1)
   rows+=1
if rows!=581012: raise ValueError(f'unexpected row count {rows}')
for s in streams:
 for part,a in s['arrays'].items():
  f,h,n=outputs[(s['seed'],part)]; f.close()
  if n!=a['shape'][0]: raise ValueError('incomplete selection')
  a.update(raw_records_sha256=h.hexdigest(),binary_sha256=digest(root/a['path']),byte_size=(root/a['path']).stat().st_size)
source={'source_url':'https://archive.ics.uci.edu/dataset/31/covertype','doi':'10.24432/C50K5N','creator':'Jock Blackard','year':1998,'license':'CC BY 4.0','archive_path':str(archive),'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,'archive_entries':entries,'data_member':'covtype.data.gz','uncompressed_data_sha256':datahash.hexdigest(),'uncompressed_data_bytes':size,'validated_source_rows':rows,'numeric_fields_per_row':55,'feature_fields':54,'target_field_zero_based':54,'target_counts':dict(sorted(counts.items())),'missing_or_parse_errors':0,'feature_min':mins,'feature_max':maxs}
save('source_manifest.json',source)
save('stream_manifest.json',{'indexing':'zero-based, stop exclusive','selected_actual_rows':5*(1024+1024+8192),'streams':streams,'class_target_excluded':True,'binary_layout':'row-major little-endian signed int32','raw_record_hash':'SHA256 concatenated original uncompressed record bytes, including line endings, in row order'})
# Independent small-matrix adapter parity using actual selected arrays and exact Python integers.
def read(seed,part,n):
 with (root/f'selected/{seed}_{part}.i32').open('rb') as f: return [list(struct.unpack('<54i',f.read(216))) for _ in range(n)]
def mul(a,b,q): return [[sum(x*y for x,y in zip(row,col))%q for col in zip(*b)] for row in a]
def trans(a): return [list(x) for x in zip(*a)]
q=2147483647; X=read(101,'X',4); Y=read(101,'Y',4); R=read(101,'replacement',2); D=[[R[i][j]-X[i][j] for j in range(54)] for i in range(2)]; X2=[R[0],R[1],X[2],X[3]]
A=mul(X,trans(Y),q); A2=mul(X2,trans(Y),q); delta=mul(D,trans(Y),q); assert A2==[[(A[i][j]+(delta[i][j] if i<2 else 0))%q for j in range(4)] for i in range(4)]
B=[[1 if i==j or j==i+1 else 0 for j in range(4)] for i in range(4)]; C=mul(A,B,q); C2=mul(A2,B,q); H=mul(trans(Y),B,q); dh=mul(D,H,q); assert C2==[[(C[i][j]+(dh[i][j] if i<2 else 0))%q for j in range(4)] for i in range(4)]
save('parity_check.json',{'status':'passed','n':4,'b':2,'q':q,'stream_id':101,'B':'upper triangular with unit diagonal and superdiagonal; determinant 1','identities':['updated X Y^T = A + E D Y^T','updated C = C + E D (Y^T B)'],'scope':'adapter parity only, exact Python integers modulo q','runtime_seconds':time.monotonic()-t0,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
