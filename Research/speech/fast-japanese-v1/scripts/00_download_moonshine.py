#!/usr/bin/env python3
"""Fetch pinned official Japanese native model files and verify size/CRC32C."""
import argparse,base64,hashlib,json,re,subprocess
from pathlib import Path

TABLE=[]
for index in range(256):
 value=index
 for _ in range(8):value=(value>>1)^ (0x82F63B78 if value&1 else 0)
 TABLE.append(value)
def crc32c(path):
 value=0xFFFFFFFF
 with path.open('rb') as handle:
  while chunk:=handle.read(1<<20):
   for byte in chunk:value=TABLE[(value^byte)&255]^(value>>8)
 return ((value^0xFFFFFFFF)&0xFFFFFFFF).to_bytes(4,'big')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('metadata',type=Path);parser.add_argument('destination',type=Path);args=parser.parse_args()
 text=args.metadata.read_text();records=[]
 for match in re.finditer(r'\{((?:\s*"[^"]*"\s*)+),\s*(\d+),\s*"([^"]+)",\s*"crc32c"\}',text):
  url=''.join(re.findall(r'"([^"]*)"',match[1]))
  if not any('/'+name+'-streaming-ja/quantized_26_08_23/' in url for name in ['small','tiny']):continue
  size=int(match[2]);crc=match[3];name='small' if '/small-streaming-ja/' in url else 'tiny'
  destination=args.destination/name/url.rsplit('/',1)[-1];destination.parent.mkdir(parents=True,exist_ok=True)
  if not destination.exists():subprocess.run(['curl','--fail','--location','--silent','--show-error',url,'--output',str(destination)],check=True)
  assert destination.stat().st_size==size,(destination,'size')
  assert base64.b64encode(crc32c(destination)).decode()==crc,(destination,'CRC32C')
  digest=hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()
  records.append({'model':name,'file':destination.name,'url':url,'bytes':size,'publisherCRC32C':crc,'sha256':digest})
  print(f'Verified {name}/{destination.name}: {size} bytes',flush=True)
 assert len(records)==16, len(records)
 (args.destination/'manifest.json').write_text(json.dumps({'source':'Moonshine native metadata 234f60faa0eb388b01cdf7e60aca232af37aefda','files':records},indent=2)+'\n')
if __name__=='__main__':main()
