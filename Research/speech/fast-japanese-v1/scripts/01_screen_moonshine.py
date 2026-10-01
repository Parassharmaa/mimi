#!/usr/bin/env python3
import argparse,hashlib,json,math,os,platform,subprocess,time,unicodedata
from pathlib import Path

def normalized(text):return [c for c in unicodedata.normalize('NFKC',text).casefold() if unicodedata.category(c)[0] not in {'P','S','Z','C'}]
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
def main():
 p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('models',type=Path);p.add_argument('suite',type=Path);p.add_argument('output',type=Path);p.add_argument('--mode',choices=['direct','paced'],default='direct');p.add_argument('--limit',type=int);p.add_argument('--candidates',nargs='+',choices=['small','tiny'],default=['small','tiny']);p.add_argument('--interval',type=float,default=.5);args=p.parse_args()
 rows=[json.loads(x) for x in args.suite.read_text().splitlines() if x.strip()]
 if args.limit:rows=rows[:args.limit]
 attempts=[]
 for index,row in enumerate(rows):
  for model in (args.candidates if index%2==0 else list(reversed(args.candidates))):
   start=time.perf_counter();command=[str(args.probe.resolve()),str((args.models/model).resolve()),'4' if model=='small' else '2',str((args.suite.parent/row['audio']).resolve()),args.mode,str(args.interval)]
   record={'caseID':row['caseID'],'candidate':model,'mode':args.mode,'reference':row['reference'],'audioSHA256':row['audioSha256'],'outcome':'failed','timeoutSeconds':120,'intervalSeconds':args.interval}
   try:
    completed=subprocess.run(command,capture_output=True,text=True,timeout=120)
    record.update(exitCode=completed.returncode,stderr=completed.stderr)
    if completed.returncode:record['failureClass']='runtime';record['rawOutput']=completed.stdout
    else:
     result=json.loads(completed.stdout.strip().splitlines()[-1]);record['result']=result
     reference=normalized(row['reference']);hypothesis=normalized(result['finalText']);edits=distance(hypothesis,reference)
     first=normalized(result['updates'][0]['text']) if result['updates'] else []
     record.update(outcome='passed' if hypothesis else 'failed',characterEdits=edits,referenceCharacters=len(reference),CER=edits/max(1,len(reference)),firstPrefixEdits=distance(first,reference[:len(first)]),firstPrefixCharacters=len(first),firstPrefixCoverage=len(first)/max(1,len(reference)))
     if not hypothesis:record['failureClass']='blank-final'
   except subprocess.TimeoutExpired:record['failureClass']='timeout';record['elapsedSeconds']=120
   except (ValueError,KeyError) as error:record['failureClass']='report-parse';record['error']=str(error)
   record.setdefault('elapsedSeconds',time.perf_counter()-start);attempts.append(record)
   args.output.parent.mkdir(parents=True,exist_ok=True)
   args.output.write_text(json.dumps({'schemaVersion':1,'runtime':'Moonshine native v0.1.5 CPU','hardware':platform.machine(),'mode':args.mode,'suiteSHA256':hashlib.file_digest(args.suite.open('rb'),'sha256').hexdigest(),'runtimeEnvironment':{k:os.environ[k] for k in ['MIMI_MOONSHINE_STEADY_INTERVAL','MIMI_MOONSHINE_VAD_WINDOW','MIMI_MOONSHINE_PROVIDER','MOONSHINE_ORT_SINGLE_THREAD'] if k in os.environ},'attempts':attempts},ensure_ascii=False,indent=2)+'\n')
   print(f'{row["caseID"]} {model}: {record["outcome"]}, CER={record.get("CER")}, RTF={record.get("result",{}).get("computeRTF")}',flush=True)
if __name__=='__main__':main()
