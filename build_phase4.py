#!/usr/bin/env python3
"""Deterministic Phase 4 ranking. Run: python build_phase4.py --data-dir /mnt/data --output /mnt/data/kuku_phase4/output"""
import argparse,csv,json,re,zipfile
from pathlib import Path
from collections import Counter
WEIGHTS={'hook_score':.25,'cliffhanger_score':.25,'clarity_score':.25,'emotion_score':.15,'coherence_score':.10}
REQUIRED=['candidate_id',*WEIGHTS,'hook_pass','cliffhanger_pass','clarity_pass','coherence_pass','immediate_resolution','boundary_cut_issue','decision','rejection_reasons','hook_evidence','cliffhanger_evidence','brief_reasoning','confidence']
def load(data):
 with zipfile.ZipFile(data/'kuku_phase2_corrected.zip') as z: candidates=json.loads(z.read('promo_candidates.json'))['candidates']
 responses={}
 for batch in range(1,18):
  name=f'batch_{batch:02d}_response.json'
  if batch==1: d=json.loads((data/'batch_01_response_revised.json').read_text())
  elif batch==2:d=json.loads((data/name).read_text())
  else:
   zname=('batches_03-05_evaluations.zip' if batch<=5 else 'kuku_batches_06-10_json.zip' if batch<=10 else 'kuku_remaining_batches_11-17_json.zip')
   with zipfile.ZipFile(data/zname) as z:d=json.loads(z.read(name))
  assert d['batch_id']==f'batch_{batch:02d}',name
  expected=10 if batch<17 else 2
  assert len(d['evaluations'])==expected,(name,len(d['evaluations']))
  for e in d['evaluations']:
   assert e['candidate_id'] not in responses,('duplicate',e['candidate_id'])
   responses[e['candidate_id']]=(batch,e)
 return candidates,responses

def overlap(a,b):
 if a['episode_number']!=b['episode_number']:return 0.0
 intersection=max(0,min(a['end_ms'],b['end_ms'])-max(a['start_ms'],b['start_ms']))
 return intersection/max(1,min(a['duration_s']*1000,b['duration_s']*1000))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'inputs');ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'output');ap.add_argument('--top',type=int,default=20);ap.add_argument('--overlap-threshold',type=float,default=.5)
 args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 candidates,responses=load(args.data_dir);ids={c['candidate_id'] for c in candidates};assert len(ids)==len(candidates)==len(responses)==162
 rows=[];issues=[]
 for c in candidates:
  cid=c['candidate_id']; batch,e=responses[cid]
  missing=[k for k in REQUIRED if k not in e]
  if missing:issues.append(f'{cid} missing {missing}');continue
  for k in WEIGHTS:
   if not isinstance(e[k],(int,float)) or isinstance(e[k],bool) or not 0<=e[k]<=5:issues.append(f'{cid} invalid {k}')
  for k in ['hook_pass','cliffhanger_pass','clarity_pass','coherence_pass','immediate_resolution','boundary_cut_issue']:
   if not isinstance(e[k],bool):issues.append(f'{cid} invalid boolean {k}')
  gates=all(e[k] for k in ['hook_pass','cliffhanger_pass','clarity_pass','coherence_pass']) and not e['boundary_cut_issue']
  if e['decision'] not in ('PASS','REJECT'):issues.append(f'{cid} invalid decision')
  if e['decision']=='PASS' and not gates:issues.append(f'{cid} PASS despite failed gates')
  if e['decision']=='REJECT' and gates:issues.append(f'{cid} REJECT despite passed gates (manual review)')
  score=round(sum(e[k]*w for k,w in WEIGHTS.items()),4)
  rows.append(dict(candidate_id=cid,batch_id=f'batch_{batch:02d}',episode_number=c['episode_number'],source_file=c['source_file'],asset_id=c['asset_id'],scene_id=c['scene_id'],start=c['start_raw'],end=c['end_raw'],start_ms=c['start_ms'],end_ms=c['end_ms'],duration_s=c['duration_s'],weighted_score=score,decision=e['decision'],hook_score=e['hook_score'],cliffhanger_score=e['cliffhanger_score'],clarity_score=e['clarity_score'],emotion_score=e['emotion_score'],coherence_score=e['coherence_score'],hook_pass=e['hook_pass'],cliffhanger_pass=e['cliffhanger_pass'],clarity_pass=e['clarity_pass'],coherence_pass=e['coherence_pass'],immediate_resolution=e['immediate_resolution'],boundary_cut_issue=e['boundary_cut_issue'],confidence=e['confidence'],hook_evidence=e['hook_evidence'],cliffhanger_evidence=e['cliffhanger_evidence'],brief_reasoning=e['brief_reasoning'],rejection_reasons=' | '.join(e['rejection_reasons']),boundary_began_before=' | '.join(c.get('boundary_event_ids',{}).get('began_before_clip',[])),boundary_continues_after=' | '.join(c.get('boundary_event_ids',{}).get('continues_after_clip',[]))))
 if issues:raise ValueError('\n'.join(issues))
 rows.sort(key=lambda r:(r['decision']!='PASS',-r['weighted_score'],-r['cliffhanger_score'],-r['hook_score'],r['candidate_id']))
 passed=[r for r in rows if r['decision']=='PASS'];candidate_lookup={c['candidate_id']:c for c in candidates}
 # Greedy overlap-aware shortlist, never modify source boundaries or ranking scores.
 shortlist=[];suppressed=[]
 for r in passed:
  if len(shortlist)>=args.top:break
  collisions=[(s,overlap(r,s)) for s in shortlist if overlap(r,s)>=args.overlap_threshold]
  if collisions:
   s,ratio=max(collisions,key=lambda x:x[1]);suppressed.append(dict(candidate_id=r['candidate_id'],overlaps_with=s['candidate_id'],overlap_fraction_of_shorter=round(ratio,3)))
  else:shortlist.append(r)
 # Remaining eligible candidates may also overlap; report all same-episode overlap pairs above threshold.
 pairs=[]
 for i,a in enumerate(passed):
  for b in passed[i+1:]:
   ratio=overlap(a,b)
   if ratio>=args.overlap_threshold:pairs.append(dict(candidate_a=a['candidate_id'],candidate_b=b['candidate_id'],episode=a['episode_number'],overlap_fraction_of_shorter=round(ratio,3)))
 def csvwrite(path,data):
  if not data:return
  with path.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 csvwrite(args.output/'all_ranked_candidates.csv',rows);csvwrite(args.output/'eligible_pass_candidates.csv',passed);csvwrite(args.output/'top20_diversified.csv',shortlist);csvwrite(args.output/'overlap_pairs.csv',pairs);csvwrite(args.output/'suppressed_during_top20.csv',suppressed)
 detail=[]
 for rank,r in enumerate(shortlist,1):
  c=candidate_lookup[r['candidate_id']]
  detail.append(dict(rank=rank,**r,screenplay_events=[dict(category=e['category'],speaker=e.get('speaker'),start=e['start_raw'],end=e['end_raw'],text=e['text']) for e in c['events']],lookahead=c.get('lookahead'),boundary_event_ids=c.get('boundary_event_ids'),video_verified=False))
 (args.output/'top20_editor_review.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
 counts=Counter(r['decision'] for r in rows)
 report=dict(total=len(rows),passed=counts['PASS'],rejected=counts['REJECT'],pass_rate=round(counts['PASS']/len(rows),4),top_n_requested=args.top,top_n_selected=len(shortlist),overlap_threshold=args.overlap_threshold,overlap_pairs=len(pairs),suppressed_during_selection=len(suppressed),batch_counts={f'batch_{i:02d}':dict(total=sum(r['batch_id']==f'batch_{i:02d}' for r in rows),passed=sum(r['batch_id']==f'batch_{i:02d}' and r['decision']=='PASS' for r in rows)) for i in range(1,18)},validation_errors=issues,video_verified=False)
 (args.output/'phase4_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2));print('TOP 20:',[(r['candidate_id'],r['weighted_score']) for r in shortlist])
if __name__=='__main__':main()
