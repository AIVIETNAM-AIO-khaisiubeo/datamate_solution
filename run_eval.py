import sys, os, json, time, pickle, collections, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dill
from datamate.model import build_model
from datamate.intents import answer
from datamate.timeparse import load_calendar
from datamate.scorer import score
DATA_ROOT=os.environ.get('DATA_ROOT','/home/claude/data')
CACHE=os.environ.get('CACHE_DIR','/home/claude/dm/cache'); os.makedirs(CACHE, exist_ok=True)
def get_model(split, b, rebuild=False):
    p=f'{CACHE}/{b}.dill'
    if os.path.exists(p) and not rebuild:
        return dill.load(open(p,'rb'))
    M=build_model(f'{DATA_ROOT}/{split}/data/{b}')
    for a in ('_item_idx',):
        if hasattr(M,a): delattr(M,a)
    dill.dump(M, open(p,'wb'))
    return M
if __name__=='__main__':
    split=sys.argv[1]; only=sys.argv[2].split(',') if len(sys.argv)>2 and sys.argv[2] else None
    rebuild='--rebuild' in sys.argv
    Q=[json.loads(l) for l in open(f'{DATA_ROOT}/{split}/questions.jsonl')]
    A={}
    if os.path.exists(f'{DATA_ROOT}/{split}/answers.jsonl'):
        A={json.loads(l)['id']:json.loads(l) for l in open(f'{DATA_ROOT}/{split}/answers.jsonl')}
    cal=load_calendar(f'{DATA_ROOT}/{split}/calendar.csv')
    by=collections.defaultdict(list)
    sealed=set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'splits','split.json')))['sealed']) if split=='train' else set()
    if sealed and '--include-sealed' not in sys.argv:
        Q=[q for q in Q if q['business_id'] not in sealed]   # sealed shops: use eval_sealed.py (aggregate only)
    for q in Q: by[q['business_id']].append(q)
    res=[]
    for b in sorted(by):
        if only and b not in only: continue
        try:
            M=get_model(split,b,rebuild)
        except Exception as e:
            print('MODEL FAIL',b,e); M=None
        for q in by[b]:
            t=time.time()
            try:
                r=answer(M,q['question'],q['as_of_date'],cal) if M else {'type':'abstain','reason':'model fail'}
            except Exception as e:
                r={'type':'abstain','reason':'EXC '+repr(e)[:200],'handler':'exc'}
            r['id']=q['id']; r['dt']=time.time()-t
            g=A.get(q['id'])
            r['score']=score(r,g) if g else None
            r['gold']=g; r['q']=q['question']; r['b']=b; r['as_of']=q['as_of_date']
            res.append(r)
    pickle.dump(res, open(os.path.join(os.path.dirname(CACHE.rstrip('/')), f'res_{split}.pkl'),'wb'))
    if A:
        n=len(res); s=collections.Counter(r['score'] for r in res)
        print(f'N={n} correct={s[1]} wrong={s[-1]} abstain={s[0]} PA={(s[1]-s[-1])/n:.4f}')
        byh=collections.defaultdict(collections.Counter)
        for r in res: byh[r.get('handler')][r['score']]+=1
        for h,cn in sorted(byh.items(), key=lambda t:-sum(t[1].values())):
            print(f'  {str(h):24s} +{cn[1]:4d} -{cn[-1]:4d} 0:{cn[0]:4d}')
    else:
        print(collections.Counter(r['type'] for r in res))
