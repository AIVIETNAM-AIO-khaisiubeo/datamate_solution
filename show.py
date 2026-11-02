import pickle,sys,json,random,os
res=pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'res_{sys.argv[1]}.pkl'),'rb'))
h=sys.argv[2]; n=int(sys.argv[3]) if len(sys.argv)>3 else 30
sc=int(sys.argv[4]) if len(sys.argv)>4 else -1
xs=[r for r in res if (r.get('handler')==h or h=='*' or (h=='None' and r.get('handler') is None)) and r['score']==sc]
random.seed(0); random.shuffle(xs)
for r in xs[:n]:
  g=r['gold']
  print(r['b'],r['as_of'],'|',r['q'])
  print('    pred:',r['type'],json.dumps(r.get('answer'),ensure_ascii=False)[:150],r.get('reason',''),' gold:',g['type'],json.dumps(g['answer'],ensure_ascii=False)[:150])
