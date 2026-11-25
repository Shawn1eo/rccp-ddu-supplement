"""Aggregate the twenty independent fixed-alpha replications and record both denominators."""
import argparse
from pathlib import Path
import json
from math import comb
import numpy as np
import pandas as pd
from scipy.stats import beta
from run_cells import load_source

METHODS=["CCW","CQ-CCP","P-CCP","SAA-CCP","Oracle"]
RHOS=[0.0,0.25,0.5,0.75,1.0]
IDS=list(range(50,70))
def clopper(k,n):
    return (0.0 if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1.0 if k==n else float(beta.ppf(.975,k+1,n-k)))
def rmnan(a):
    return float(np.mean(a)) if len(a) else np.nan
def sem(a):
    return float(np.std(a,ddof=1)/len(a)**.5) if len(a)>1 else np.nan

def aggregate(source, cells, output):
    output.mkdir(parents=True,exist_ok=True)
    m=load_source(source.resolve());cfg=m.ExperimentConfig()
    records=[];contexts=[];saadiag=[];audit=[];method_recs=[]
    for rho in RHOS:
        for rep in IDS:
            files=sorted(cells.glob('rho_%02d_rep_%03d_*.json'%(round(rho*10),rep)))
            if not files:raise ValueError(f"Missing rho={rho}, rep={rep}")
            raw=[json.loads(p.read_text(encoding="utf8")) for p in files]
            # Prefer original-code output when numerical cross-checks cover a cell twice.
            cur=next((it for it in raw if it['engine']=='original'),raw[0])
            ordered={row['method']:row for row in cur['rows']}
            assert sorted(ordered)==sorted(METHODS)
            rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,int(1000*rho),rep]))
            train=m.generate_historical_data(rng,cfg.n_train,rho,cfg)
            demand=train['demand'].to_numpy()
            q10=np.quantile(demand,0.10,method="higher")
            cutoff=cfg.profit_target/(cfg.prices[-1]-cfg.unit_cost)
            saadiag.append(dict(rho=rho,replication=rep,empirical_q10=q10,required_demand=cutoff,zero_fraction=float((demand==0).mean())))
            for method in METHODS:
                row=ordered[method];ds=cur['decisions'][method]
                assert len(ds)==cfg.n_test_contexts
                ok=sum(bool(rec['feasible']) for rec in ds)
                assert bool(row['policy_feasible'])==(ok==80)
                assert abs(row['estimated_context_feasibility']-ok/80)<1e-10
                if method=='SAA-CCP':assert (ok==80)==(q10>=cutoff-1e-10) and ok in (0,80)
                record={**row,'n_contexts_returned':ok,'n_contexts':len(ds),'engine':cur['engine']}
                records.append(record)
                for i,rec in enumerate(ds):
                    contexts.append(dict(rho=rho,replication=rep,method=method,context_id=i,
                        returned=bool(rec['feasible']),price=rec.get('price'),order=rec.get('order'),
                        fitted_target_attainment=rec.get('estimated_reliability')))
    rf=pd.DataFrame(records)
    rf.to_csv(output/'replication_results.csv',index=False)
    pd.DataFrame(contexts).to_csv(output/'context_decisions.csv',index=False)
    pd.DataFrame(saadiag).to_csv(output/'saa_quantile_check.csv',index=False)
    subset=[]
    for rho in RHOS:
        for method in METHODS:
            g=rf[(rf.rho==rho)&(rf.method==method)];counts=g.n_contexts_returned.to_numpy()
            n=len(g);success=int((counts==80).sum());nctx=80*n
            pred=g[g['policy_feasible']]
            interval=clopper(success,n)
            method_recs.append(dict(rho=rho,method=method,alpha=.1,n_reps=n,n_complete=success,
                total_returned_contexts=int(counts.sum()),total_test_contexts=nctx,
                mean_context_return_rate=counts.sum()/nctx,policy_feasible_rate=success/n,
                complete_rate_cp95_low=interval[0],complete_rate_cp95_high=interval[1],
                min_contexts_returned=int(min(counts)),median_contexts_returned=float(np.median(counts)),
                max_contexts_returned=int(max(counts)),mean_contexts_returned=float(np.mean(counts)),
                mean_context_rate_se=sem(counts/80.),
                expected_profit=rmnan(pred.expected_profit),profit_se=sem(pred.expected_profit),
                reliability=rmnan(pred.reliability),reliability_se=sem(pred.reliability)))
            for k in [1,5,10,20,40,80]:
                pr=np.array([comb(int(t),k)/comb(80,k) if t>=k else 0.0 for t in counts])
                subset.append(dict(rho=rho,method=method,contexts_required=k,probability=float(pr.mean()),replicate_se=sem(pr)))
    pd.DataFrame(subset).to_csv(output/'subset_completion.csv',index=False)
    final=pd.DataFrame(method_recs)
    final.to_csv(output/'summary.csv',index=False)
    print(final[['rho','method','n_complete','total_returned_contexts','mean_context_return_rate','min_contexts_returned','max_contexts_returned','reliability']].to_string(index=False))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',required=True,type=Path);a.add_argument('--cells',required=True,type=Path);a.add_argument('--out',required=True,type=Path)
    o=a.parse_args();aggregate(o.source,o.cells,o.out)
