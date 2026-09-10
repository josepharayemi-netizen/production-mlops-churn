import numpy as np


def psi(expected:list[float],actual:list[float],bins:int=10)->float:
    edges=np.quantile(np.asarray(expected,dtype=float),np.linspace(0,1,bins+1))
    edges[0],edges[-1]=-np.inf,np.inf
    e=np.histogram(expected,bins=edges)[0]/len(expected)
    a=np.histogram(actual,bins=edges)[0]/len(actual)
    e=np.clip(e,1e-6,None); a=np.clip(a,1e-6,None)
    return float(np.sum((a-e)*np.log(a/e)))


def report(baseline:dict,batch:list[dict])->dict:
    scores={name:round(psi(values,[row[name] for row in batch]),4) for name,values in baseline.items()}
    maximum=max(scores.values(),default=0)
    status="ALERT" if maximum>=.25 else "WARNING" if maximum>=.10 else "STABLE"
    return {"status":status,"max_psi":maximum,"features":scores}
