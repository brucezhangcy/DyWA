#!/usr/bin/env python3
"""CPU-only routing audit on historical per-object rates; no new robot data.

Binary competence (rate >= .5) probabilities do not equal per-episode success
probabilities. Show actual held-out object-scale success rates under fixed routing
thresholds. Results are descriptive, not a calibrated/deployable trigger.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root=Path(__file__).resolve().parents[1];out=root/'results/session5';out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(root/'results/phase_c_predictor/competence_dataset.csv')
uk=np.unique(d.key);rng=np.random.default_rng(0);rng.shuffle(uk)
fold={key:i%5 for i,key in enumerate(uk)}
f=d.key.map(fold).to_numpy(); y=(d.succ>=.5).astype(int).to_numpy()
rows=[]
for name,features in [('scale_only',['scale']),('scale_shortest_extent',['scale','scaled_S'])]:
    prob=np.zeros(len(d))
    for k in range(5):
        tr,te=f!=k,f==k
        assert not set(d.key[tr])&set(d.key[te])
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
        model.fit(d.loc[tr,features],y[tr]);prob[te]=model.predict_proba(d.loc[te,features])[:,1]
    for threshold in [.5,.7,.8,.9,.95]:
        chosen=prob>=threshold
        rows.append(dict(model=name,threshold=threshold,chosen_rows=int(chosen.sum()),
            coverage=float(chosen.mean()),chosen_objects=int(d.key[chosen].nunique()),
            actual_macro_episode_success=float(d.succ[chosen].mean()) if chosen.any() else None,
            binary_competence_precision=float(y[chosen].mean()) if chosen.any() else None))
    d['oof_'+name]=prob
pd.DataFrame(rows).to_csv(out/'trigger_routing_audit.csv',index=False)
d.to_csv(out/'trigger_oof_predictions.csv',index=False)
(out/'trigger_audit_notes.json').write_text(json.dumps(dict(
    protocol='One fixed five-fold object-disjoint split; thresholds prespecified; exploratory historical rates only',
    limitations=['Only one spawn/seed per object and scale; unknown per-object episode denominators.',
      'succ is a historical point estimate; no certified 90% reliability claim is possible.',
      'Binary competence score predicts rate>=0.5, not episode success.',
      'No matched trained-bimanual outcomes; no upper-envelope/necessity claim.'],rows=rows),indent=2))
print(pd.DataFrame(rows).round(4).to_string(index=False))

# Standalone scientific figure of the same held-out routing table.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
curve=pd.DataFrame(rows);curve=curve[curve.model=='scale_shortest_extent']
fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
axes[0].plot(curve.threshold,curve.actual_macro_episode_success,'o-',label='Historical episode success')
axes[0].plot(curve.threshold,curve.binary_competence_precision,'s--',label='Precision: success rate ≥ 50%')
axes[0].axhline(.9,color='gray',ls=':',label='90% target')
axes[0].set(xlabel='Binary competence score threshold',ylabel='Mean rate',ylim=(.5,1),title='Confidence and reliability differ')
axes[0].legend(fontsize=7)
axes[1].plot(curve.threshold,curve.coverage,'o-',color='#287b70')
axes[1].set(xlabel='Binary competence score threshold',ylabel='Fraction routed to single arm',ylim=(0,1),title='Coverage decreases with threshold')
fig.suptitle('CPU audit of historical data • object-disjoint split • no new robot experiments',fontsize=10)
fig.savefig(out/'trigger_routing_audit.png',dpi=160)
plt.close(fig)
