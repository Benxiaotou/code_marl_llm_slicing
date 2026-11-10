"""Publication-ready descriptive paired effects; reads only completed summary."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(folder):
    folder=Path(folder).resolve()
    source=folder/'summary.json'; assert source.is_file() and source.stat().st_size<500000
    summary=json.loads(source.read_text(encoding='utf8'))
    audit=folder/'audit.json'; assert audit.is_file() and audit.stat().st_size<10000
    assert summary['status']=='COMPLETE' and json.loads(audit.read_text(encoding='utf8'))['passed']
    paths=[folder/('ablation_effects.'+suffix) for suffix in ('png','pdf','json')]
    assert not any(p.exists() for p in paths)
    policies=('MAPPO','kNN','Qwen'); modes=('link','node','both')
    colors={'MAPPO':'#2266aa','kNN':'#a96315','Qwen':'#168765'}
    labels=[p+' + '+m for p in policies for m in modes]
    ys=np.arange(9)[::-1]; specs=(('V','Mean system utility V',1),
        ('g_done','Admission rate (percentage points)',100),
        ('cooperative_valid','Cooperative-valid rate (percentage points)',100),
        ('cost_sum','Mean cost',1))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for ax,(metric,title,scale) in zip(axs.flat,specs):
        for y,p,m in zip(ys,[p for p in policies for _ in modes],modes*3):
            c=summary['comparisons'][p+'_'+m+'_minus_raw'][metric]
            val=c['mean_delta']*scale; lo,hi=np.array(c['CI95'])*scale
            ax.errorbar(val,y,xerr=np.array([[max(0,val-lo)],[max(0,hi-val)]]),
                        fmt='o',color=colors[p],capsize=3,markersize=5,elinewidth=1.4)
        ax.axvline(0,color='#555555',linestyle='--',linewidth=.9)
        for y in (2.5,5.5):
            ax.axhline(y,color='#dddddd',linewidth=.6)
        ax.set_yticks(ys); ax.set_yticklabels(labels)
        ax.tick_params(axis='y',labelleft=True)
        ax.set_xlabel('Change relative to the SAME policy without correction')
        ax.set_title(title,loc='left',fontweight='bold')
        ax.grid(axis='x',alpha=.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_ylim(-.7,8.7)
    stage='Existing test attribution' if summary['kind']=='existing_test_attribution' else 'Untouched test confirmation'
    fig.suptitle(stage+' | 1,536 paired scenes, fixed postprocessors',fontsize=14,y=.985)
    fig.text(.5,.017,'Points: paired mean changes. Bars: descriptive 95% seed-stratified bootstrap intervals.\n'
             'These ablation intervals are not the Bonferroni-adjusted intervals for the three prespecified V claims.',
             ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.075,1,.955),h_pad=2,w_pad=2)
    fig.savefig(paths[0],dpi=220); fig.savefig(paths[1]); plt.close(fig)
    manifest={'source_summary':str(source),'summary_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'intervals':'descriptive95%,not multiplicity-adjusted','API_calls':0}
    with paths[2].open('x',encoding='utf8') as f:
        json.dump(manifest,f,indent=2)
    print('PLOTS_CREATED',str(paths[0]),str(paths[1]))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run_dir'); args=p.parse_args(); main(args.run_dir)
