"""Fixed dynamic quality/temporal tradeoffs; other metrics remain in full tables."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exploration_r2_data import ROOT


def main():
    out=ROOT/'research/4.temporal-consistency/report/exploration-r20';rows=json.loads((out/'domain_summary.json').read_text())
    styles={'raw':('black','*',130),'recursive_half01':('#777777','X',85),
        'pre01':('#228833','^',75),'pre05':('#228833','v',75),
        'postwarp01':('#CC6677','o',70),'postwarp05':('#CC6677','D',70),
        'postaligned01':('#4477AA','s',70),'postaligned05':('#4477AA','P',75)}
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,domain in zip(axes.flat,['test','dev','dev_native','legacy_synthetic']):
        for method,(color,marker,size) in styles.items():
            r=next(r for r in rows if r['domain']==domain and r['method']==method and not r['input_reset'])
            ax.scatter(r['psnr_rgb'],r['temporal_error'],c=color,marker=marker,s=size,label=method)
        ax.set(title=domain+' / raw reset off',xlabel='RGB PSNR (higher better)',ylabel='GT-residual temporal error (lower better)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('R20: same R14 native4000 weights / same teacher flow, two-seed means\nFixed finite alpha .1/.5 and recursive .1; reused exploration clips. SSIM/edge and events in full tables; no speed claim.',fontsize=12)
    for extension in ['png','pdf']:fig.savefig(out/('dynamic_order_tradeoffs.'+extension),dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
