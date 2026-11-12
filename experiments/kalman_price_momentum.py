"""Fixed-parameter Kalman experiment using the shared price comparison runner."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.price_signal_comparison import main
from method.kalman_momentum import build_panels, stability_table

def build(returns,calendar,members,tradable,factors,start,end):
    return build_panels(returns,calendar,members,tradable,start,end)

def report(out,panels,calendar,members,tradable,start,end):
    t=stability_table(panels,calendar,members,tradable,start,end)
    t.to_csv(out/'ranking_stability_weekly.csv',index=False)
    cols=['raw_rank_spearman','filtered_rank_spearman','raw_mean_abs_rank_change','filtered_mean_abs_rank_change','raw_mean_abs_log_score_change','filtered_mean_abs_log_score_change','raw_long_retention','filtered_long_retention','raw_short_retention','filtered_short_retention']
    t['month_changed']=t.date.dt.to_period('M')!=t.previous_date.dt.to_period('M')
    groups=[]
    for window,g in t.groupby('window'):
        for label,x in [('all_weekly',g),('month_boundary',g[g.month_changed]),('signal_update',g[g.raw_mean_abs_log_score_change>1e-12])]:
            groups.append({'window':window,'sample':label,'observations':len(x),**x[cols].mean().to_dict()})
    import pandas as pd
    pd.DataFrame(groups).to_csv(out/'ranking_stability_summary.csv',index=False)

if __name__=='__main__':
    main(experiment='kalman_price_momentum',kinds=('price','intermediate','kalman_12_2','kalman_12_7'),builder=build,warm='2017-01-01',use_factors=False,extra_report=report)
