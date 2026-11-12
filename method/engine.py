"""Close-to-close self-financing long/short accounting."""
import copy
import numpy as np
import pandas as pd
if __package__:
    from .signals import formation_dates,price_signal,news_signal,weights
    from .missing_returns import UnresolvedReturn
else:
    from signals import formation_dates,price_signal,news_signal,weights
    from missing_returns import UnresolvedReturn

class PathIncomplete(ValueError):
    def __init__(self, message, rows, diagnostics):
        super().__init__(message)
        self.frame = pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()
        self.diagnostics = pd.DataFrame(diagnostics)
        self.checkpoint = None

def run(returns, tradable, members, news, calendar, closes, start, end, kind='price', count=10, frequency='weekly', cost=.001, borrow=.01, price_returns=None, delisted=None, price_cache=None, missing_handler=None, crsp_flags=None, checkpoint=None, weight_selector=None, fully_funded=False, formation_schedule=None, target_transform=None, observer=None):
    dates=calendar[(calendar>=pd.Timestamp(start)) & (calendar<=pd.Timestamp(end))]
    forms=set(formation_dates(calendar,frequency) if formation_schedule is None else formation_schedule); month_ends=set(formation_dates(calendar,'monthly')); price_cache=price_cache if price_cache is not None else {}; position=pd.Series(dtype=float); nav=1.; pending=None; rows=[]; diagnostics=[]
    prev_date=None
    if checkpoint is not None:
        if missing_handler is None: raise ValueError('Checkpoint requires a missing handler')
        nav=checkpoint['nav'];position=checkpoint['position'].copy();pending=copy.deepcopy(checkpoint['pending'])
        prev_date=checkpoint['prev_date'];rows=copy.deepcopy(checkpoint['rows']);diagnostics=copy.deepcopy(checkpoint['diagnostics'])
        missing_handler.pending=copy.deepcopy(checkpoint['handler_pending'])
        missing_handler.events=[{**e,'scenario':missing_handler.scenario} for e in copy.deepcopy(checkpoint['handler_events'])]
        dates=dates[dates>=checkpoint['date']]
    for date in dates:
        saved=None
        if missing_handler is not None and crsp_flags is not None and any((date,p) in crsp_flags for p in position.index):
            saved={'date':date,'nav':nav,'position':position.copy(),'pending':copy.deepcopy(pending),'prev_date':prev_date,'rows':copy.deepcopy(rows),'diagnostics':copy.deepcopy(diagnostics),'handler_pending':copy.deepcopy(missing_handler.pending),'handler_events':copy.deepcopy(missing_handler.events)}
        before=nav; long_pnl=short_pnl=0.; fee=borrow_fee=turnover=0.
        prior_position=position.copy() if observer is not None else None; pnl=pd.Series(dtype=float)
        if len(position):
            r=returns.loc[date].reindex(position.index).copy()
            pr=(price_returns if price_returns is not None else returns).loc[date].reindex(position.index).copy()
            if missing_handler is not None:
                if crsp_flags is None: raise ValueError('CRSP flags required for missing return handling')
                for permno in position.index:
                    info=crsp_flags.get((date,permno),{})
                    try:
                        if getattr(missing_handler,'require_action_audit',False) and (permno in missing_handler.pending or ((pd.isna(r.loc[permno]) or pd.isna(pr.loc[permno])) and info.get('dlyretmissflg')=='MP')):
                            audit=info.get('action_audit')
                            if not audit or any(pd.isna(audit.get(k)) for k in ('dlyorddivamt','dlynonorddivamt','dlyfacprc','dlydistretflg')):
                                raise UnresolvedReturn(f'{date.date()} missing corporate-action audit for {permno}')
                            if float(audit['dlyorddivamt'])!=0 or float(audit['dlynonorddivamt'])!=0 or float(audit['dlyfacprc'])!=1 or audit['dlydistretflg']!='NO':
                                raise UnresolvedReturn(f'{date.date()} corporate action during pending valuation: {permno}')
                            if permno in missing_handler.pending and pd.notna(r.loc[permno]) and pd.notna(pr.loc[permno]) and not np.isclose(r.loc[permno],pr.loc[permno],rtol=0,atol=1e-10):
                                raise UnresolvedReturn(f'{date.date()} distribution in unresolved return interval: {permno}')
                        obs=missing_handler.process(permno,date,prev_date,r.loc[permno],pr.loc[permno],info.get('dlyretmissflg'),info.get('dlyretdurflg'),info.get('dlydelflg')=='Y',info.get('dlyprevdt'),info.get('settlement_return'),info.get('settlement_known_at'))
                    except UnresolvedReturn as exc:
                        error=PathIncomplete(str(exc),rows,diagnostics);error.checkpoint=saved
                        raise error from exc
                    r.loc[permno]=obs.total; pr.loc[permno]=obs.ex_dividend
            else:
                if r.isna().any(): raise PathIncomplete(f'{date.date()} missing held returns: {r[r.isna()].index.tolist()}',rows,diagnostics)
                if pr.isna().any(): raise PathIncomplete(f'{date.date()} missing held ex-dividend returns: {pr[pr.isna()].index.tolist()}',rows,diagnostics)
            pnl=position*r; long_pnl=pnl[position>0].sum(); short_pnl=pnl[position<0].sum(); nav+=pnl.sum()
            gap=(date-prev_date).days; borrow_fee=(-position[position<0].sum())*borrow*gap/365.; nav-=borrow_fee
            # Ex-dividend return changes position value; distributions settle in cash.
            position=position*(1+pr)
            if delisted is not None:
                terminated=delisted.loc[date].reindex(position.index,fill_value=False)
                if terminated.any():
                    diagnostics.append(dict(formation=pd.NaT,execution=date,eligible=int(terminated.sum()),reason='CRSP terminal return settled to cash: '+','.join(map(str,position.index[terminated]))))
                    position=position[~terminated]
        if pending is not None:
            target,formation=pending
            idx=position.index.union(target.index); old=position.reindex(idx,fill_value=0); desired=target.reindex(idx,fill_value=0)*nav
            can_trade=tradable.loc[date].reindex(idx,fill_value=False).copy()
            if missing_handler is not None and missing_handler.pending:
                can_trade.loc[can_trade.index.isin(missing_handler.pending)]=False
                if getattr(missing_handler,'require_action_audit',False) and ((desired-old).abs()>1e-12).any():
                    raise PathIncomplete(f'{date.date()} cannot safely rebalance during pending valuation: {sorted(missing_handler.pending)}',rows,diagnostics)
            delta=desired-old; bad=delta[(delta.abs()>1e-12)&(~can_trade)]
            if len(bad):
                # A full basket cannot be executed: liquidate tradable existing positions.
                # Untradeable holdings remain until a later opportunity; never fake a fill.
                diagnostics.append(dict(formation=formation,execution=date,eligible=len(target),reason='untradeable orders: '+','.join(map(str,bad.index))))
                desired=old.where(~can_trade,0.)
            if fully_funded and not len(bad):
                # Solve target dollar holdings against post-transaction-cost equity.
                target_w=target.reindex(idx,fill_value=0.)
                if (target_w<0).any() or target_w.sum()>1+1e-12:
                    raise ValueError('Fully funded targets must be long-only and sum to at most one')
                low,high=0.,nav
                for _ in range(60):
                    mid=(low+high)/2
                    if mid+cost*(target_w*mid-old).abs().sum()>nav: high=mid
                    else: low=mid
                desired=target_w*low
            delta=desired-old
            turnover=delta.abs().sum()/before; fee=delta.abs().sum()*cost; nav-=fee; position=desired[desired.abs()>1e-12]; pending=None
        if nav<=0: raise PathIncomplete('Portfolio equity exhausted',rows,diagnostics)
        if observer is not None: observer(date,prior_position,position.copy(),pnl.copy(),nav,before)
        rows.append(dict(date=date,net=nav/before-1,gross=(nav+fee+borrow_fee)/before-1,long=long_pnl/before,short=short_pnl/before,trade_cost=fee/before,borrow_cost=borrow_fee/before,turnover=turnover,gross_exposure=position.abs().sum()/nav,net_exposure=position.sum()/nav))
        if getattr(missing_handler,'require_action_audit',False): rows[-1]['pending_valuation']=bool(missing_handler.pending)
        if date in forms:
            eligible=set(members.loc[(members.mbrstartdt<=date)&(members.mbrenddt>=date),'permno'])
            variant=('sentiment' if kind=='covered_price' else kind)
            ns=(news[variant].loc[date].dropna() if isinstance(news,dict) else news_signal(news,calendar,closes,date,variant=variant)) if kind!='price' else pd.Series(dtype=float)
            if kind in ('price','covered_price'):
                key=(date.to_period('M'),date in month_ends)
                if key not in price_cache: price_cache[key]=price_signal(returns,calendar,date)
                signal=price_cache[key]
            else: signal=ns
            if kind=='covered_price': signal=signal[signal.index.isin(ns.index)]
            signal=signal[tradable.loc[date].reindex(signal.index,fill_value=False)]
            if missing_handler is not None and missing_handler.pending: signal=signal[~signal.index.isin(missing_handler.pending)]
            target,n=(weights if weight_selector is None else weight_selector)(signal,eligible,count)
            if target_transform is not None: target=target_transform(date,target,rows)
            pending=(target,date)
            minimum=2*count if weight_selector is None else count
            diagnostics.append(dict(formation=date,eligible=n,reason='insufficient eligible securities' if n<minimum else ''))
        prev_date=date
    if missing_handler is not None and missing_handler.pending:
        raise PathIncomplete(f'Unresolved stale holdings at end: {sorted(missing_handler.pending)}',rows,diagnostics)
    return pd.DataFrame(rows).set_index('date'),pd.DataFrame(diagnostics)
