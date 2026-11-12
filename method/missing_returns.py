"""Stateful, point-in-time handling of CRSP missing held-security returns."""
from dataclasses import dataclass
import math
import pandas as pd

class UnresolvedReturn(ValueError):
    pass

@dataclass
class Observation:
    total: float
    ex_dividend: float
    kind: str
    settlement: bool = False
    assumed: bool = False

class MissingReturnHandler:
    def __init__(self, scenario=None, max_stale_sessions=10):
        if scenario not in (None, -.30, -.55, -1.):
            raise ValueError('Unsupported terminal-loss scenario')
        self.scenario=scenario
        self.max_stale_sessions=max_stale_sessions
        self.pending={}
        self.events=[]

    def process(self, permno, date, previous_date, total, ex_dividend, missing_flag, duration, delisted, previous_price_date=None, settlement_return=None, settlement_known_at=None):
        flag='' if pd.isna(missing_flag) else str(missing_flag).strip()
        dur='' if pd.isna(duration) else str(duration).strip()
        missing=pd.isna(total) or pd.isna(ex_dividend)
        terminal=bool(delisted) if not pd.isna(delisted) else False
        item={'date':pd.Timestamp(date),'permno':int(permno),'flag':flag,'duration':dur,'scenario':self.scenario}
        if missing and terminal and flag in ('DG','DM','DP'):
            known=pd.to_datetime(settlement_known_at,errors='coerce')
            if pd.notna(settlement_return) and pd.notna(known) and known.normalize()<=pd.Timestamp(date).normalize() and math.isfinite(float(settlement_return)) and float(settlement_return)>=-1:
                self.pending.pop(permno,None)
                self.events.append({**item,'kind':'observed_delisting_settlement','assumed':False,'settlement_known_at':known})
                return Observation(float(settlement_return),float(settlement_return),'observed_delisting_settlement',True,False)
            if self.scenario is None:
                self.events.append({**item,'kind':'unresolved_delisting','assumed':False})
                raise UnresolvedReturn(f'{date.date()} missing delisting return: {permno} ({flag})')
            self.pending.pop(permno,None)
            self.events.append({**item,'kind':'assumed_delisting_settlement','assumed':True,'assumed_return':self.scenario,'valuation_date':date,'cash_settlement_date':date,'limitation':'Assumed settlement at close; halt and borrow availability unverified'})
            return Observation(self.scenario,self.scenario,'assumed_delisting_settlement',True,True)
        if missing and not terminal and flag=='MP':
            state=self.pending.setdefault(permno,{'count':0,'last_date':pd.Timestamp(previous_date)})
            state['count']+=1
            if state['count']>self.max_stale_sessions:
                self.events.append({**item,'kind':'unresolved_stale','assumed':False})
                raise UnresolvedReturn(f'{date.date()} stale price exceeded {self.max_stale_sessions} sessions: {permno}')
            self.events.append({**item,'kind':'stale_valuation','assumed':False,'stale_sessions':state['count']})
            return Observation(0.,0.,'stale_valuation')
        if missing:
            self.events.append({**item,'kind':'unresolved_other','assumed':False})
            raise UnresolvedReturn(f'{date.date()} unresolved missing return: {permno} ({flag or "no flag"})')
        if not math.isfinite(float(total)) or not math.isfinite(float(ex_dividend)):
            raise UnresolvedReturn(f'{date.date()} invalid non-finite return: {permno}')
        state=self.pending.get(permno)
        if state:
            expected=f'P{state["count"]}'
            prev=pd.to_datetime(previous_price_date,errors='coerce')
            if dur!=expected or pd.isna(prev) or prev.normalize()!=state['last_date'].normalize():
                self.events.append({**item,'kind':'unresolved_recovery','assumed':False})
                raise UnresolvedReturn(f'{date.date()} unverified multi-period recovery: {permno}, {dur}, expected {expected}')
            self.pending.pop(permno)
            self.events.append({**item,'kind':'observed_multiperiod_recovery','assumed':False,'stale_sessions':state['count']})
            return Observation(float(total),float(ex_dividend),'observed_multiperiod_recovery',terminal)
        if dur.startswith('P') and dur[1:].isdigit():
            raise UnresolvedReturn(f'{date.date()} multi-period return without tracked stale position: {permno} ({dur})')
        if terminal:
            self.events.append({**item,'kind':'observed_delisting_return','assumed':False})
        return Observation(float(total),float(ex_dividend),'observed',terminal)
