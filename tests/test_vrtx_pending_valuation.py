import unittest
import numpy as np
import pandas as pd
from method.engine import run,PathIncomplete
from experiments.intermediate_long_history import StrictObserved

class VRTXPendingTests(unittest.TestCase):
    def inputs(self):
        dates=pd.DatetimeIndex(['2015-05-08','2015-05-11','2015-05-12','2015-05-13','2015-05-14']);p=76744
        r=pd.DataFrame({p:[0.,-.026136,np.nan,.011605,0.]},index=dates);tradable=pd.DataFrame(True,index=dates,columns=[p]);tradable.loc['2015-05-12',p]=False
        members=pd.DataFrame({'permno':[p],'mbrstartdt':[pd.Timestamp('2000-01-01')],'mbrenddt':[pd.Timestamp('2020-01-01')]});panels={'x':pd.DataFrame({p:[1.]*5},index=dates)}
        action=dict(dlyorddivamt=0.,dlynonorddivamt=0.,dlyfacprc=1.,dlydistretflg='NO')
        flags={(dates[2],p):dict(dlyretmissflg='MP',dlyretdurflg='MR',dlydelflg='N',action_audit=action.copy()),(dates[3],p):dict(dlyretmissflg='NA',dlyretdurflg='P1',dlydelflg='N',dlyprevdt=dates[1],action_audit=action.copy())}
        return dates,p,r,tradable,members,panels,flags
    def test_vrtx_pending_recovery_and_ledger(self):
        d,p,r,tr,m,panels,flags=self.inputs(); observations={}
        def observe(date,before,after,pnl,nav,prev):observations[date]=(after.copy(),nav,nav-after.sum())
        h=StrictObserved();out,_=run(r,tr,m,panels,d,pd.Series(index=d,dtype=float),str(d[0].date()),str(d[-1].date()),kind='x',count=1,cost=0,borrow=0,missing_handler=h,crsp_flags=flags,weight_selector=lambda s,mem,n:(pd.Series({p:.5}),1),formation_schedule=[d[0]],observer=observe)
        self.assertTrue(out.loc[d[2],'pending_valuation']);self.assertFalse(out.loc[d[3],'pending_valuation'])
        self.assertEqual(out.loc[d[2],'turnover'],0);self.assertAlmostEqual(observations[d[1]][0][p],observations[d[2]][0][p])
        self.assertAlmostEqual(observations[d[3]][0][p],.5*(1+.011605));self.assertAlmostEqual(out.loc[d[3],'net'],.5*.011605)
        self.assertAlmostEqual(observations[d[3]][2],.5);self.assertAlmostEqual(observations[d[3]][1],observations[d[3]][0].sum()+observations[d[3]][2])
        self.assertAlmostEqual((1+out.net).prod(),1+.5*.011605);self.assertEqual([x['kind'] for x in h.events],['stale_valuation','observed_multiperiod_recovery'])
    def test_future_recovery_does_not_change_prior_nav_or_orders(self):
        d,p,r,tr,m,panels,flags=self.inputs()
        def replay(returns):
            return run(returns,tr,m,panels,d,pd.Series(index=d,dtype=float),str(d[0].date()),str(d[-1].date()),kind='x',count=1,cost=0,borrow=0,missing_handler=StrictObserved(),crsp_flags=flags,weight_selector=lambda s,mem,n:(pd.Series({p:.5}),1),formation_schedule=[d[0]])[0]
        before=replay(r);changed=r.copy();changed.loc[d[3],p]=.5;after=replay(changed)
        pd.testing.assert_frame_equal(before.loc[:d[2]],after.loc[:d[2]])

    def test_pending_orders_and_actions_fail_closed(self):
        d,p,r,tr,m,panels,flags=self.inputs();tr.loc[d[2],p]=True;panels['x'].loc[d[1],p]=0.
        with self.assertRaises(PathIncomplete):run(r,tr,m,panels,d,pd.Series(index=d,dtype=float),str(d[0].date()),str(d[-1].date()),kind='x',count=1,cost=0,borrow=0,missing_handler=StrictObserved(),crsp_flags=flags,weight_selector=lambda s,mem,n:(pd.Series({p:.5 if s.loc[p]>0 else 0.}),1),formation_schedule=[d[0],d[1]])
        flags[(d[3],p)]['action_audit']['dlyorddivamt']=1.
        with self.assertRaises(PathIncomplete):run(r,tr,m,panels,d,pd.Series(index=d,dtype=float),str(d[0].date()),str(d[-1].date()),kind='x',count=1,cost=0,borrow=0,missing_handler=StrictObserved(),crsp_flags=flags,weight_selector=lambda s,mem,n:(pd.Series({p:.5}),1),formation_schedule=[d[0]])

if __name__=='__main__':unittest.main()
