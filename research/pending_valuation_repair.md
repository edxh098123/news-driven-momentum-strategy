# Pending valuation repair

The former experiment-local `StrictObserved.process` intercepted MP and raised before the shared `MissingReturnHandler` could enter pending valuation. The shared handler already recorded stale valuation and required a matching Pn duration and prior valid price date before recognizing the cumulative CRSP return once.

The experiment now uses that handler with additional action-audit fields. A prior price mark is a temporary carrying value, not an observed zero market return. Daily records include `pending_valuation`; risk statistics calculated from these temporarily marked daily NAVs are provisional even if the cumulative return is subsequently recovered. This marking is causal and does not backdate the recovery.

The engine prohibits execution in a pending security. In this strict experiment it stops if a rebalance would require trades while a holding is pending. It requires daily CRSP ordinary/nonordinary dividends, price-adjustment factor and distribution flag to indicate no corporate action throughout the pending and recovery interval, and requires total and ex-dividend returns to agree on recovery. Missing audit information, a corporate action, unverified recovery or a prolonged stale period results in an incomplete path. These restrictions are general, not PERMNO-specific. Delisting handling and all parameters, ranking, dates and costs remain unchanged.

VRTX's 2015-05-12 observation is MP, 2015-05-13 is P1 referring to 2015-05-11, with no distribution or split fields on either day. The P1 total return is recognized once on May 13; no daily market return is inferred for May 12. Values from the prior prematurely stopped run remain archived separately.
