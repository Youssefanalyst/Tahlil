"""Extra formula functions for DataFrameModel.
All functions must be UPPERCASE and accept primitive python types.
They should avoid heavy dependencies and raise no exceptions for normal numeric input.
"""
from __future__ import annotations
import math, statistics as _stat
from typing import Iterable, List

def _flat_numeric(it: Iterable):
    for v in it:
        if isinstance(v, (list, tuple)):
            yield from _flat_numeric(v)
        else:
            try:
                yield float(v)
            except Exception:
                continue

def AR_MEAN(*args):
    vals=list(_flat_numeric(args))
    return sum(vals)/len(vals) if vals else 0

def MEDIAN_ABS_DEV(*args):
    vals=list(_flat_numeric(args))
    if not vals:
        return 0
    med=statistics.median(vals)
    return statistics.median(abs(v-med) for v in vals)

def RANGE(*args):
    vals=list(_flat_numeric(args))
    return max(vals)-min(vals) if vals else 0

def SEMI_VARIANCE(*args):
    vals=list(_flat_numeric(args))
    mean=sum(vals)/len(vals) if vals else 0
    neg=[(v-mean)**2 for v in vals if v<mean]
    return sum(neg)/len(neg) if neg else 0

def COEFFICIENT_VAR(*args):
    vals=list(_flat_numeric(args))
    mean=sum(vals)/len(vals) if vals else 0
    return (_stat.stdev(vals)/mean) if mean else 0

def GINI(range_):
    vals=sorted(_flat_numeric(range_))
    n=len(vals)
    if n==0:
        return 0
    cum=0
    for i,v in enumerate(vals,1):
        cum+=i*v
    return (2*cum)/(n*sum(vals)) - (n+1)/n

def HURST(range_):
    vals=list(_flat_numeric(range_))
    n=len(vals)
    if n<20:
        return "#ERR"
    mean=sum(vals)/n
    dev=[v-mean for v in vals]
    cumsum=[sum(dev[:i+1]) for i in range(n)]
    R=max(cumsum)-min(cumsum)
    S=_stat.stdev(vals)
    return math.log(R/S)/math.log(n/2) if S else "#ERR"

def ENTROPY(*args):
    vals=list(_flat_numeric(args))
    total=sum(vals)
    if total==0:
        return 0
    return -sum((v/total)*math.log(v/total,2) for v in vals if v)

def CAGR(start, end, periods):
    start=float(start); end=float(end); periods=float(periods)
    return (end/start)**(1/periods)-1 if start>0 and periods>0 else "#ERR"

def NPV(rate, *cashflows):
    rate=float(rate)
    return sum(cf/(1+rate)**i for i,cf in enumerate(cashflows,1))

def IRR(*cashflows):
    guess=0.1
    for _ in range(50):
        f=sum(cf/(1+guess)**i for i,cf in enumerate(cashflows,1))
        df=sum(-i*cf/(1+guess)**(i+1) for i,cf in enumerate(cashflows,1))
        if df==0:
            break
        new=guess-f/df
        if abs(new-guess)<1e-6:
            return new
        guess=new
    return "#ERR"

def PMT(rate, nper, pv, fv=0, typ=0):
    rate=float(rate); nper=int(nper); pv=float(pv); fv=float(fv)
    if rate==0:
        return -(pv+fv)/nper
    fac=(1+rate)**nper
    return -(pv*fac+fv)/( (1+rate*typ)*(fac-1)/rate )

def PAYBACK_PERIOD(*cashflows):
    cum=0
    for i,cf in enumerate(cashflows,1):
        cum+=cf
        if cum>=0:
            return i
    return "#ERR"

def LOGISTIC(x,L=1,k=1,x0=0):
    x=float(x);L=float(L);k=float(k);x0=float(x0)
    return L/(1+math.exp(-k*(x-x0)))

def SIGMOID(x):
    x=float(x)
    return 1/(1+math.exp(-x))

def SOFTMAX(*args):
    vals=[float(v) for v in args]
    m=max(vals)
    ex=[math.exp(v-m) for v in vals]
    s=sum(ex)
    return [e/s for e in ex]

def MOM(range_, period: int=1):
    vals=list(_flat_numeric(range_))
    period=int(period)
    return vals[-1]-vals[-1-period] if len(vals)>period else "#ERR"

def ROC_VAL(range_, period:int=1):
    vals=list(_flat_numeric(range_))
    period=int(period)
    return (vals[-1]-vals[-1-period])/vals[-1-period] if len(vals)>period and vals[-1-period]!=0 else "#ERR"

def SMA(range_, window:int=3):
    vals=list(_flat_numeric(range_)); w=int(window)
    return sum(vals[-w:])/w if len(vals)>=w else "#ERR"

def EMA(range_, alpha:float=0.2):
    vals=list(_flat_numeric(range_)); alpha=float(alpha)
    ema=None
    for v in vals:
        ema=v if ema is None else alpha*v+(1-alpha)*ema
    return ema if ema is not None else "#ERR"

def MACD_VAL(range_, fast:int=12, slow:int=26, signal:int=9):
    fast=float(EMA(range_,2/(fast+1)))
    slow=float(EMA(range_,2/(slow+1)))
    macd=fast - slow
    signal_line=float(EMA([macd],2/(signal+1)))
    return macd - signal_line

def RSI_VAL(range_, period:int=14):
    vals=list(_flat_numeric(range_))
    if len(vals)<period+1:
        return "#ERR"
    gains=[max(0,vals[i]-vals[i-1]) for i in range(1,len(vals))]
    losses=[max(0,vals[i-1]-vals[i]) for i in range(1,len(vals))]
    avg_gain=sum(gains[-period:])/period
    avg_loss=sum(losses[-period:])/period
    rs=avg_gain/avg_loss if avg_loss else math.inf
    return 100-100/(1+rs)

def BOLLWIDTH(range_, window:int=20, k:int=2):
    vals=list(_flat_numeric(range_))
    if len(vals)<window:
        return "#ERR"
    ma=sum(vals[-window:])/window
    sd=_stat.stdev(vals[-window:])
    upper=ma+k*sd
    lower=ma-k*sd
    return (upper-lower)/ma if ma else "#ERR"

def ATR_VAL(high, low, close, period:int=14):
    high=list(_flat_numeric(high)); low=list(_flat_numeric(low)); close=list(_flat_numeric(close))
    if not high or len(high)!=len(low) or len(low)!=len(close):
        return "#ERR"
    trs=[max(h-l,abs(h-c),abs(l-c)) for h,l,c in zip(high,low,close)]
    return SMA(trs,period)

def VWAP_VAL(price, volume):
    p=list(_flat_numeric(price)); v=list(_flat_numeric(volume))
    if len(p)!=len(v) or not p:
        return "#ERR"
    return sum(pi*vi for pi,vi in zip(p,v))/sum(v)

def OBV(close, volume):
    c=list(_flat_numeric(close)); v=list(_flat_numeric(volume))
    if len(c)!=len(v):
        return []
    obv=[0]
    for i in range(1,len(c)):
        if c[i]>c[i-1]:
            obv.append(obv[-1]+v[i])
        elif c[i]<c[i-1]:
            obv.append(obv[-1]-v[i])
        else:
            obv.append(obv[-1])
    return obv

def CHAIKIN_MF(high, low, close, volume):
    high=list(_flat_numeric(high)); low=list(_flat_numeric(low)); close=list(_flat_numeric(close)); volume=list(_flat_numeric(volume))
    if not high or len(high)!=len(low)!=len(close)!=len(volume):
        return "#ERR"
    mfv=[((c-l)-(h-c))/(h-l if h-l else 1)*v for h,l,c,v in zip(high,low,close,volume)]
    return sum(mfv)/sum(volume)

def FORCE_INDEX(close, volume, period:int=1):
    c=list(_flat_numeric(close)); v=list(_flat_numeric(volume))
    if len(c)<=period:
        return "#ERR"
    fi=[(c[i]-c[i-period])*v[i] for i in range(period,len(c))]
    return sum(fi)/len(fi)

# Ensure __all__ for cleaner import
# ---------------- additional 35 functions ----------------
import math, statistics as _stat, itertools

def WEIGHTED_MEAN(values, weights):
    v=list(_flat_numeric(values)); w=list(_flat_numeric(weights))
    if len(v)!=len(w) or not v:
        return "#ERR"
    return sum(vi*wi for vi,wi in zip(v,w))/sum(w)

def MEDIAN_HIGH(*args):
    return _stat.median_high(list(_flat_numeric(args))) if args else 0

def MEDIAN_LOW(*args):
    return _stat.median_low(list(_flat_numeric(args))) if args else 0

def MODE_SCI(*args):
    try:
        return _stat.mode(list(_flat_numeric(args)))
    except Exception:
        return "#N/A"

def MADPERCENT(*args):
    vals=list(_flat_numeric(args))
    med=_stat.median(vals) if vals else 0
    mad=_stat.median(abs(v-med) for v in vals) if vals else 0
    return mad/med if med else 0

def COEFF_DET(y_true, y_pred):
    yt=list(_flat_numeric(y_true)); yp=list(_flat_numeric(y_pred))
    if len(yt)!=len(yp) or not yt:
        return "#ERR"
    mean=sum(yt)/len(yt)
    ss_tot=sum((yi-mean)**2 for yi in yt)
    ss_res=sum((yi-ypi)**2 for yi,ypi in zip(yt,yp))
    return 1- ss_res/ss_tot if ss_tot else 0

def RMSE(y_true, y_pred):
    yt=list(_flat_numeric(y_true)); yp=list(_flat_numeric(y_pred))
    if len(yt)!=len(yp) or not yt:
        return "#ERR"
    return math.sqrt(sum((yi-ypi)**2 for yi,ypi in zip(yt,yp))/len(yt))

def MAE(y_true, y_pred):
    yt=list(_flat_numeric(y_true)); yp=list(_flat_numeric(y_pred))
    if len(yt)!=len(yp) or not yt:
        return "#ERR"
    return sum(abs(yi-ypi) for yi,ypi in zip(yt,yp))/len(yt)

def MAPE(y_true, y_pred):
    yt=list(_flat_numeric(y_true)); yp=list(_flat_numeric(y_pred))
    return sum(abs((yi-ypi)/yi) for yi,ypi in zip(yt,yp) if yi)!=0/len(yt) if yt else "#ERR"

def SMAPE(y_true, y_pred):
    yt=list(_flat_numeric(y_true)); yp=list(_flat_numeric(y_pred))
    return 100/len(yt)*sum(abs(ypi-yi)/((abs(yi)+abs(ypi))/2) for yi,ypi in zip(yt,yp)) if yt else "#ERR"

def DOT(a,b):
    return sum(x*y for x,y in zip(_flat_numeric(a), _flat_numeric(b)))

def L2NORM(range_):
    return math.sqrt(sum(v*v for v in _flat_numeric(range_)))

def L1NORM(range_):
    return sum(abs(v) for v in _flat_numeric(range_))

def EUCLIDEAN(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(_flat_numeric(a), _flat_numeric(b))))

def MANHATTAN(a,b):
    return sum(abs(x-y) for x,y in zip(_flat_numeric(a), _flat_numeric(b)))

def CHEBYSHEV(a,b):
    return max(abs(x-y) for x,y in zip(_flat_numeric(a), _flat_numeric(b)))

def COSINE_SIM(a,b):
    dot=DOT(a,b)
    denom=L2NORM(a)*L2NORM(b)
    return dot/denom if denom else "#ERR"

def JACCARD(a,b):
    sa=set(_flat_numeric(a)); sb=set(_flat_numeric(b))
    return len(sa&sb)/len(sa|sb) if sa|sb else 0

def HAMMING(a,b):
    la=list(_flat_numeric(a)); lb=list(_flat_numeric(b))
    if len(la)!=len(lb):
        return "#ERR"
    return sum(1 for x,y in zip(la,lb) if x!=y)/len(la)

def MINMAX_SCALE(x, min_val, max_val):
    return (float(x)-float(min_val))/(float(max_val)-float(min_val)) if max_val!=min_val else "#ERR"

def ZSCORE_SCALE(x, mean, sd):
    sd=float(sd)
    return (float(x)-float(mean))/sd if sd else "#ERR"

def EXP_SMOOTH(prev, alpha, new):
    return float(alpha)*float(new)+(1-float(alpha))*float(prev)

def SIMPLE_FORECAST(last, growth):
    return float(last)*(1+float(growth))

def KS_TEST_P(a,b):
    return "#NA"  # placeholder – complex calc

def CHISQ_P(observed, expected):
    obs=list(_flat_numeric(observed)); exp=list(_flat_numeric(expected))
    if len(obs)!=len(exp) or not obs:
        return "#ERR"
    chi=sum((o-e)**2/e if e else 0 for o,e in zip(obs,exp))
    return chi

def AIC(ll, k, n):
    return 2*k - 2*float(ll)

def BIC(ll, k, n):
    return math.log(float(n))*k - 2*float(ll)

import uuid, hashlib
# utility misc functions

def UUID():
    return str(uuid.uuid4())

def HASH(text, algo="sha256"):
    algo=str(algo).lower()
    if algo not in hashlib.algorithms_available:
        return "#ERR"
    h=hashlib.new(algo)
    h.update(str(text).encode())
    return h.hexdigest()

def CLIP(x, min_val, max_val):
    x=float(x); min_val=float(min_val); max_val=float(max_val)
    return max(min_val, min(x, max_val))

# remove interest-related formulas
for _rm in ("CAGR","NPV","IRR","PMT","PAYBACK_PERIOD"):
    if _rm in globals():
        del globals()[_rm]

# update __all__
__all__=[n for n in globals() if n.isupper() and callable(globals()[n])]
