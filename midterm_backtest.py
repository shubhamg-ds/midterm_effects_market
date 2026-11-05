# midterm_backtest.py - chop detector + long (election->Jul 15) + short (Jul 15->Aug/Sep) backtest on SPX daily
# Usage: edit CSV and OUT below, then:  python midterm_backtest.py
CSV=r'C:\projects\research\mid_terms_effects\data\SPX_1D.csv'
OUT=r'midterm_chop_long_short_backtest.xlsx'
import pandas as pd, numpy as np
def _unused_load():
    d=pd.read_csv('/mnt/user-data/uploads/SPX_1D.csv'); d['time']=pd.to_datetime(d.time,format='%d-%m-%Y')
    d=d[d.time>='1960-01-01'].set_index('time')[['open','high','low','close']].copy()
    d['pc']=d.close.shift()
    d['rng']=(d.high-d.low)/d.pc*100
    d['ret']=d.close.pct_change()*100
    d['gap']=(d.open-d.pc)/d.pc*100
    return d
def elec(y):
    t=pd.Timestamp(y,11,1)
    while t.weekday()!=0: t+=pd.Timedelta(days=1)
    return t+pd.Timedelta(days=1)
YEARS=list(range(1962,2023,4))
def on_or_before(idx,ts): return idx[idx<=ts][-1]
import pandas as pd, numpy as np

def load2():
    d=pd.read_csv(CSV); d['time']=pd.to_datetime(d.time,format='%d-%m-%Y')
    d=d[d.time>='1962-01-01'].set_index('time')[['open','high','low','close']].copy()
    d['pc']=d.close.shift()
    d['rng']=(d.high-d.low)/d.pc*100
    d['ret']=d.close.pct_change()*100
    d['gap']=(d.open-d.pc)/d.pc*100
    return d

def add_chop(d,N=20,pct=0.80):
    d=d.copy()
    thr=d.rng.rolling(250,min_periods=120).quantile(pct).shift(1)   # no look-ahead: threshold from prior 250d
    d['big']=(d.rng>=thr).astype(int)
    d['bigcnt']=d.big.rolling(N).sum()
    g=(d.big==0).cumsum(); run=d.big.groupby(g).cumsum()
    d['maxrun']=run.rolling(N).max()
    net=(d.close-d.close.shift(N)).abs(); path=d.close.diff().abs().rolling(N).sum()
    d['er']=net/path
    d['gap20']=d.gap.abs().rolling(N).mean().where(d.index>='1982-01-01')   # opens are real only from ~1980
    d['atr20']=d.rng.rolling(N).mean()
    return d

def weekly(d):
    w=d.resample('W-FRI').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),
                              first=('close',lambda s: s.index[0]),last=('close',lambda s: s.index[-1])).dropna()
    return w

# ---------------- LONG ----------------
LONG_ENTRIES=['E0_close','NextDay_open','D_brk5','D_brk10','D_ma20','W_brkhigh','W_10wMA','Wk8_Monday']
LONG_SLBUF=[None,0.0,0.01,0.02,0.03]
def sl_name(b): return 'No SL' if b is None else f'2M low -{b*100:.0f}%'
DEADLINE_DAYS=70

def long_signal(d,w,E,rule):
    """return (entry_date, entry_price, field) ; entry at 'close' of E or 'open' of date"""
    idx=d.index; pos=idx.get_loc(E)
    dl=E+pd.Timedelta(days=DEADLINE_DAYS)
    if rule=='E0_close': return E,d.close[E],'close'
    if rule=='NextDay_open': t=idx[pos+1]; return t,d.open[t],'open'
    if rule in('D_brk5','D_brk10'):
        n=5 if rule=='D_brk5' else 10
        hh=d.high.rolling(n).max().shift(1)
        for t in idx[pos+1:]:
            if t>dl: return None
            if d.close[t]>hh[t]:
                nt=idx[idx.get_loc(t)+1]; return nt,d.open[nt],'open'
    if rule=='D_ma20':
        ma=d.close.rolling(20).mean()
        for t in idx[pos+1:]:
            if t>dl: return None
            i=idx.get_loc(t)
            if d.close[t]>ma[t] and ma[t]>ma.iloc[i-5]:
                nt=idx[i+1]; return nt,d.open[nt],'open'
    if rule in('W_brkhigh','W_10wMA'):
        ma=w.close.rolling(10).mean()
        for k in range(1,len(w)):
            if w['last'].iloc[k]<E: continue
            lt=w['last'].iloc[k]
            if lt>dl: return None
            ok = w.close.iloc[k]>w.high.iloc[k-1] if rule=='W_brkhigh' else (w.close.iloc[k]>ma.iloc[k] and ma.iloc[k]>ma.iloc[k-1])
            if ok:
                i=idx.get_loc(lt); nt=idx[i+1]; return nt,d.open[nt],'open'
    if rule=='Wk8_Monday':
        mon=E-pd.Timedelta(days=E.weekday())+pd.Timedelta(days=56)
        t=idx[idx>=mon][0]; return t,d.open[t],'open'
    return None

def run_long(d,w,E,rule,buf,end):
    s=long_signal(d,w,E,rule)
    if s is None: return None
    t0,px,fld=s
    if t0>=end: return None
    prelow=d.low.loc[E-pd.DateOffset(months=2):E].min()
    sl=None if buf is None else prelow*(1-buf)
    if sl is not None and px<=sl: return dict(skip='entry<=SL')
    idx=d.index; i0=idx.get_loc(t0)
    start=i0+1 if fld=='close' else i0
    exit_t=end; exit_px=d.close[end]; why='Jul15 exit'
    if sl is not None:
        for t in idx[start:idx.get_loc(end)+1]:
            if d.low[t]<=sl:
                exit_t=t; exit_px=d.open[t] if d.open[t]<=sl else sl; why='SL hit'; break
    seg=d.loc[idx[start]:exit_t]
    return dict(entry_date=t0,entry_px=px,sl=sl,exit_date=exit_t,exit_px=exit_px,why=why,
                bars=len(d.loc[t0:exit_t])-1 if fld=='close' else len(d.loc[t0:exit_t]),
                mae=seg.low.min()-px, mfe=seg.high.max()-px, prelow=prelow)

# ---------------- SHORT ----------------
SHORT_ENTRIES=['S0_Jul15_close','S_brk10','S_ma20','S_W_brklow']
SHORT_EXITS=['Aug31','Sep30']
SHORT_SL=['No SL','Fixed 3%','Fixed 5%','30d high']
def short_start(d,y): return on_or_before(d.index,pd.Timestamp(y+1,7,15))
def exit_date(d,y,ex): return on_or_before(d.index,pd.Timestamp(y+1,8,31) if ex=='Aug31' else pd.Timestamp(y+1,9,30))

def short_signal(d,w,y,rule):
    idx=d.index; st=short_start(d,y); pos=idx.get_loc(st); dl=pd.Timestamp(y+1,8,15)
    if rule=='S0_Jul15_close': return st,d.close[st],'close',st
    if rule=='S_brk10':
        ll=d.low.rolling(10).min().shift(1)
        for t in idx[pos+1:]:
            if t>dl: return None
            if d.close[t]<ll[t]:
                nt=idx[idx.get_loc(t)+1]; return nt,d.open[nt],'open',t
    if rule=='S_ma20':
        ma=d.close.rolling(20).mean()
        for t in idx[pos+1:]:
            if t>dl: return None
            i=idx.get_loc(t)
            if d.close[t]<ma[t] and ma[t]<ma.iloc[i-5]:
                nt=idx[i+1]; return nt,d.open[nt],'open',t
    if rule=='S_W_brklow':
        for k in range(1,len(w)):
            lt=w['last'].iloc[k]
            if lt<=st: continue
            if lt>dl: return None
            if w.close.iloc[k]<w.low.iloc[k-1]:
                nt=idx[idx.get_loc(lt)+1]; return nt,d.open[nt],'open',lt
    return None

def run_short(d,w,y,rule,ex,slr):
    s=short_signal(d,w,y,rule)
    if s is None: return None
    t0,px,fld,sig=s
    end=exit_date(d,y,ex)
    if t0>=end: return None
    idx=d.index
    if slr=='No SL': sl=None
    elif slr=='Fixed 3%': sl=px*1.03
    elif slr=='Fixed 5%': sl=px*1.05
    else:
        sl=d.high.loc[sig-pd.Timedelta(days=30):sig].max()
        if sl<=px: return dict(skip='30d high<=entry')
    i0=idx.get_loc(t0); start=i0+1 if fld=='close' else i0
    exit_t=end; exit_px=d.close[end]; why=f'{ex} exit'
    if sl is not None:
        for t in idx[start:idx.get_loc(end)+1]:
            if d.high[t]>=sl:
                exit_t=t; exit_px=d.open[t] if d.open[t]>=sl else sl; why='SL hit'; break
    seg=d.loc[idx[start]:exit_t]
    return dict(entry_date=t0,entry_px=px,sl=sl,exit_date=exit_t,exit_px=exit_px,why=why,
                bars=len(d.loc[t0:exit_t])-(1 if fld=='close' else 0),
                mae=px-seg.high.max(), mfe=px-seg.low.min())
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter as CL

d=add_chop(load2()); w=weekly(d)
H=20
d['fwd']=(d.close.shift(-H)/d.close-1)*100
d['fdd']=(d.low.rolling(H).min().shift(-H)/d.close-1)*100

# ---------- cycle windows ----------
cyc={}
for y in YEARS:
    E=on_or_before(d.index,elec(y)); end=on_or_before(d.index,pd.Timestamp(y+1,7,15))
    cyc[y]=(E,end)

FONT='Arial'
f_norm=Font(name=FONT,size=10); f_b=Font(name=FONT,size=10,bold=True)
f_h=Font(name=FONT,size=10,bold=True,color='FFFFFF'); fill_h=PatternFill('solid',fgColor='1F3864')
f_in=Font(name=FONT,size=10,color='0000FF',bold=True); fill_in=PatternFill('solid',fgColor='FFF2CC')
f_t=Font(name=FONT,size=14,bold=True); fill_sec=PatternFill('solid',fgColor='D9E1F2')
thin=Side(style='thin',color='BFBFBF')

wb=Workbook()
def sheet(name,first=False):
    ws=wb.active if first else wb.create_sheet()
    ws.title=name; ws.sheet_view.showGridLines=False; return ws
def hdr(ws,r,c,labels):
    for i,l in enumerate(labels):
        x=ws.cell(r,c+i,l); x.font=f_h; x.fill=fill_h; x.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
    ws.row_dimensions[r].height=32
def put(ws,r,c,v,fmt=None,font=None,fill=None):
    x=ws.cell(r,c,v); x.font=font or f_norm
    if fmt: x.number_format=fmt
    if fill: x.fill=fill
    return x
def widths(ws,ws_w):
    for i,wd in enumerate(ws_w,1): ws.column_dimensions[CL(i)].width=wd
def sec(ws,r,text,ncol=8):
    for c in range(1,ncol+1): ws.cell(r,c).fill=fill_sec
    put(ws,r,1,text,font=f_b,fill=fill_sec)

# =============== README (filled at the end) ===============
ws_r=sheet('README',True)

# =============== Chop_Daily ===============
ws_d=sheet('Chop_Daily')
hdr(ws_d,1,1,['date','cycle','day_no (0=election)','SPX close','range %','big candle (1/0)','big candles in last 20d','longest big-candle run (20d)','efficiency ratio 20d','avg |gap| 20d %','CHOP FLAG','fwd 20d return %','fwd 20d max drawdown %'])
r=2; drow={}
for y in YEARS:
    E,end=cyc[y]; seg=d.loc[E:end].iloc[:-H]
    drow[y]=(r,r+len(seg)-1)
    for k,(t,x) in enumerate(seg.iterrows()):
        put(ws_d,r,1,t.to_pydatetime(),'yyyy-mm-dd'); put(ws_d,r,2,y); put(ws_d,r,3,k)
        put(ws_d,r,4,round(x.close,2),'#,##0.00'); put(ws_d,r,5,round(x.rng,3),'0.00')
        put(ws_d,r,6,int(x.big)); put(ws_d,r,7,None if pd.isna(x.bigcnt) else int(x.bigcnt))
        put(ws_d,r,8,None if pd.isna(x.maxrun) else int(x.maxrun)); put(ws_d,r,9,None if pd.isna(x.er) else round(x.er,3),'0.00')
        put(ws_d,r,10,None if pd.isna(x.gap20) else round(x.gap20,3),'0.00')
        put(ws_d,r,11,f'=IF(AND(G{r}>=Chop_Probs!$C$4,I{r}<=Chop_Probs!$C$5),1,0)')
        put(ws_d,r,12,round(x.fwd,3),'0.00'); put(ws_d,r,13,round(x.fdd,3),'0.00')
        r+=1
DL=r-1
widths(ws_d,[12,8,12,11,9,10,12,14,12,11,10,12,14]); ws_d.freeze_panes='A2'; ws_d.auto_filter.ref=f'A1:M{DL}'
def DR(col): return f'Chop_Daily!${col}$2:${col}${DL}'

# =============== Chop_Probs ===============
ws_p=sheet('Chop_Probs')
put(ws_p,1,1,'Choppy-market detector: stats & probabilities (SPX daily, election day -> Jul 15, 16 midterm cycles 1962-2022)',font=f_t)
put(ws_p,2,1,'Blue/yellow cells are inputs - change them and every table here and the flag column in Chop_Daily recalculates. Forward numbers are close(t) -> close(t+20 trading days); drawdown = lowest low in those 20 days vs close(t).',font=f_norm)
put(ws_p,4,1,'Chop flag: min # of big candles in last 20d',font=f_b); put(ws_p,4,3,6,font=f_in,fill=fill_in)
put(ws_p,5,1,'Chop flag: max efficiency ratio (20d)',font=f_b); put(ws_p,5,3,0.25,'0.00',font=f_in,fill=fill_in)
put(ws_p,6,1,'"Pullback" threshold for fwd 20d drawdown (%)',font=f_b); put(ws_p,6,3,-3,'0.0',font=f_in,fill=fill_in)
put(ws_p,7,1,'Big candle = daily range (H-L)/prev close in the top 20% of the prior 250 days (relative, so it works across eras). Efficiency ratio = |net move over 20d| / sum of |daily moves| (low = lots of motion, no progress = chop).',font=f_norm)

def stat_block(ws,r0,title,labels,conds):
    sec(ws,r0,title,9)
    hdr(ws,r0+1,1,['bucket','','days','% of days','avg fwd 20d ret %','P(fwd 20d > 0) %','P(drawdown <= threshold) %','avg max drawdown %','avg fwd ret per unit of drawdown'])
    for i,(lab,cond) in enumerate(zip(labels,conds)):
        rr=r0+2+i
        put(ws,rr,1,lab,font=f_b)
        put(ws,rr,3,f'=COUNTIFS({cond})','0')
        put(ws,rr,4,f'=C{rr}/COUNT({DR("L")})*100','0.0')
        put(ws,rr,5,f'=IFERROR(AVERAGEIFS({DR("L")},{cond}),"")','0.00')
        put(ws,rr,6,f'=IFERROR(COUNTIFS({cond},{DR("L")},">0")/C{rr}*100,"")','0.0')
        put(ws,rr,7,f'=IFERROR(COUNTIFS({cond},{DR("M")},"<="&$C$6)/C{rr}*100,"")','0.0')
        put(ws,rr,8,f'=IFERROR(AVERAGEIFS({DR("M")},{cond}),"")','0.00')
        put(ws,rr,9,f'=IFERROR(E{rr}/ABS(H{rr}),"")','0.00')
    return r0+2+len(labels)
nr=stat_block(ws_p,9,'A. Chop flag ON vs OFF',['FLAG ON (chop)','FLAG OFF','ALL DAYS'],
   [f'{DR("K")},1',f'{DR("K")},0',f'{DR("A")},">0"'])
bc=[(0,2),(3,5),(6,8),(9,11),(12,20)]
nr=stat_block(ws_p,nr+1,'B. By number of big candles in the last 20 days',[f'{a}-{b} big candles' for a,b in bc],
   [f'{DR("G")},">={a}",{DR("G")},"<={b}"' for a,b in bc])
er=[(0,0.15),(0.15,0.25),(0.25,0.35),(0.35,0.5),(0.5,1.01)]
nr=stat_block(ws_p,nr+1,'C. By efficiency ratio (low = choppy, high = trending)',[f'ER {a:.2f} to {min(b,1):.2f}' for a,b in er],
   [f'{DR("I")},">={a}",{DR("I")},"<{b}"' for a,b in er])
# gap buckets from tertiles (1982+ only)
g=d.loc[[t for y in YEARS if y>=1982 for t in d.loc[cyc[y][0]:cyc[y][1]].index[:-H]],'gap20'].dropna()
q1,q2=[round(x,2) for x in g.quantile([1/3,2/3])]
gb=[(0,q1),(q1,q2),(q2,99)]
nr=stat_block(ws_p,nr+1,f'D. By average |gap| (20d) - only 1982+ cycles, opens before ~1980 equal the prior close so gaps cannot be measured',[f'gap {a:.2f}% to {b:.2f}%' if b<99 else f'gap > {a:.2f}%' for a,b in gb],
   [f'{DR("J")},">={a}",{DR("J")},"<{b}"' for a,b in gb])
# combination
nr=stat_block(ws_p,nr+1,'E. Combination: lots of big candles AND tiny net progress vs the opposite',
   ['chop: >=6 big candles & ER<=0.25','calm trend: <=2 big candles & ER>=0.35'],
   [f'{DR("G")},">=6",{DR("I")},"<=0.25"',f'{DR("G")},"<=2",{DR("I")},">=0.35"'])
widths(ws_p,[46,2,10,10,14,14,16,14,16]); 

# =============== Chop_Cycles ===============
ws_c=sheet('Chop_Cycles')
put(ws_c,1,1,'Cycle-level view: does "bigger candles + no progress" mark the weak post-election cycles?',font=f_t)
put(ws_c,2,1,'Window = election day to Jul 15 of next year. Class thresholds (inputs):',font=f_norm)
put(ws_c,2,6,'BULL if ret >=',font=f_b); put(ws_c,2,7,15,'0',font=f_in,fill=fill_in)
put(ws_c,2,8,'WEAK if ret <=',font=f_b); put(ws_c,2,9,10,'0',font=f_in,fill=fill_in)
cols=['cycle','election day','ret % (election close -> Jul 15 close)','max drawdown % (lowest low vs election close)','class','avg range %','big-candle share %','avg big candles / 20d','avg efficiency ratio','avg |gap| 20d % (1982+)','chop-flag days %','chop-flag days % in first 40 days','ret % from day 40 -> Jul 15','max DD % after day 40']
hdr(ws_c,4,1,cols)
c0=5
for i,y in enumerate(YEARS):
    rr=c0+i; E,end=cyc[y]; seg=d.loc[E:end]
    a,b=drow[y]
    ret=(seg.close.iloc[-1]/seg.close.iloc[0]-1)*100; dd=(seg.low.min()/seg.close.iloc[0]-1)*100
    s40=seg.iloc[40]; r40=(seg.close.iloc[-1]/s40.close-1)*100; dd40=(seg.iloc[41:].low.min()/s40.close-1)*100
    put(ws_c,rr,1,y,font=f_b); put(ws_c,rr,2,E.to_pydatetime(),'yyyy-mm-dd'); put(ws_c,rr,3,round(ret,2),'0.00'); put(ws_c,rr,4,round(dd,2),'0.00')
    put(ws_c,rr,5,f'=IF(C{rr}>=$G$2,"BULL",IF(C{rr}<=$I$2,"WEAK","MID"))',font=f_b)
    put(ws_c,rr,6,f'=AVERAGEIFS({DR("E")},{DR("B")},A{rr})','0.00')
    put(ws_c,rr,7,f'=AVERAGEIFS({DR("F")},{DR("B")},A{rr})*100','0.0')
    put(ws_c,rr,8,f'=AVERAGEIFS({DR("G")},{DR("B")},A{rr})','0.0')
    put(ws_c,rr,9,f'=AVERAGEIFS({DR("I")},{DR("B")},A{rr})','0.00')
    put(ws_c,rr,10,f'=IFERROR(AVERAGEIFS({DR("J")},{DR("B")},A{rr}),"")','0.00')
    put(ws_c,rr,11,f'=SUMIFS({DR("K")},{DR("B")},A{rr})/COUNTIFS({DR("B")},A{rr})*100','0.0')
    put(ws_c,rr,12,f'=SUMIFS({DR("K")},{DR("B")},A{rr},{DR("C")},"<=40")/COUNTIFS({DR("B")},A{rr},{DR("C")},"<=40")*100','0.0')
    put(ws_c,rr,13,round(r40,2),'0.00'); put(ws_c,rr,14,round(dd40,2),'0.00')
c1=c0+len(YEARS)-1
R=lambda col: f'${col}${c0}:${col}${c1}'
rr=c1+2; sec(ws_c,rr,'Average by class',14)
hdr(ws_c,rr+1,1,['class','cycles','avg ret %','avg max DD %','','avg range %','big-candle share %','avg big candles / 20d','avg efficiency ratio','avg |gap| %','chop-flag days %','early (first 40d) chop-flag %','avg ret day40->Jul15 %','avg DD after day 40 %'])
for j,cl in enumerate(['BULL','MID','WEAK']):
    q=rr+2+j; put(ws_c,q,1,cl,font=f_b)
    put(ws_c,q,2,f'=COUNTIF({R("E")},A{q})','0')
    for col,src,fmt in [(3,'C','0.00'),(4,'D','0.00'),(6,'F','0.00'),(7,'G','0.0'),(8,'H','0.0'),(9,'I','0.00'),(10,'J','0.00'),(11,'K','0.0'),(12,'L','0.0'),(13,'M','0.00'),(14,'N','0.00')]:
        put(ws_c,q,col,f'=IFERROR(AVERAGEIFS({R(src)},{R("E")},$A{q}),"")',fmt)
cc=rr+6; sec(ws_c,cc,'Correlation across the 16 cycles (positive = more of the feature goes with a better outcome)',14)
hdr(ws_c,cc+1,1,['feature','vs ret % (election->Jul 15)','vs max DD % (less negative = shallower)','vs ret % day40->Jul15'])
for j,(lab,src) in enumerate([('avg range %','F'),('big-candle share %','G'),('avg big candles / 20d','H'),('avg efficiency ratio','I'),('chop-flag days %','K')]):
    q=cc+2+j; put(ws_c,q,1,lab,font=f_b)
    put(ws_c,q,2,f'=CORREL({R(src)},{R("C")})','0.00'); put(ws_c,q,3,f'=CORREL({R(src)},{R("D")})','0.00'); put(ws_c,q,4,f'=CORREL({R(src)},{R("M")})','0.00')
q=cc+7; put(ws_c,q,1,'EARLY chop-flag % (first 40 days)',font=f_b)
put(ws_c,q,2,f'=CORREL({R("L")},{R("C")})','0.00'); put(ws_c,q,3,f'=CORREL({R("L")},{R("D")})','0.00'); put(ws_c,q,4,f'=CORREL({R("L")},{R("M")})','0.00')
put(ws_c,q+1,1,'Only 16 cycles - treat correlations as direction, not proof (|r| below ~0.5 is noise at n=16).',font=f_norm)
widths(ws_c,[22,12,16,18,8,11,12,12,12,12,12,14,14,12]); ws_c.freeze_panes='B5'

# =============== LONG ===============
ws_lt=sheet('Long_Trades'); ws_ls=sheet('Long_Summary')
lh=['variant','entry rule','stop-loss rule','cycle','election day','entry date','entry px','SL px','exit date','exit px','exit reason','PnL pts (1 unit)','PnL %','bars held','MAE pts (worst dip vs entry)','MFE pts (best run vs entry)','2M pre-election low']
hdr(ws_lt,1,1,lh)
vid=0; variants=[]; r=2
for rule in LONG_ENTRIES:
    for b in LONG_SLBUF:
        vid+=1; vn=f'L{vid:02d}'; variants.append((vn,rule,sl_name(b)))
        for y in YEARS:
            E,end=cyc[y]; res=run_long(d,w,E,rule,b,end)
            if res is None or 'skip' in res: continue
            vals=[vn,rule,sl_name(b),y,E.to_pydatetime(),res['entry_date'].to_pydatetime(),round(res['entry_px'],2),None if res['sl'] is None else round(res['sl'],2),res['exit_date'].to_pydatetime(),round(res['exit_px'],2),res['why']]
            for c,v in enumerate(vals,1): put(ws_lt,r,c,v,'yyyy-mm-dd' if c in(5,6,9) else ('#,##0.00' if c in(7,8,10) else None))
            put(ws_lt,r,12,f'=J{r}-G{r}','#,##0.00;[Red]-#,##0.00'); put(ws_lt,r,13,f'=L{r}/G{r}*100','0.00;[Red]-0.00')
            put(ws_lt,r,14,res['bars']); put(ws_lt,r,15,round(res['mae'],2),'#,##0.00'); put(ws_lt,r,16,round(res['mfe'],2),'#,##0.00'); put(ws_lt,r,17,round(res['prelow'],2),'#,##0.00')
            r+=1
LL=r-1
widths(ws_lt,[9,14,14,7,12,12,10,10,12,10,12,12,9,9,14,14,12]); ws_lt.freeze_panes='A2'; ws_lt.auto_filter.ref=f'A1:Q{LL}'
LR=lambda col: f'Long_Trades!${col}$2:${col}${LL}'

def summary_table(ws,r0,variants,TR,ncyc,title):
    sec(ws,r0,title,13)
    hdr(ws,r0+1,1,['variant','entry rule','stop-loss rule','trades','no-trade cycles','wins','win %','total PnL pts','avg PnL pts','avg PnL %','worst trade pts','worst trade %','SL hits','profit factor','avg bars held'])
    for i,(vn,a,b) in enumerate(variants):
        q=r0+2+i
        put(ws,q,1,vn,font=f_b); put(ws,q,2,a); put(ws,q,3,b)
        put(ws,q,4,f'=COUNTIFS({TR("A")},A{q})','0'); put(ws,q,5,f'={ncyc}-D{q}','0')
        put(ws,q,6,f'=COUNTIFS({TR("A")},A{q},{TR("L")},">0")','0'); put(ws,q,7,f'=IFERROR(F{q}/D{q}*100,"")','0.0')
        put(ws,q,8,f'=SUMIFS({TR("L")},{TR("A")},A{q})','#,##0.00;[Red]-#,##0.00'); put(ws,q,9,f'=IFERROR(H{q}/D{q},"")','#,##0.00;[Red]-#,##0.00')
        put(ws,q,10,f'=IFERROR(AVERAGEIFS({TR("M")},{TR("A")},A{q}),"")','0.00;[Red]-0.00')
        put(ws,q,11,f'=_xlfn.MINIFS({TR("L")},{TR("A")},A{q})','#,##0.00;[Red]-#,##0.00'); put(ws,q,12,f'=_xlfn.MINIFS({TR("M")},{TR("A")},A{q})','0.00;[Red]-0.00')
        put(ws,q,13,f'=COUNTIFS({TR("A")},A{q},{TR("K")},"SL hit")','0')
        put(ws,q,14,f'=IFERROR(SUMIFS({TR("L")},{TR("A")},A{q},{TR("L")},">0")/-SUMIFS({TR("L")},{TR("A")},A{q},{TR("L")},"<0"),"no losers")','0.00')
        put(ws,q,15,f'=IFERROR(AVERAGEIFS({TR("N")},{TR("A")},A{q}),"")','0')
    return r0+2+len(variants)

put(ws_ls,1,1,'LONG-ONLY backtest: election day -> Jul 15 (SPX, 1 unit, PnL in index points; % column added because early-era points are tiny vs today)',font=f_t)
put(ws_ls,2,1,'Entries fill at the stated price; signals confirmed on a close fill at the NEXT open. Stop = lowest low in the 2 calendar months up to election day, minus the buffer. SL gap-through fills at the open. Signal entries must trigger within 10 weeks of the election.',font=f_norm)
put(ws_ls,3,1,'Trades per variant are capped at 16 (one per cycle, 1962-2022). 2026 is not in the data yet (ends Oct 27).',font=f_norm)
e=summary_table(ws_ls,5,variants,LR,16,'Variant table (all formulas read from Long_Trades)')
# matrix
m0=e+2; sec(ws_ls,m0,'Matrix: average PnL % per trade (rows = entry rule, columns = stop-loss rule)',13)
slnames=[sl_name(b) for b in LONG_SLBUF]
hdr(ws_ls,m0+1,1,['entry rule']+slnames)
for i,rule in enumerate(LONG_ENTRIES):
    q=m0+2+i; put(ws_ls,q,1,rule,font=f_b)
    for j,s in enumerate(slnames):
        put(ws_ls,q,2+j,f'=AVERAGEIFS({LR("M")},{LR("B")},$A{q},{LR("C")},{CL(2+j)}${m0+1})','0.00')
m1=m0+2+len(LONG_ENTRIES)+1; sec(ws_ls,m1,'Matrix: win % and SL hits (SL hits / trades)',13)
hdr(ws_ls,m1+1,1,['entry rule']+slnames)
for i,rule in enumerate(LONG_ENTRIES):
    q=m1+2+i; put(ws_ls,q,1,rule,font=f_b)
    for j,s in enumerate(slnames):
        put(ws_ls,q,2+j,f'=COUNTIFS({LR("B")},$A{q},{LR("C")},{CL(2+j)}${m1+1},{LR("K")},"SL hit")&" SL / "&COUNTIFS({LR("B")},$A{q},{LR("C")},{CL(2+j)}${m1+1})&" tr"')
widths(ws_ls,[16,15,16,9,10,8,8,13,12,11,13,12,9,11,11]); ws_ls.freeze_panes='A5'

# =============== SHORT ===============
ws_st=sheet('Short_Trades'); ws_ss=sheet('Short_Summary'); ws_sw=sheet('Short_Window_Stats')
sh=['variant','entry rule','exit','stop-loss rule','cycle','entry date','entry px','SL px','exit date','exit px','exit reason','PnL pts (short, 1 unit)','PnL %','bars held','MAE pts (worst move against)','MFE pts (best move for)']
hdr(ws_st,1,1,sh)
vid=0; svars=[]; r=2
for rule in SHORT_ENTRIES:
  for ex in SHORT_EXITS:
    for slr in SHORT_SL:
        vid+=1; vn=f'S{vid:02d}'; svars.append((vn,f'{rule} | {ex}',slr))
        for y in YEARS:
            res=run_short(d,w,y,rule,ex,slr)
            if res is None or 'skip' in res: continue
            vals=[vn,rule,ex,slr,y,res['entry_date'].to_pydatetime(),round(res['entry_px'],2),None if res['sl'] is None else round(res['sl'],2),res['exit_date'].to_pydatetime(),round(res['exit_px'],2),res['why']]
            for c,v in enumerate(vals,1): put(ws_st,r,c,v,'yyyy-mm-dd' if c in(6,9) else ('#,##0.00' if c in(7,8,10) else None))
            put(ws_st,r,12,f'=G{r}-J{r}','#,##0.00;[Red]-#,##0.00'); put(ws_st,r,13,f'=L{r}/G{r}*100','0.00;[Red]-0.00')
            put(ws_st,r,14,res['bars']); put(ws_st,r,15,round(res['mae'],2),'#,##0.00'); put(ws_st,r,16,round(res['mfe'],2),'#,##0.00')
            r+=1
SL_=r-1
widths(ws_st,[9,16,8,12,7,12,10,10,12,10,12,13,9,9,15,15]); ws_st.freeze_panes='A2'; ws_st.auto_filter.ref=f'A1:P{SL_}'
SR=lambda col: f'Short_Trades!${col}$2:${col}${SL_}'
# short trade columns differ: variant A, rule B, exit C, sl D, cycle E ... PnL L, % M, reason K, bars N
put(ws_ss,1,1,'SHORT backtest: after Jul 15 -> Aug 31 / Sep 30 (SPX, 1 unit, PnL in points)',font=f_t)
put(ws_ss,2,1,'Entry rules: S0 = sell the Jul 15 close; S_brk10 = first close below the prior 10-day low (fill next open); S_ma20 = first close below a falling 20d average (next open); S_W_brklow = first weekly close below the prior week low (next Monday open). Signal entries must trigger by Aug 15.',font=f_norm)
put(ws_ss,3,1,'Stops: fixed % above entry, or the highest high of the 30 calendar days before the signal. Gap-through fills at the open. 16 cycles, 1962-2022.',font=f_norm)
sec(ws_ss,5,'Variant table (all formulas read from Short_Trades)',13)
hdr(ws_ss,6,1,['variant','entry rule | exit','stop-loss rule','trades','no-trade cycles','wins','win %','total PnL pts','avg PnL pts','avg PnL %','worst trade pts','worst trade %','SL hits','profit factor','avg bars held'])
for i,(vn,a,b) in enumerate(svars):
    q=7+i; TR=SR
    put(ws_ss,q,1,vn,font=f_b); put(ws_ss,q,2,a); put(ws_ss,q,3,b)
    put(ws_ss,q,4,f'=COUNTIFS({TR("A")},A{q})','0'); put(ws_ss,q,5,f'=16-D{q}','0')
    put(ws_ss,q,6,f'=COUNTIFS({TR("A")},A{q},{TR("L")},">0")','0'); put(ws_ss,q,7,f'=IFERROR(F{q}/D{q}*100,"")','0.0')
    put(ws_ss,q,8,f'=SUMIFS({TR("L")},{TR("A")},A{q})','#,##0.00;[Red]-#,##0.00'); put(ws_ss,q,9,f'=IFERROR(H{q}/D{q},"")','#,##0.00;[Red]-#,##0.00')
    put(ws_ss,q,10,f'=IFERROR(AVERAGEIFS({TR("M")},{TR("A")},A{q}),"")','0.00;[Red]-0.00')
    put(ws_ss,q,11,f'=_xlfn.MINIFS({TR("L")},{TR("A")},A{q})','#,##0.00;[Red]-#,##0.00'); put(ws_ss,q,12,f'=_xlfn.MINIFS({TR("M")},{TR("A")},A{q})','0.00;[Red]-0.00')
    put(ws_ss,q,13,f'=COUNTIFS({TR("A")},A{q},{TR("K")},"SL hit")','0')
    put(ws_ss,q,14,f'=IFERROR(SUMIFS({TR("L")},{TR("A")},A{q},{TR("L")},">0")/-SUMIFS({TR("L")},{TR("A")},A{q},{TR("L")},"<0"),"no losers")','0.00')
    put(ws_ss,q,15,f'=IFERROR(AVERAGEIFS({TR("N")},{TR("A")},A{q}),"")','0')
e=7+len(svars)+1
for k,ex in enumerate(SHORT_EXITS):
    m0=e+k*(len(SHORT_ENTRIES)+4); sec(ws_ss,m0,f'Matrix: average PnL % per short trade, exit {ex}',13)
    hdr(ws_ss,m0+1,1,['entry rule']+SHORT_SL)
    for i,rule in enumerate(SHORT_ENTRIES):
        q=m0+2+i; put(ws_ss,q,1,rule,font=f_b)
        for j,s in enumerate(SHORT_SL):
            put(ws_ss,q,2+j,f'=IFERROR(AVERAGEIFS({SR("M")},{SR("B")},$A{q},{SR("C")},"{ex}",{SR("D")},{CL(2+j)}${m0+1}),"")','0.00;[Red]-0.00')
widths(ws_ss,[16,26,14,9,10,8,8,13,12,11,13,12,9,11,11]); ws_ss.freeze_panes='A7'

# Short_Window_Stats
put(ws_sw,1,1,'What the market does after Jul 15 (no strategy, just the window)',font=f_t)
hdr(ws_sw,3,1,['cycle','Jul 15 close','Aug 31 close','Sep 30 close','ret % to Aug 31','ret % to Sep 30','highest high % above Jul 15 close (before Sep 30)','lowest low % below Jul 15 close (before Sep 30)','short Jul15->Aug31 wins?','short Jul15->Sep30 wins?'])
for i,y in enumerate(YEARS):
    q=4+i; st=short_start(d,y); a8=exit_date(d,y,'Aug31'); a9=exit_date(d,y,'Sep30'); seg=d.loc[st:a9]
    put(ws_sw,q,1,y,font=f_b); put(ws_sw,q,2,round(d.close[st],2),'#,##0.00'); put(ws_sw,q,3,round(d.close[a8],2),'#,##0.00'); put(ws_sw,q,4,round(d.close[a9],2),'#,##0.00')
    put(ws_sw,q,5,f'=(C{q}/B{q}-1)*100','0.00;[Red]-0.00'); put(ws_sw,q,6,f'=(D{q}/B{q}-1)*100','0.00;[Red]-0.00')
    put(ws_sw,q,7,round((seg.high.max()/d.close[st]-1)*100,2),'0.00'); put(ws_sw,q,8,round((seg.low.min()/d.close[st]-1)*100,2),'0.00')
    put(ws_sw,q,9,f'=IF(E{q}<0,"yes","no")'); put(ws_sw,q,10,f'=IF(F{q}<0,"yes","no")')
q1=4; q2=4+len(YEARS)-1; q=q2+2
for lab,fx in [('average','AVERAGE'),('median','MEDIAN'),('worst (max)','MAX'),('best (min)','MIN')]:
    put(ws_sw,q,1,lab,font=f_b)
    for col in 'EFGH': put(ws_sw,q,'ABCDEFGH'.index(col)+1,f'={fx}({col}{q1}:{col}{q2})','0.00;[Red]-0.00')
    q+=1
put(ws_sw,q,1,'P(market lower) %',font=f_b); put(ws_sw,q,5,f'=COUNTIF(E{q1}:E{q2},"<0")/COUNT(E{q1}:E{q2})*100','0'); put(ws_sw,q,6,f'=COUNTIF(F{q1}:F{q2},"<0")/COUNT(F{q1}:F{q2})*100','0')
q+=2; sec(ws_sw,q,'Regime split (does the post-Jul-15 weakness only exist in recent cycles?)',10)
for lab,cr in [('cycles before 1998','"<1998"'),('cycles 1998 and later','">=1998"')]:
    q+=1; put(ws_sw,q,1,lab,font=f_b)
    put(ws_sw,q,5,f'=AVERAGEIFS(E{q1}:E{q2},A{q1}:A{q2},{cr})','0.00;[Red]-0.00'); put(ws_sw,q,6,f'=AVERAGEIFS(F{q1}:F{q2},A{q1}:A{q2},{cr})','0.00;[Red]-0.00')
    put(ws_sw,q,9,f'=COUNTIFS(A{q1}:A{q2},{cr},I{q1}:I{q2},"yes")&" of "&COUNTIFS(A{q1}:A{q2},{cr})'); put(ws_sw,q,10,f'=COUNTIFS(A{q1}:A{q2},{cr},J{q1}:J{q2},"yes")&" of "&COUNTIFS(A{q1}:A{q2},{cr})')
widths(ws_sw,[22,12,12,12,12,12,20,20,14,14]); ws_sw.freeze_panes='A4'

# =============== 2026_Setup ===============
ws_s=sheet('2026_Setup')
last=d.index[-1]; E26=pd.Timestamp(2026,11,3)
pre=d.loc[E26-pd.DateOffset(months=2):last]
put(ws_s,1,1,'2026 setup - provisional, built from data ending '+last.strftime('%Y-%m-%d')+' (Nov 3 election day is NOT in the files yet; re-run build.py on a fresh export)',font=f_t)
put(ws_s,3,1,'2-month look-back start',font=f_b); put(ws_s,3,2,(E26-pd.DateOffset(months=2)).to_pydatetime(),'yyyy-mm-dd')
put(ws_s,4,1,'2-month pre-election LOW (so far)',font=f_b); put(ws_s,4,2,round(pre.low.min(),2),'#,##0.00'); put(ws_s,4,3,'on '+pre.low.idxmin().strftime('%Y-%m-%d')+' - only valid if no lower low prints on Nov 3')
put(ws_s,5,1,'Last close in file',font=f_b); put(ws_s,5,2,round(d.close.iloc[-1],2),'#,##0.00')
put(ws_s,6,1,'Stop buffer %',font=f_b); put(ws_s,6,2,2,'0.0',font=f_in,fill=fill_in); put(ws_s,6,3,'<- change me')
put(ws_s,7,1,'Stop-loss price',font=f_b); put(ws_s,7,2,'=B4*(1-B6/100)','#,##0.00')
put(ws_s,8,1,'Distance from last close to stop %',font=f_b); put(ws_s,8,2,'=(B5/B7-1)*100','0.00')
put(ws_s,10,1,'Trigger levels to watch after the election',font=f_b)
put(ws_s,11,1,'Daily: close above prior-5-day high (D_brk5), now'); put(ws_s,11,2,round(d.high.iloc[-5:].max(),2),'#,##0.00')
put(ws_s,12,1,'Daily: close above prior-10-day high (D_brk10), now'); put(ws_s,12,2,round(d.high.iloc[-10:].max(),2),'#,##0.00')
put(ws_s,13,1,'Daily: 20-day average (D_ma20), now'); put(ws_s,13,2,round(d.close.iloc[-20:].mean(),2),'#,##0.00')
wl=w.iloc[-1]; put(ws_s,14,1,'Weekly: high of last completed week in file ('+w['last'].iloc[-1].strftime('%Y-%m-%d')+')'); put(ws_s,14,2,round(wl.high,2),'#,##0.00')
put(ws_s,15,1,'These levels move every day - recompute from fresh data on the day.',font=f_norm)
put(ws_s,17,1,'Chop reading on the last bar',font=f_b)
lx=d.iloc[-1]
put(ws_s,18,1,'big candles in last 20d'); put(ws_s,18,2,int(lx.bigcnt)); put(ws_s,19,1,'efficiency ratio 20d'); put(ws_s,19,2,round(lx.er,3),'0.00')
put(ws_s,20,1,'avg |gap| 20d %'); put(ws_s,20,2,round(lx.gap20,3),'0.00')
put(ws_s,21,1,'Chop flag (>=6 big & ER<=0.25)'); put(ws_s,21,2,'ON' if (lx.bigcnt>=6 and lx.er<=0.25) else 'OFF',font=f_b)
widths(ws_s,[58,14,70])

R=ws_r; put(R,1,1,'Midterm election study - choppy-market detector + long-only (election -> Jul 15) + short (Jul 15 -> Aug/Sep) backtests',font=f_t)
lines=[
('SHEETS',None),
('Chop_Probs','Stats & probabilities for the chop detector (inputs in yellow: min big candles, max efficiency ratio, pullback threshold). Everything recalculates.'),
('Chop_Cycles','Per-cycle chop features vs outcome, class averages (BULL/MID/WEAK) and correlations.'),
('Chop_Daily','Daily feature table for every post-election day (election -> Jul 15, last 20 days dropped so the forward return exists).'),
('Long_Summary / Long_Trades','Long-only backtest: 8 entry rules x 5 stop rules = 40 variants, 1 unit, PnL in points (+% column). Trade sheet is filterable by variant.'),
('Short_Window_Stats','Plain post-Jul-15 behaviour per cycle (no rules) + pre-1998 / 1998+ split.'),
('Short_Summary / Short_Trades','Short backtest: 4 entry rules x 2 exits (Aug 31, Sep 30) x 4 stop rules = 32 variants.'),
('2026_Setup','Provisional stop level and trigger levels from data ending Oct 27, 2026 (election day itself is not in the file).'),
('',None),
('HOW IT WAS BUILT',None),
('Data','SPX_1D.csv, daily OHLC from 1962. 16 midterm cycles: 1962...2022. Election day = Tuesday after the first Monday of November. 2026 is not in the sample.'),
('Big candle','Daily range (H-L)/prev close in the top 20% of the previous 250 days (relative, so 1960s and 2020s are comparable). Early-era opens equal the prior close until ~1980, so gap stats use 1982+ cycles only.'),
('Efficiency ratio','|close(t) - close(t-20)| / sum of |daily close changes| over 20 days. Low = a lot of movement, no net progress.'),
('Chop flag','>= 6 big candles in the last 20 days AND efficiency ratio <= 0.25 (editable). ON about 10% of days.'),
('Long entry rules','E0_close = election-day close; NextDay_open; D_brk5 / D_brk10 = first daily close above the prior 5 / 10-day high (fill next open); D_ma20 = close above a rising 20d average; W_brkhigh = weekly close above prior week high (fill Monday); W_10wMA = weekly close above a rising 10-week average; Wk8_Monday = Monday open 8 weeks after election week. Signal entries must fire within 10 weeks.'),
('Long exit / stop','Hold to the last trading day on or before Jul 15 (close). Stop = lowest low of the 2 calendar months up to election day, minus a buffer (0-3%). Intraday touch exits at the stop; a gap through fills at the open.'),
('Short rules','Sell Jul 15 close, or first close below the prior 10-day low / below a falling 20d average / weekly close below prior week low (fill next open; trigger by Aug 15). Cover at Aug 31 or Sep 30 close, or at the stop (fixed 3% / 5% above entry, or the 30-day high before the signal).'),
('Points vs %','PnL in points is dominated by recent cycles (SPX 4,500 vs 60). Judge variants by avg PnL %, win %, worst trade % - points are there because you asked for a single unit.'),
('',None),
('WHAT THE NUMBERS SAY',None),
('Chop - flag','Flag ON: next-20-day return averages +1.2% vs +1.9% OFF; up in 66% of cases vs 75%; a 3%+ pullback in 39% vs 29%. So it marks higher risk, it is not a sell signal - the market still rose on average inside flagged stretches.'),
('Chop - count vs size','Weak cycles (<=10%: 1978, 2002, 2014, 2018) had MORE big candles (3.6 per 20d vs 2.7 in BULL), lower efficiency (0.22 vs 0.28), bigger gaps (0.18% vs 0.09%) and deeper drawdowns (-8.2% vs -4.1%). But the average candle SIZE was not bigger (1.33% vs 1.44% range), so your observation holds as frequency/clustering of big candles, not size.'),
('Chop - limits','Damage is clearest at 9+ big candles in 20 days (3%+ pullback in 38-45% of cases vs ~25-30% at 0-8). 2002 (weak, -13.8% dip) never flagged, and 1986 (strong) flagged 29% of days - so it is a risk dial, not a filter. Early chop in the first 40 days correlates -0.37 with later drawdown depth: directional only, n=16.'),
('Long - base case','Buying the election-day close and holding to Jul 15 made money in all 16 cycles (avg +16.7%, worst +4.7%) but sat through dips of -14.8% (2018), -14.6% (1974), -13.8% (2002) from entry.'),
('Long - the stop','The 2-month-low stop (0% buffer) was hit in 1994 and 2018; both recovered afterwards, so the stop cost return (avg +14.3% vs +16.7%) while capping the worst trade near -5.5%. With a 2% buffer only 2018 stopped (-7.4%), avg +15.6%. The stop is insurance, not an edge: it never improved the average in this sample.'),
('Long - entry timing','Breakout / moving-average / weekly entries did not beat simply buying election day: avg 14.5-16.2% vs 16.7%, and they still got stopped in 2018. Monday of week 8 had no stop hits and avg +15.5% (matches your playbook) but the stop level is not what protects it - lateness is.'),
('Short - window','Jul 15 -> Aug 31: SPX lower in 7 of 16 (avg -0.9%, median +0.1%). -> Sep 30: lower in 9 of 16 (avg -1.8%, median -0.7%). Before 1998 only 2 of 9 cycles fell by Aug 31; from 1998 on, 5 of 7 did by Aug 31 (avg -3.9%) and all 7 by Sep 30 (avg -5.7%). See the regime split on Short_Window_Stats - recent-era effect or small-sample luck, 7 cycles cannot tell.'),
('Short - best rule','Selling the Jul 15 close with a Sep 30 cover averaged +1.7 to +1.8% per trade (no stop or 3% stop), win rate 50-56%, worst trade -6.8%. Aug 31 exits are near a coin flip. Treat the short as marginal compared with the long side.'),
('',None),
('CAVEATS',None),
('Sample','16 cycles. Every best-of-40 or best-of-32 table here is a fitted result; prefer rules that are good across the whole grid.'),
('Execution','Daily bars only. Stops assume fills at the stop price (or the open on a gap); no slippage, costs, or dividends. SPX index, not a tradable instrument.'),
]
r=3
for k,v in lines:
    if v is None:
        if k: sec(R,r,k,2)
    else:
        put(R,r,1,k,font=f_b); x=put(R,r,2,v); x.alignment=Alignment(wrap_text=True,vertical='top'); R.cell(r,1).alignment=Alignment(vertical='top')
        R.row_dimensions[r].height=max(15,15*(len(v)//115+1))
    r+=1
R.column_dimensions['A'].width=26; R.column_dimensions['B'].width=125

order=['README','Chop_Probs','Chop_Cycles','Chop_Daily','Long_Summary','Long_Trades','Short_Window_Stats','Short_Summary','Short_Trades','2026_Setup']
wb._sheets=[wb[n] for n in order]
wb.save(OUT)
import pickle; 
print('saved',DL,LL,SL_)
