"""What a 20 s slice of EEG-IO actually contains: annotated blinks, a corrupt
interval, and blinks that are in the signal but not in the label file."""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt, detrend, find_peaks

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CAND = (os.path.join(ROOT,"data","EEG-IO"), os.path.join(ROOT,"data","EEG-EyeBlinks","EEG-IO"))
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=None)
ap.add_argument("--subject", default="S00")
ap.add_argument("--t0", type=float, default=0.5)
ap.add_argument("--span", type=float, default=20.0)
ap.add_argument("--k", type=float, default=4.0, help="detector threshold, in robust sigma")
ap.add_argument("--out", default=os.path.join(HERE,"eegio_segment.png"))
args = ap.parse_args()
D = args.data_dir or next((d for d in CAND if os.path.isdir(d)), CAND[0])
if not os.path.isdir(D):
    sys.exit(f"No EEG-IO recordings in {D!r}. Searched: {', '.join(CAND)}")

FS = 250.0
SURF, INK, SEC, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
ANN, UNANN, CRIT = "#2a78d6", "#eb6834", "#d03b3b"

def load(s): return np.loadtxt(os.path.join(D,s+"_data.csv"),delimiter=";",skiprows=1,usecols=(0,1,2))
def labels(s):
    corr,bl,n=[],[],0
    for row in open(os.path.join(D,s+"_labels.csv")):
        r=[c.strip() for c in row.split(",")]
        if r[0]=="corrupt": n=int(r[1])
        elif n>0: corr.append((float(r[0]),float(r[1]))); n-=1
        elif r[0]=="blinks": pass
        elif len(r)>=2: bl.append((float(r[0]),int(r[1])))
    return corr,np.array(bl)
B,A = butter(2,[0.5,10],btype="band",fs=FS)
clean = lambda x: filtfilt(B,A,detrend(x))

def detect(x,k):
    sig = 1.4826*np.median(np.abs(x-np.median(x)))
    pk,_ = find_peaks(-x, height=k*sig, distance=int(0.25*FS), prominence=k*sig)
    return pk/FS, sig

def style(ax):
    ax.set_facecolor(SURF); ax.grid(True,color=GRID,lw=0.6); ax.set_axisbelow(True)
    for k,sp in ax.spines.items():
        sp.set_visible(k in ("left","bottom")); sp.set_color(BASE); sp.set_lw(1.0)
    ax.tick_params(colors=MUTED,labelsize=8.5,length=3)
    for lb in ax.get_xticklabels()+ax.get_yticklabels(): lb.set_color(SEC)

SUB=args.subject; a=load(SUB); corr,bl=labels(SUB)
t=a[:,0]; x1=clean(a[:,1]); x2=clean(a[:,2])
pt,_=detect(x1,args.k); ann=bl[:,0]
unann=np.array([v for v in pt if np.min(np.abs(ann-v))>0.30
                and not any(c0<=v<=c1 for c0,c1 in corr)])

fig=plt.figure(figsize=(13.2,8.4),facecolor=SURF)
gs=GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.26,left=0.07,right=0.985,top=0.885,bottom=0.075,
            height_ratios=[1.25,1.0])

# ---- A: the segment ----
axA=fig.add_subplot(gs[0,:]); style(axA)
w0,w1=args.t0,args.t0+args.span; m=(t>=w0)&(t<=w1)
axA.plot(t[m],x1[m],lw=1.2,color=INK,zorder=4)
for c0,c1 in corr:
    if c0<w1 and c1>w0:
        axA.axvspan(max(c0,w0),min(c1,w1),color=CRIT,alpha=0.20,lw=0,zorder=1)
        axA.annotate("corrupt interval\n(flagged in the label file)",
                     xy=((max(c0,w0)+min(c1,w1))/2,0.985),xycoords=("data","axes fraction"),
                     ha="center",va="top",fontsize=8,color=CRIT)
for v in ann:
    if w0<=v<=w1:
        axA.axvline(v,color=ANN,lw=1.3,zorder=2)
        axA.plot([v],[x1[int(v*FS)]],marker="v",ms=8,color=ANN,zorder=5,
                 mec=SURF,mew=1.2,clip_on=False)
for v in unann:
    if w0<=v<=w1:
        axA.axvline(v,color=UNANN,lw=1.3,ls=(0,(4,2)),zorder=2)
        axA.plot([v],[x1[int(v*FS)]],marker="o",ms=7,color=UNANN,zorder=5,mec=SURF,mew=1.2)
axA.set_xlim(w0,w1); axA.set_ylabel("Fp1  (µV)",fontsize=9.5,color=SEC)
ok=np.ones(m.sum(),bool); tw=t[m]
for c0,c1 in corr: ok[(tw>=c0)&(tw<=c1)]=False
lim=1.6*np.percentile(np.abs(x1[m][ok]),99.5)
axA.set_ylim(-lim,lim)
pkv=np.abs(x1[m][~ok]).max() if (~ok).any() else 0
if pkv>lim:
    axA.annotate(f"reaches {pkv:,.0f} µV — clipped",xy=(0.5*(max(corr[0][0],w0)+min(corr[0][1],w1)),-lim),
                 xytext=(0,6),textcoords="offset points",ha="center",va="bottom",fontsize=7.5,color=CRIT)
axA.set_xlabel("time  (s)",fontsize=9.5,color=SEC)
na=int(((ann>=w0)&(ann<=w1)).sum()); nu=int(((unann>=w0)&(unann<=w1)).sum())
axA.set_title(f"A   {SUB}, {w0:.0f}–{w1:.0f} s, detrended and band-passed 0.5–10 Hz.  "
              f"{na} blinks are annotated; {nu} more are in the signal and not in the label file.",
              fontsize=10.5,color=INK,loc="left",pad=10)
axA.legend(handles=[
    Line2D([],[],color=ANN,lw=1.6,marker="v",ms=7,mec=SURF,label="annotated blink"),
    Line2D([],[],color=UNANN,lw=1.6,ls=(0,(4,2)),marker="o",ms=6,mec=SURF,label="blink-like, NOT annotated"),
    Patch(facecolor=CRIT,alpha=0.20,label="corrupt interval")],
    frameon=False,fontsize=8.5,labelcolor=SEC,ncol=1,loc="lower left")

# ---- B: the two populations, shape ----
axB=fig.add_subplot(gs[1,0]); style(axB)
half=int(0.6*FS)
def stack(times,subjects):
    out=[]
    for s in subjects:
        aa=load(s); cc,bb=labels(s); xx=clean(aa[:,1])
        if times=="ann": tt=bb[:,0]
        else:
            pp,_=detect(xx,args.k); an=bb[:,0]
            tt=[v for v in pp if np.min(np.abs(an-v))>0.30 and not any(c0<=v<=c1 for c0,c1 in cc)]
        for v in tt:
            j=int(v*FS)
            if j-half>=0 and j+half<len(xx): out.append(xx[j-half:j+half])
    return np.array(out)
subs=[f"S{i:02d}" for i in range(20)]
SA=stack("ann",subs); SU=stack("un",subs)
tt=np.arange(-half,half)/FS
for S,c,lab in ((SA,ANN,f"annotated  (n={len(SA)})"),(SU,UNANN,f"not annotated  (n={len(SU)})")):
    md=np.median(S,axis=0)
    axB.fill_between(tt,np.percentile(S,25,axis=0),np.percentile(S,75,axis=0),color=c,alpha=0.18,lw=0)
    axB.plot(tt,md,lw=2.0,color=c,label=lab)
axB.axvline(0,color=MUTED,ls="--",lw=1.0)
axB.set_xlabel("time from event  (s)",fontsize=9,color=SEC); axB.set_ylabel("µV",fontsize=9,color=SEC)
axB.legend(frameon=False,fontsize=8.5,labelcolor=SEC,loc="lower right")
axB.set_title("B   Same waveform, either way",fontsize=10.5,color=INK,loc="left",pad=8)
axB.set_xlim(tt[0],tt[-1])

# ---- C: amplitude distributions ----
axC=fig.add_subplot(gs[1,1]); style(axC)
pa=np.ptp(SA,axis=1); pu=np.ptp(SU,axis=1)
rng=np.random.default_rng(0)
for i,(v,c,lab) in enumerate(((pa,ANN,"annotated"),(pu,UNANN,"not annotated"))):
    axC.scatter(np.full(len(v),i)+rng.uniform(-.13,.13,len(v)),v,s=11,color=c,alpha=0.45,lw=0)
    axC.plot([i-.28,i+.28],[np.median(v)]*2,color=c,lw=2.4,solid_capstyle="round")
    axC.annotate(f"median\n{np.median(v):.0f} µV",xy=(i,np.median(v)),xytext=(0,12),
                 textcoords="offset points",ha="center",fontsize=8,color=SEC)
axC.set_xticks([0,1]); axC.set_xticklabels(["annotated","not annotated"],fontsize=9)
axC.set_ylabel("peak-to-peak  (µV)",fontsize=9,color=SEC)
axC.set_title("C   Same size, either way",fontsize=10.5,color=INK,loc="left",pad=8)
axC.set_xlim(-.5,1.5)
top=1.15*np.percentile(np.concatenate([pa,pu]),99)
nhi=int((pa>top).sum()+(pu>top).sum())
if nhi:
    axC.set_ylim(0,top)
    axC.annotate(f"{nhi} event(s) above {top:,.0f} µV not shown",xy=(0.5,0.015),
                 xycoords="axes fraction",ha="center",fontsize=7.5,color=MUTED)

# ---- D: one unannotated event, both electrodes ----
axD=fig.add_subplot(gs[1,2]); style(axD)
def rho(v, x, y, w=0.3):
    j=int(v*FS); h=int(w*FS)
    if j-h<0 or j+h>=len(x): return np.nan
    return np.corrcoef(x[j-h:j+h],y[j-h:j+h])[0,1]
rs=np.array([rho(v,x1,x2) for v in unann]); rs_ok=rs[~np.isnan(rs)]
# the representative event: the one whose correlation is the median of this subject's
v=unann[int(np.nanargmin(np.abs(rs-np.nanmedian(rs))))]
r=rho(v,x1,x2)
mm=(t>=v-0.6)&(t<=v+0.6)
axD.plot(t[mm],x1[mm],lw=1.8,color=ANN,label="Fp1")
axD.plot(t[mm],x2[mm],lw=1.8,color=UNANN,label="Fp2")
axD.axvline(v,color=MUTED,ls="--",lw=1.0)
axD.set_xlabel("time  (s)",fontsize=9,color=SEC); axD.set_ylabel("µV",fontsize=9,color=SEC)
axD.legend(frameon=False,fontsize=8.5,labelcolor=SEC,loc="lower right",ncol=2)
axD.set_title(f"D   An unannotated event at {v:.1f} s.  Fp1/Fp2\ncorrelation over ±0.3 s: {r:.2f} — common-mode",
              fontsize=10.5,color=INK,loc="left",pad=8)

fig.suptitle("EEG-IO: the label file marks the stimulated blinks, not every blink",
             fontsize=12.5,color=INK,x=0.07,ha="left",y=0.962)
fig.savefig(args.out,dpi=170,facecolor=SURF,bbox_inches="tight")
print("wrote",args.out)
print(f"annotated n={len(SA)} median p-p {np.median(pa):.0f} uV | unannotated n={len(SU)} median p-p {np.median(pu):.0f} uV")
