"""Render the five main-results charts from their frozen numerical data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family":"serif",
    "font.serif":["TeX Gyre Termes","Nimbus Roman","Liberation Serif","DejaVu Serif"],
    "font.size":9, "axes.labelsize":9, "legend.fontsize":7.5,
    "xtick.labelsize":8, "ytick.labelsize":8,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":.65,"xtick.direction":"in","ytick.direction":"in",
    "svg.fonttype":"none","pdf.fonttype":42,"savefig.bbox":"tight",
})
BLUE="#336BB0"; ADD="#882266"; PINK="#C27BAE"; GREEN="#228833"; GREY="#666666"; INK="#333333"
def save(fig,number):
    for ext in ("svg","png","pdf"):
        fig.savefig(OUT/f"fig{number}.{ext}", dpi=600, bbox_inches="tight",
                    facecolor="white",pad_inches=.03)
    plt.close(fig)
def tag(ax,letter):
    ax.set_title(f"({letter})",loc="left",pad=5,fontweight="bold")

# Figure 2: lower bound on comparisons needed to certify a longer horizon.
fig,(a,b)=plt.subplots(1,2,figsize=(5.5,2.45),gridspec_kw={"wspace":.32})
T=np.array([5.1,6,10,15,25],dtype=float)
data={
    -.2: [22.823,29.782,64.898,111.797,198.750],
    0: [33.085,45.792,127.200,286.199,794.999],
    .2: [52.216,79.280,353.333,1788.747,None],
}
colors={-.2:"#D98E04",0:"#B1460B",.2:"#5C1A1B"}
markers={-.2:"o",0:"s",.2:"^"}
for delta,vals in data.items():
    good=[(t,v) for t,v in zip(T,vals) if v is not None]
    t=np.array([x[0] for x in good])
    n=np.array([x[1] for x in good])
    a.plot(t,n,color=colors[delta],marker=markers[delta],lw=1.2,ms=4)
    b.scatter(5/t-delta,n,s=20,marker=markers[delta],color=colors[delta],zorder=4)
a.set_yscale("log");b.set_xscale("log");b.set_yscale("log")
a.set_xlabel(r"Required horizon $T$");a.set_ylabel("Necessary expected labels");a.set_xlim(4.4,26)
a.axvline(25,color=colors[.2],lw=.8,ls=":")
a.text(24.5,27,r"$d_T=0$",rotation=90,ha="right",fontsize=7.6,color=colors[.2])
a.annotate("112–1,789",xy=(15,112),xytext=(16,210),fontsize=7.5,
           arrowprops=dict(arrowstyle="-",lw=.7,color=".5"))
xx=np.logspace(-1,np.log10(1.4),200)
b.plot(xx,31.8/xx**2,color=INK,ls="--",lw=1)
b.set_xlabel(r"Distance to dynamic infeasibility $d_T$");b.set_xlim(.09,1.5)
for ax,letter in ((a,"a"),(b,"b")):
    tag(ax,letter);ax.grid(axis="y",alpha=.2,lw=.5)
handles=[Line2D([],[],color=colors[d],marker=markers[d],lw=1.1,
                label=rf"$\delta_\star={d:g}$") for d in (-.2,0,.2)]
handles.append(Line2D([],[],color=INK,lw=1,ls="--",label=r"$N_{\mathrm{lower}}=31.8d_T^{-2}$"))
fig.legend(handles=handles,loc="upper center",ncol=4,frameon=False,
           bbox_to_anchor=(.5,1.045),fontsize=7.5)
fig.subplots_adjust(top=.79,bottom=.22,left=.13,right=.985)
save(fig,2)

# Figure 3: maximum-speed reward trajectory and the feasible boundary.
time=np.array([0,.00198,.004005,.005985,.00783,.007875,.00801,.00999,.012015,.013995,.01602,.018])
phi=.8428043202+.04*time
lower=np.array([.841804320,.842137125,.842476848,.842808393,.843116774,.843124289,
                .843146831,.843477122,.843814279,.844143320,.844479201,.844806997])
gain=np.array([.0612592,.060939490,.060613586,.060295970,.060000938,.059993753,
               .059972202,.059656668,.059335023,.059021559,.058702025,.058390619])
best=np.array([.114730878,.114535824,.114336751,.114142505,.113961862,.113957460,
               .113944257,.113750815,.113553388,.113360748,.113164139,.112972297])
fig,(a,b)=plt.subplots(1,2,figsize=(5.5,2.6),gridspec_kw={"wspace":.35})
a.plot(time,phi,color=BLUE,lw=1.5)
a.plot(time,lower,color=INK,lw=1.25)
a.fill_between(time,phi,lower,where=lower>phi,color="#EBC4C4",alpha=.55)
a.scatter([.0078358743],[.8428043202+.04*.0078358743],
          s=21,facecolor="white",edgecolor=INK,zorder=5)
a.set_ylim(.84155,.84515);a.set_xlabel("Policy-gradient flow time")
a.set_ylabel(r"Reward coefficient $\phi$")
b.plot(time,gain,color=BLUE,lw=1.5)
b.plot(time,best,color=GREEN,lw=1.25)
b.axhline(.06,color=INK,ls="--",lw=.85)
b.set_ylim(.057,.117);b.set_yticks([.059,.060,.061,.113,.115])
b.set_xlabel("Policy-gradient flow time");b.set_ylabel("Progress rate")
for ax,letter in ((a,"a"),(b,"b")):
    tag(ax,letter);ax.set_xlim(0,.018)
    ax.axvline(.0078358743,color=".50",ls=":",lw=.9)
    ax.axvline(.01000602,color=".40",ls="-.",lw=.9)
fig.legend(handles=[
    Line2D([],[],color=BLUE,label="Deployed reward trajectory"),
    Line2D([],[],color=INK,label="Lower boundary of certifying interval"),
    Line2D([],[],color=GREEN,label="Pointwise optimum"),
    Line2D([],[],color=INK,ls="--",label="Required progress"),
    Line2D([],[],color=".5",ls=":",label="Numerical contact")],
    loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.085),fontsize=7.0)
fig.subplots_adjust(top=.77,bottom=.20,left=.12,right=.985)
save(fig,3)

# Figure 4: final RL policy behavior in Deep-Sea-Treasure.
fig,(a,b)=plt.subplots(1,2,figsize=(5.5,2.15),
                      gridspec_kw={"width_ratios":[1.35,1],"wspace":.28})
methods=[
    "Original preference\ninformation",
    "Uniformly selected\ncomparisons",
    "Informative\ncomparisons",
    "Fitted reward\nestimate",
    "Ordinary policy gradient,\ntrue reward",
]
returns=[.683069,.682654,.677708,.651416,.637899]
deeper=[45.6156,None,40.9901,12.6108,.8187]
colors=[BLUE,PINK,ADD,GREEN,GREY]
yy=np.arange(5)
a.barh(yy,returns,color=colors,height=.65)
a.set_yticks(yy,methods);a.invert_yaxis();a.set_xlim(0,.8)
a.set_xlabel("Final true return")
for k,v in enumerate(returns):
    a.text(v+.007,k,f"{v:.3f}",va="center",fontsize=7.1)
b.barh(yy,[v or 0 for v in deeper],color=colors,height=.65)
b.set_yticks(yy,[""]*5);b.invert_yaxis();b.set_xlim(0,56)
b.set_xlabel("Probability of reaching\na deeper treasure (%)")
for k,v in enumerate(deeper):
    b.text(.8 if v is None else v+1.1,k,
           "not measured" if v is None else f"{v:.1f}%",
           va="center",fontsize=7.0,
           color=".55" if v is None else INK)
for ax,letter in ((a,"a"),(b,"b")):
    tag(ax,letter);ax.grid(axis="x",lw=.5,alpha=.2);ax.set_axisbelow(True)
fig.subplots_adjust(left=.33,right=.985,top=.85,bottom=.24)
save(fig,4)

# Figure 5: human preference accuracy and NLL.
fig,(a,b)=plt.subplots(1,2,figsize=(5.5,2.5),gridspec_kw={"wspace":.36})
n=np.array([257,1024,4098])
accuracy=np.array([[.509548,.516583,.489447],
                  [.549749,.554774,.542714],
                  [.573869,.572864,.542714]])*100
nll=[.800293,.724063,.681543]
a.plot(n,accuracy.mean(axis=1),color=ADD,lw=1.5,marker="o",ms=4)
for k,x in enumerate(n):
    a.scatter([x]*3,accuracy[k],s=18,facecolors="none",edgecolors=ADD,lw=.8)
logistic=np.array([.586935,.571859,.590955])*100
a.scatter([4098]*3,logistic,s=20,c=GREY,marker="D")
a.scatter([4098],[logistic.mean()],s=30,facecolors="white",edgecolors=GREY,marker="D",zorder=4)
a.set_xticks(n,["257","1,024","4,098"]);a.set_ylim(48,60.5)
a.set_xlabel("Human preference comparisons")
a.set_ylabel("Held-out preference accuracy (%)")
b.plot(n,nll,color=ADD,lw=1.5,marker="o",ms=4)
b.set_xticks(n,["257","1,024","4,098"]);b.set_ylim(.66,.82)
b.set_xlabel("Human preference comparisons")
b.set_ylabel("Held-out negative log-likelihood")
for ax,letter in ((a,"a"),(b,"b")):
    tag(ax,letter);ax.grid(axis="y",lw=.5,alpha=.25)
fig.legend(handles=[
    Line2D([],[],color=ADD,marker="o",label="Reward model, mean"),
    Line2D([],[],color=ADD,marker="o",markerfacecolor="white",ls="none",label="Reward model, seed"),
    Line2D([],[],color=GREY,marker="D",ls="none",label="Logistic regression")],
    loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,1.075),fontsize=7.5)
fig.subplots_adjust(top=.78,bottom=.25,left=.11,right=.985)
save(fig,5)

# Figure 6: functional movement at each update and held-out summary quality.
fig,(a,b)=plt.subplots(1,2,figsize=(5.5,2.5),gridspec_kw={"wspace":.34})
events=[16,32,48]
mov0=[.049077,.048671,.049647]
mov1=[.048550,.048712,.049723]
a.scatter(events,mov0,s=22,color=ADD,marker="o",label="Seed 0")
a.scatter(events,mov1,s=22,color=ADD,marker="s",label="Seed 1")
a.axhline(.05,color=INK,ls="--",lw=.85)
a.text(32,.05006,r"$b=0.05$ per event",fontsize=7.3)
a.set_xlim(0,64);a.set_ylim(.047,.0504);a.set_xticks([0,16,32,48,64])
a.set_xlabel("Policy update");a.set_ylabel("Reward movement on\ncomparison set")
updates=[0,16,32,48,64]
fixed=np.array([[.160856,.165440,.162607,.163692,.166303],
                [.160856,.160513,.158101,.159437,.162076]])
updated=np.array([[.160856,.161599,.164789,.166390,.163466],
                  [.160856,.158476,.157765,.159105,.161469]])
for i in range(2):
    b.plot(updates,fixed[i],color=BLUE,alpha=.4,lw=.9)
    b.plot(updates,updated[i],color=ADD,alpha=.4,lw=.9)
b.plot(updates,fixed.mean(axis=0),color=BLUE,marker="o",lw=1.6,ms=3.6)
b.plot(updates,updated.mean(axis=0),color=ADD,marker="o",lw=1.6,ms=3.6)
for k in events:
    b.axvline(k,color=".8",ls=":",lw=.7)
b.set_xlim(0,64);b.set_ylim(.1555,.168)
b.set_xticks(updates);b.set_xlabel("Policy updates");b.set_ylabel("Held-out ROUGE-L")
for ax,letter in ((a,"a"),(b,"b")):tag(ax,letter)
fig.legend(handles=[
    Line2D([],[],color=BLUE,marker="o",label="Fixed reward (original preference data)"),
    Line2D([],[],color=ADD,marker="o",label="Updated reward (additional preference feedback)")],
    loc="upper center",bbox_to_anchor=(.5,1.07),ncol=2,frameon=False,fontsize=7.1)
fig.subplots_adjust(top=.76,bottom=.21,left=.13,right=.985)
save(fig,6)
print("Rendered five main result figures in",OUT)
