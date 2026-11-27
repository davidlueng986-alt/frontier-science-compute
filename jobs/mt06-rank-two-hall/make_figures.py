"""Publication figures from derived formulas, with independent interface checks."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from multiway_audit import multiway_interface,extract_vector
from verify_hall import cross

plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
def F(r,p):return 4*p*(r+2-(r-1)*p)/(r+1-(r-1)*p)**2
def attained(r,phi):
    p=(1+phi)/2; eps=2*phi/(1+phi);R=np.zeros((3,3))
    for j in range(3):
        B=np.zeros(3,dtype=complex);B[j]=1e-8j
        child=multiway_interface([np.eye(3),r*np.eye(3)+cross(B)],[1-p,p],[1,0,0])
        K=multiway_interface([np.eye(3),child],[1-eps,eps],[0,0,1])
        R[:,j]=extract_vector(K.imag/1e-8)
    return np.trace(R)

def main():
    out=Path('figures');out.mkdir(exist_ok=True);res=Path('results');res.mkdir(exist_ok=True)
    phi=np.linspace(0,1,501);fig,ax=plt.subplots(figsize=(6.3,3.8))
    colors=['#245C73','#C17C2C','#8B4464'];rows=[]
    for r,color in zip([4,16,100],colors):
        ax.plot(phi,F(r,phi),color=color,lw=2,label=f'$r={r}$: exact minimum')
        points=np.linspace(.05,.95,10);values=[attained(r,p) for p in points]
        ax.plot(points,values,'o',ms=3.2,mfc='white',mec=color,mew=1)
        for p,v in zip(points,values):
            pred=F(r,p);assert abs(v-pred)<1e-9
            rows.append(dict(contrast=r,high_phase_fraction=float(p),interface_trace=float(v),analytic_minimum=float(pred),absolute_error=float(abs(v-pred))))
    ax.plot(phi,3*phi,'--',lw=1,color='.45',label='$r=1$')
    ax.set(xlabel='Hall-active high-phase volume fraction $\\phi$',ylabel='Normalized Hall trace $\\mathrm{tr}\\,R$',xlim=(0,1),ylim=(0,3.05))
    ax.legend(loc='upper left',frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(out/'sharp_envelope.pdf');fig.savefig(out/'sharp_envelope.png',dpi=180);plt.close(fig)
    with (res/'attainment.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    fig,axs=plt.subplots(1,2,figsize=(6.3,2.5))
    for ax in axs:ax.set(xlim=(0,1),ylim=(0,1));ax.set_aspect('equal');ax.axis('off')
    # Orthogonal rank-one and rank-two geometry: illustrative fractions only.
    p=.7;eps=4/7
    for i in range(8):
        axs[0].add_patch(Rectangle((i/8,0),1/8,1,facecolor='#E5EDF0',edgecolor='none'))
        axs[0].add_patch(Rectangle((i/8,0),p/8,1,facecolor='#245C73',edgecolor='none'))
    axs[0].add_patch(Rectangle((0,0),1,1,fill=False,lw=.7))
    for j in range(4):
        axs[1].add_patch(Rectangle((0,j/4),1,1/4,facecolor='#E5EDF0',edgecolor='none'))
        for i in range(20):
            axs[1].add_patch(Rectangle((i/20,j/4),p/20,eps/4,facecolor='#245C73',edgecolor='none'))
    axs[1].add_patch(Rectangle((0,0),1,1,fill=False,lw=.7))
    axs[0].set_title('Inner rank-one laminate\n$u=e_1$, active fraction $p$',fontsize=10)
    axs[1].set_title('Outer laminate\n$n=e_3$, child fraction $\\varepsilon$',fontsize=10)
    fig.text(.5,.01,'Dark: Hall-active phase     Light: inactive phase     Ideal separated scales',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'attaining_structure.pdf');fig.savefig(out/'attaining_structure.png',dpi=180);plt.close(fig)
    print('PASS: 30 independent interface checks on the attaining family; generated two figures')

if __name__=='__main__':main()
