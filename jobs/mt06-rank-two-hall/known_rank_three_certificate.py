"""Exact re-verification of Kern's published rank-three example, not a new design."""
import json
from pathlib import Path
from fractions import Fraction as F
from exact_verify import eye,zeros,scale,laminate_dual,assert_physical
from multiway_audit import cross_exact,extract_vector

def run():
    spec=json.loads(Path(__file__).with_name('baseline_rational.json').read_text())
    r=F(str(spec['contrast'])); R=zeros(3,3)
    for j,field in enumerate(eye(3)):
        vals=[]
        leaves={-1:(eye(3),zeros(3,3)),-2:(scale(r,eye(3)),cross_exact(field))}
        for node in spec['nodes']:
            l,h=node['left'],node['right']
            x=leaves[l] if l<0 else vals[l];y=leaves[h] if h<0 else vals[h]
            v=laminate_dual(x,y,F(str(node['f'])),[F(str(z)) for z in node['normal']])
            assert_physical(*v);vals.append(v)
        col=extract_vector(vals[-1][1])
        for i in range(3):R[i][j]=col[i]
    trace=sum(R[i][i] for i in range(3))
    assert trace<0
    return dict(provenance='Kern, Phys. Rev. B 108, 235303 (2023), Eq.21, rounded rational normals',
                source='https://arxiv.org/html/2310.08404v2',
                new_construction=False,contrast=str(r),
                normalized_trace_exact=str(trace),normalized_trace=float(trace),
                negative_trace_certified=True,method='exact rational interface differentiation')

if __name__=='__main__':
    out=run();Path('results').mkdir(exist_ok=True)
    Path('results/known_rank_three.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
