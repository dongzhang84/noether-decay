"""
核查 review 第2条: ||C|| ~ e^{-2μt} 到底是哪个力给的?
  力A: 位置阻尼 -μw  (weight decay, 是保守力 ∇(μ/2|w|^2), 该并进V)
  力B: 速度阻尼 -μẇ  (momentum里的耗散, 真正的非保守摩擦)
分别跑, 看哪个给出 e^{-2μt}, 以及衰减率各是多少。
模型: 两层线性网, 诺特荷 C=W1W1^T-W2^T W2, 看 ||C|| 衰减。
"""
import numpy as np

np.random.seed(0)
d=6; N=300; dt=0.005; T=8000; mu=0.3
W1_0=np.random.randn(d,d)*0.5
W2_0=np.random.randn(d,d)*0.5
X=np.random.randn(d,N)
M=(np.random.randn(d,2)@np.random.randn(2,d))/np.sqrt(2)   # 固定任务
Y=M@X

def grads(W1,W2):
    H=W1@X; P=W2@H; R=(P-Y)/N
    return (W2.T@R)@X.T, R@H.T
def Cnorm(W1,W2): return np.linalg.norm(W1@W1.T-W2.T@W2)

# ---- 力A: 位置阻尼 -μw (weight decay), 纯一阶梯度下降(无惯性) ----
W1,W2=W1_0.copy(),W2_0.copy()
A_hist=[]
for t in range(T):
    if t%20==0: A_hist.append(Cnorm(W1,W2))
    g1,g2=grads(W1,W2)
    W1-=dt*(g1+mu*W1); W2-=dt*(g2+mu*W2)
A_hist=np.array(A_hist)

# ---- 力B: 速度阻尼 -μẇ (momentum衰减), 需要二阶动力学(带惯性m) ----
m=1.0
W1,W2=W1_0.copy(),W2_0.copy()
V1=np.zeros_like(W1); V2=np.zeros_like(W2)   # 速度
B_hist=[]
for t in range(T):
    if t%20==0: B_hist.append(Cnorm(W1,W2))
    g1,g2=grads(W1,W2)
    # m a = -∇Loss - μ v   (速度阻尼, 无位置项)
    A1=(-g1-mu*V1)/m; A2=(-g2-mu*V2)/m
    V1+=dt*A1; V2+=dt*A2
    W1+=dt*V1; W2+=dt*V2
B_hist=np.array(B_hist)

# ---- 拟合衰减率: 看 log||C|| 的斜率 ----
tt=np.arange(0,T,20)*dt
def fit_rate(h):
    # 取衰减段(前面下降部分), 拟合 log(h) ~ -rate * t
    mask=(h>1e-6)&(tt<tt[-1]*0.6)
    if mask.sum()<10: return np.nan
    p=np.polyfit(tt[mask], np.log(h[mask]), 1)
    return -p[0]

rate_A=fit_rate(A_hist)
rate_B=fit_rate(B_hist)
print("力A 位置阻尼 -μw (weight decay):")
print("   ||C||:  起=%.3f  末=%.4e   拟合衰减率=%.4f   (2μ=%.2f, μ=%.2f)"%(A_hist[0],A_hist[-1],rate_A,2*mu,mu))
print("力B 速度阻尼 -μẇ (momentum衰减):")
print("   ||C||:  起=%.3f  末=%.4e   拟合衰减率=%.4f   (2μ=%.2f, μ=%.2f)"%(B_hist[0],B_hist[-1],rate_B,2*mu,mu))
print()
print("判定:")
print("  力A衰减率/2μ = %.3f"%(rate_A/(2*mu)))
print("  力B衰减率/2μ = %.3f"%(rate_B/(2*mu) if not np.isnan(rate_B) else float('nan')))
