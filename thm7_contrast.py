import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 三次"死"指向一个结构性事实, 这个实验把它钉死:
#
# 诺特荷 C = W1 W1^T - W2^T W2 度量的是"两层之间的不平衡"(重缩放冗余方向),
#   这个方向 = 任务正交的"死方向"(我们框架的核心结论)。
# => 所以诺特荷的衰减/平衡, 本质上不可能编码任务信息。
#
# 任务信息在哪? 在"乘积" W2 W1 (真正决定预测的东西) 的奇异值谱里。
#   秩为 r 的任务 -> 乘积的有效秩 ≈ r。
#
# 本实验: 同一网络+优化器, 换不同秩任务, 并排比较两样东西:
#   (左) 诺特荷谱     -> 预期: 任务无关 (三次死的原因)
#   (右) 乘积奇异谱   -> 预期: 任务相关 (任务真正活的地方)
# =====================================================================

np.random.seed(0)
d_in=d_h=d_out=8; N=400; T=4000; dt=0.01; mu=0.3
W1_init=np.random.randn(d_h,d_in)*0.5
W2_init=np.random.randn(d_out,d_h)*0.5
X=np.random.randn(d_in,N)

def make_target(rank,seed=10):
    r=np.random.default_rng(seed)
    return (r.standard_normal((d_out,rank))@r.standard_normal((rank,d_in)))/np.sqrt(rank)

def train(M):
    W1,W2=W1_init.copy(),W2_init.copy(); Y=M@X
    for t in range(T):
        H=W1@X; P=W2@H; R=(P-Y)/N
        W2-=dt*(R@H.T + mu*W2)
        W1-=dt*((W2.T@R)@X.T + mu*W1)
    return W1,W2

ranks=[1,2,4,8]
noether_specs={}; product_specs={}
print("=== 诺特荷谱(任务无关?) vs 乘积奇异谱(任务相关?) ===")
for rk in ranks:
    M=make_target(rk); W1,W2=train(M)
    C=W1@W1.T - W2.T@W2
    noether=np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
    prod=np.linalg.svd(W2@W1, compute_uv=False)   # 乘积的奇异值
    noether_specs[rk]=noether; product_specs[rk]=prod
    # 乘积有效秩(奇异值>最大10%)
    eff_rank=np.sum(prod>0.1*prod[0])
    print(f"rank={rk}:  诺特荷谱范数={np.linalg.norm(noether):.4f}  "
          f"乘积有效秩={eff_rank}  乘积前4奇异值={np.round(prod[:4],2)}")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
colors=plt.cm.coolwarm(np.linspace(0,1,len(ranks)))
w=0.18; xi=np.arange(d_h)

for i,rk in enumerate(ranks):
    ax1.bar(xi+i*w, noether_specs[rk], w, color=colors[i], label=f'task rank={rk}')
ax1.set_title('LEFT: Noether charge spectrum\n-> IDENTICAL across tasks (lives in dead direction)')
ax1.set_xlabel('index'); ax1.set_ylabel('|Noether eigenvalue|'); ax1.legend(); ax1.grid(alpha=0.25)

for i,rk in enumerate(ranks):
    ax2.bar(xi+i*w, product_specs[rk], w, color=colors[i], label=f'task rank={rk}')
ax2.set_title('RIGHT: product W2W1 singular spectrum\n-> CHANGES with task (task lives here)')
ax2.set_xlabel('index'); ax2.set_ylabel('singular value'); ax2.legend(); ax2.grid(alpha=0.25)
plt.suptitle('Where does task information live? NOT in the Noether charge — in the product.',fontsize=13)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/thm7_where_is_task.png',dpi=130,bbox_inches='tight')
print("saved")
