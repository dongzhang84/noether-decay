# noether-decay

**Learning as the decay of a Noether charge — a dissipative-systems view of training.**

A theory draft that treats the training of a deep network as a *dissipative dynamical system* and describes it with a **generalized Noether theorem** (a balance law, rather than a conservation law).

## Core idea

In an overparameterized network, rescaling symmetries (e.g. $W_1 \to G W_1,\ W_2 \to W_2 G^{-1}$) give rise to conserved Noether charges such as $C = W_1 W_1^\top - W_2^\top W_2$. The central claim:

> **Learning is the forced breaking of rescaling symmetry.** A direction in which the Noether charge stays conserved is exactly a direction in which the network learns *nothing*.

So at any instant the parameter space splits in two: the **conserved subspace** (the blind spots, where nothing is being learned) and the **decaying subspace** (where learning actually happens). What the charge decay measures is *how* the network organizes itself internally — a quantity that is orthogonal to *what* task it ends up learning.

## What's inside

- **Mechanism.** The decay is driven by weight decay, a *conservative* regularizer that explicitly breaks the rescaling symmetry, not by a non-conservative friction force. (This was checked numerically: velocity friction alone does not make the charge decay.)
- **Statements.** A chain of propositions — some rigorous, some only sketched, some still aspirational — on how the charge decays, why training has no closed-form solution, and how dissipation contracts phase space onto a low-dimensional attractor.
- **Two candidate invariants** meant to survive from the idealized regime all the way to real SGD: a **symmetry-breaking number** (an integer) and an **attractor dimension**.
- **One toy model.** Everything is demonstrated on a single two-layer linear network.

The writeup is deliberate about labeling what is proven, what is illustrative, and what is still a conjecture.

## Full text

The Chinese writeup lives in [`docs/`](docs/):

- `诺特衰减学习_Discussion.md` — the authoritative, complete record (**source**).
- `学习作为诺特荷的衰减——一个耗散系统的视角.md` — a refined public version (**projection** of the source).

## Status

Early draft. The results hold in the idealized regime (isotropic, constant regularization); generalization to realistic SGD is a direction, not yet a result.
