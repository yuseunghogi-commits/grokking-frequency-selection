# Related work and novelty assessment

This is an honest assessment of what in this repository is already known and what appears to be new, based on a
literature search done on 2026-10-19. Quotes were obtained through a summarising fetch tool and should be checked
against the papers before being cited verbatim. Search coverage is limited; this area produced many preprints in 2026.

**Kumar & Sankaranarayanan 2026 (arXiv:2610.22593), read in full text on 2026-10-19.** This paper asks whether
features can keep improving while training cross-entropy stays flat at a positive level. It proves finite-window
theorems for one-hidden-layer ReLU networks from constructed initialisations, using Polar/Muon-style updates, and shows
the same coexistence empirically in deeper networks and a one-block transformer. It does not ask which frequencies a run
selects, and it does not separate the effects of the initialisation and the split. The closest point of contact is
Appendix D.5: a train-only ridge readout restricted to the 8 frequencies with the largest *training energy*
E_k = (1/n) Σ_{x∈S} |(r̂(x) − r̄_S) e^{−iω_k y(x)}|² leads the network's held-out accuracy. That E_k is computed from
the *learned* features r̂ during training and is used to choose a readout. It is not a prediction of frequency
selection from the data alone before training. Our D(k) uses only the list of training pairs and is computed before any
training. The two quantities share the idea that the training set alone tells which frequencies matter, so the paper
should be cited as related, but it does not overlap with findings 1–3 below.

## Finding by finding

| Our finding | Verdict | Closest prior work |
|---|---|---|
| 1. The frequencies a run uses are fixed early (tens of steps), jointly and roughly additively by the initialisation **and the training subset** | Partially known. The init effect and early commitment are known; the systematic **split** effect and the additive init + data decomposition were not found | Nanda et al. 2023 (frequencies vary by seed; open question "why these frequencies"); He et al. 2026 (lottery ticket, full dataset); Ding et al. 2024 (early dynamics); Wang et al. 2026 (treat the finite subset as zero-mean noise); LeDoux 2026 (training *order*, not subset, decides the frequency in a very different regime) |
| 2. A closed-form, training-free predictor of the data side: D(k), the Fourier power of the shared-answer matrix N[a,a'] | Apparently new **as a stated and tested per-frequency predictor on finite splits**. The ingredients are known: shared answers = "parallelograms" (Liu et al. 2022); label Gram / empirical energy (Tian 2025, Li2); probability signatures and label collisions (Yao & Xu 2025; Wang et al. 2026). Kernel/NTK analyses on the full table are degenerate across k (Rubin et al. 2024) | Liu et al. 2022; Tian 2025; Yao & Xu 2025; Wang et al. 2026; Kunin et al. 2025 (a data spectrum orders frequencies, but for input encodings on the full table) |
| 3. Init side readable from the Fourier power of W_E·W_V·W_O at init; combined with D(k) it predicts the winners of fresh runs before training | Partially known. Quantitative init-only prediction exists for MLPs (Ding et al. 2024: Pearson 0.85 between initial embedding power and survival). The transformer OV-path readout and the combined init + data prediction were not found | Ding et al. 2024; He et al. 2026 |
| 4. Derivation ΔE_a ≈ η Σ_c N[a,c] r_c | **Known in form.** It is the leading term of the probability-signature dynamics (Yao & Xu 2025, Cor. 1; Wang et al. 2026, Thm 1) and of the lazy-stage gradient in Li2 (Tian 2025). Our E Eᵀ ≈ η²N and its Fourier power D(k) are a short corollary | Yao & Xu 2025; Wang et al. 2026; Tian 2025 |

Shift-invariance of D(k) under relabelling and its collapse under label shuffling follow from its definition; they are
sanity checks, not discoveries.

## Honest positioning

Prior work established that modular-addition networks use a sparse, seed-dependent set of Fourier frequencies, that the
initialisation predicts which frequencies survive (for MLPs, quantitatively), and that early embedding updates are driven by
label statistics of the training set. Those analyses either use the full table or treat the finite training subset as
zero-mean noise. This project shows that the subset is not neutral: with the initialisation held fixed, which pairs are in
the training set shifts the favoured frequencies systematically, early, and roughly additively with the init effect. A
one-line consequence of the known early-gradient form, the Fourier power D(k) of the shared-answer (parallelogram-count)
matrix, predicts this preference before training. The prediction holds in transformers and MLPs, for addition and for
multiplication (in the discrete-log basis), and for several primes. Combined with an init readout through the attention
OV path, it predicts the winning frequencies of fresh runs well above chance. All of this is for small models on modular
arithmetic.

## References

1. Nanda, Chan, Lieberum, Smith, Steinhardt. Progress measures for grokking via mechanistic interpretability. ICLR 2023. arXiv:2301.05217
2. Gromov. Grokking modular arithmetic. 2023. arXiv:2301.02679
3. Chughtai, Chan, Nanda. A Toy Model of Universality. ICML 2023. arXiv:2302.03025
4. Zhong, Liu, Tegmark, Andreas. The Clock and the Pizza. NeurIPS 2023. arXiv:2306.17844
5. Morwani, Edelman, Oncescu, Zhao, Kakade. Feature emergence via margin maximization. ICLR 2024. arXiv:2311.07568
6. Liu, Kitouni, Nolte, Michaud, Tegmark, Williams. Towards Understanding Grokking: An Effective Theory of Representation Learning. NeurIPS 2022. arXiv:2205.10343
7. Ding, Guo, Michaud, Liu, Tegmark. Survival of the Fittest Representation: A Case Study with Modular Addition. 2024. arXiv:2405.17420
8. He, Wang, Chen, Yang. On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking. 2026. arXiv:2602.16849
9. Wang, Yao, Liu, Lu, Xu. Probability-Signature Dynamics: Unpacking Modular Addition Learning Within Two-Layer Networks. 2026. arXiv:2610.11833
10. Yao, Xu. Probability Signature: Bridging Data Semantics and Embedding Structure in Language Models. 2025. arXiv:2509.20124
11. Tian. Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking (Li2). ICLR 2026. arXiv:2509.21519
12. Kunin et al. Alternating Gradient Flows. NeurIPS 2025. arXiv:2506.06489
13. Mohamadi, Li, Wu, Sutherland. Why Do You Grok? ICML 2024. arXiv:2407.12332
14. Kumar, Bordelon, Gershman, Pehlevan. Grokking as the Transition from Lazy to Rich Training Dynamics. ICLR 2024. arXiv:2310.06110
15. Rubin, Seroussi, Ringel. Grokking as a First Order Phase Transition in Two Layer Networks. ICLR 2024. arXiv:2310.03789
16. Mallinar et al. Emergence in non-neural models. ICML 2025. arXiv:2407.20199; Tomàs, Mallinar, Belkin. Breaking Data Symmetry is Needed For Generalization in Feature Learning Kernels. AISTATS 2026. arXiv:2604.00316
17. Doshi, Das, Wei, Gromov. To grok or not to grok. 2023. arXiv:2310.13061
18. LeDoux. The Order Is The Message. 2026. arXiv:2603.25047
19. Kumar, Sankaranarayanan. Flat Loss, Evolving Features: Grokking in Modular Addition. 2026. arXiv:2610.22593
