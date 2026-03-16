# Research Brief: SustainHub-Concordia Evolution
**Target Audience:** AI Agent / Computational Social Scientist (Claude / BICA 2026 Reviewer)
**Context:** Multi-agent simulation of Open Source Software (OSS) sustainability using Concordia (DeepMind) and Active Inference.

---

## 1. Theoretical Foundation: Ostrom × Active Inference
The core theoretical gap is bridged by framing **Institutional Governance as a Shared Generative Model (SGM)**.

*   **Key Concept:** Ostrom’s 8 Design Principles function as **Bayesian Priors**. Cooperation emerges not from utility maximization, but from **Social Free Energy minimization**. Agents follow rules because they reduce "surprise" in social interactions.
*   **The Action Arena:** Uses David et al.'s (2021) AIC model to map Ostrom's IAD framework into a POMDP where agents minimize **Expected Free Energy (EFE)**.
*   **New Citations (BICA 2026 focus):**
    *   **Constant, A., et al. (2019). *Regime of Attention*.** Frames cities/communities as multi-scale generative models.
    *   **David, S., Cordes, R. J., & Friedman, D. A. (2021). *Active Inference in Modeling Conflict*.** Explicit mapping of IAD to Active Inference.
    *   **Anderies, J. M., et al. (2019).** On knowledge infrastructures as collective inference mechanisms.

---

## 2. Empirical Calibration: OSS Failure & Burnout
Calibration data based on GitHub 'Octoverse' and survival analysis of large-scale repositories.

*   **Project Failure (Inactivity):** ~16% of active projects become unmaintained annually. 50% fail within 4 years.
*   **Burnout Timelines:** 60% of maintainers have considered quitting. Average tenure follows a heavy-tailed distribution (50% are "one-and-done").
*   **Succession Patterns:** Only 41% of abandoned popular projects find a new lead. Internal promotion is the primary successful path.
*   **Bus Factor:** 65% of popular projects have a Bus Factor ≤ 2, significantly increasing failure risk.

---

## 3. Social Dilemma Parameters
*   **MPCR (Marginal Per Capita Return):** Set to **0.45** to maximize free-rider tension.
*   **Group Size:** **8-12 agents** provides the optimal balance of visibility and anonymity for social dilemma emergence.
*   **Graduated Sanctions:** 1:3 punishment ratio is recommended for stabilization.

---

## 4. Dual-Process Architecture (System 1 / System 2)
*   **System 1 (JaxMARL):** Habitual, fast execution.
*   **System 2 (Concordia):** Deliberative, slow reasoning triggered by **Critical Slowing Down (CSD)**. CSD is detected via spikes in temporal autocorrelation, signaling an impending phase transition (community collapse).

---
**Status:** Dashboard updated with Comparison View and Belief Radar for BICA demonstration.
