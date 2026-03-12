# Research Brief: SustainHub-Concordia Evolution
**Target Audience:** AI Agent / Computational Social Scientist (Claude)
**Context:** Multi-agent simulation of Open Source Software (OSS) sustainability using Concordia (DeepMind) and Active Inference.

---

## 1. Theoretical Foundation: Ostrom × Active Inference
The core theoretical gap is bridged by framing **Institutional Governance as a Shared Generative Model (SGM)**.

*   **Key Concept:** Ostrom’s 8 Design Principles function as **Bayesian Priors**. Cooperation emerges not from utility maximization, but from **Social Free Energy minimization**. Agents follow rules because they reduce "surprise" in social interactions.
*   **The Action Arena:** Uses David et al.'s (2021) AIC model to map Ostrom's IAD framework into a POMDP where agents minimize **Expected Free Energy (EFE)**.
*   **Key Citation:** David, S., Cordes, R. J., & Friedman, D. A. (2021). *Active Inference in Modeling Conflict*.

---

## 2. Empirical Calibration: The OSS "Funnel"
Simulations must move away from flat probability distributions to reflect the "Funnel" of real-world OSS (Linux, npm, Apache data).

*   **Churn Benchmarks:**
    *   Newcomer Churn: ~60% in the first month.
    *   Annual Core Turnover: 30-70%.
    *   Long-term Retention: Only 2-12% of newcomers stay >1 year.
*   **Success Rate Benchmarks:**
    *   Apprentice: 25% success.
    *   Regular: 75% success.
    *   Expert/Core: 95% success.
*   **Role Distribution (1-9-90 Rule):** 1% Core Maintainers, 9% Occasional Contributors, 90% Users/Lurkers.

---

## 3. Social Dilemma Parameters
To avoid trivial equilibria (constant cooperation or constant defection), the following parameters are recommended based on experimental economics (Fehr, Ostrom, Nowak):

*   **MPCR (Marginal Per Capita Return):** Set to **0.45**.
    *   *Logic:* Above 0.7 makes cooperation trivial; below 0.3 makes defection inevitable. 0.45 ensures a steady decay of cooperation that requires active governance to sustain.
*   **Group Size:** **8 agents** is the empirical "sweet spot" for maximizing free-rider tension.
*   **Graduated Sanctions:** Implementing a **1:3 punishment ratio** (1 cost to punisher : 3 penalty to target) is the most effective mechanism for stabilizing social dilemmas.

---

## 4. Multi-Agent Active Inference (MAAI)
Social coordination is modeled as **reciprocal belief updating**.

*   **Precision Weighting:** Acts as the "Social Volume Control." Agents with high uncertainty about their own goals (low prior precision) are more easily influenced by the "Social Prior" of the group.
*   **Shared Narratives:** Friston's concept of coordination as **Generalized Synchrony** via coupled Markov Blankets.
*   **Epistemic Foraging:** Agents prioritize actions that reduce uncertainty about the codebase or teammate reliability before pursuing pragmatic rewards.

---

## 5. Dual-Process Architecture (System 1 / System 2)
A multi-fidelity approach to solve the LLM-latency bottleneck.

*   **System 1 (Reactive):** Hardware-accelerated (JaxMARL) executing Habitual/FSM-based policies at $10^4$ FPS.
*   **System 2 (Deliberative):** LLM-based (Concordia) executing complex social reasoning at $10^{-1}$ FPS.
*   **Phase Transition Detection:** Monitor for **Critical Slowing Down (CSD)**. A spike in temporal autocorrelation signals that the System 1 policy is breaking down, triggering a System 2 "Community Retrospective" to re-deliberate the community social contract.
*   **Key Citation:** Moskovitz et al. (2024). *Understanding dual process cognition via the minimum description length principle*.

---
**Next Implementation Goal:** Integrate variable success rates and the 0.45 MPCR target into the SustainHubPayoff engine.
