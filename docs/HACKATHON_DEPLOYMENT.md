# FTC Hackathon Deployment Guide: March 14, 2026

## 1. Local Dashboard Setup
Run the dashboard locally to verify the layout:
```bash
streamlit run hackathon_dashboard.py
```

## 2. Remote Hosting (Streamlit Community Cloud)
1.  **Push Code:** Commit `hackathon_dashboard.py` and `requirements.txt` to your GitHub.
2.  **Deploy:** Link your repo to [share.streamlit.io](https://share.streamlit.io).
3.  **Secrets:** If using Gemini directly, add `GEMINI_API_KEY` to the Streamlit Cloud secrets manager.

## 3. Real-Time Data Refresh (The Gist Method)
Since the simulation runs on your laptop/DGX, use a GitHub Gist as a "Live Bridge":
1.  **Create a Gist:** Upload a dummy `live_results.json`.
2.  **Simulation Side:** Update the Gist using a `requests.patch` call at the end of every sprint in your simulation.
3.  **Dashboard Side:** Set the Data Source to **"Remote URL"** and paste the raw Gist URL.
4.  **Refresh:** The dashboard will poll the Gist every few seconds and update the HI/RQ charts live for the judges.

## 4. Demo Talking Points
*   **Active Inference:** Explain that agents aren't just "maximizing reward" but "minimizing surprise" about community health.
*   **Ostrom Priors:** Mention that the agents use Ostrom's 8 Principles as Bayesian Priors for coordination.
*   **Resilience Quotient:** Show the RQ bounce back after you trigger a "Maintainer Dropout" live.
