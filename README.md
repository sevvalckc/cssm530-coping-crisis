# Coping in Crisis
## Computational Modeling of Coping Styles in Digital Crisis Discourse

**CSSM 530 — Spring 2026 | Şevval Çakıcı | Koç University**  
**Domain Expert:** Assoc. Prof. Merih Angın

---

## Research Question
How do different psychological coping styles appear in digital discourse
during the February 6, 2023 earthquake in Türkiye, and how are they related
to blame, help-seeking, and solidarity patterns across time?

---

## Dataset
- **Source:** Politus corpus (Yörük et al.)
- **Size:** 1,037,663 Turkish tweets (Feb 6 – Mar 6, 2023)
- **Keywords:** #deprem, #6Şubat, #KahramanmaraşDepremi + variants
- **Annotation:** 500 tweets labeled by 3 human annotators + GPT-4o

---

## Model Performance

| Metric | BERTurk | Zero-Shot (mDeBERTa) |
|--------|---------|----------------------|
| Macro F1 | 0.693 | 0.324 |
| F1 Problem-Focused | 0.755 | 0.361 |
| F1 Emotion-Focused | 0.704 | 0.162 |
| F1 Meaning-Making | 0.619 | 0.450 |

---

## Key Findings
1. Problem-focused coping dominant in Phase 1 (0.562), declines sharply thereafter
2. Emotion-focused coping rises and stabilizes from Phase 2 onward (0.54–0.57)
3. Meaning-making shows monotonic increase across all phases (0.12 → 0.37)
4. Anger strongly correlates with meaning-making (r = 0.387, p < .05)
5. BERTurk significantly outperforms zero-shot baseline for problem-focused detection (McNemar p < .001)

---

## Repository Structure

| File | Description |
|------|-------------|
| `cssm530_coping_crisis.ipynb` | Full analysis pipeline |
| `cssm530_gold_standard.csv` | 500 annotated tweets (gold labels) |
| `cssm530_kappa_results.csv` | Inter-annotator agreement (Cohen's κ) |
| `cssm530_model_comparison.csv` | BERTurk vs. zero-shot results |
| `cssm530_mcnemar.csv` | McNemar test results |
| `cssm530_test_results.csv` | Test set evaluation metrics |
| `fig1_temporal_coping.png` | Daily coping style distribution |
| `fig2_phase_coping.png` | Phase-level comparison |
| `fig3_engagement.png` | Engagement correlations |
| `fig4_emotion_heatmap.png` | Emotion × coping heatmap |

---

## Requirements
- transformers==4.40.0
- datasets==2.19.0
- torch==2.3.0
- scikit-learn==1.4.2
- pandas==2.2.2
- numpy==1.26.4
- scipy==1.13.0
- seaborn==0.13.2
- matplotlib==3.8.4
- openai==1.30.0
- statsmodels==0.14.2

---

## References
- Lazarus, R. S., & Folkman, S. (1984). *Stress, appraisal, and coping.* Springer.
- Schweter, S. (2020). BERTurk — BERT models for Turkish. Zenodo.
- Jin, Y. (2009). The effects of public's cognitive appraisal of emotions in crises. *Public Relations Review.*
