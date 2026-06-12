# Coping in Crisis
## Computational Modeling of Coping Styles in Digital Crisis Discourse During the 2023 Türkiye Earthquake

**CSSM 530 — Automated Text Processing for Social Sciences | Spring 2026**
**Author:** Şevval Çakıcı | Koç University
**Domain Expert:** Assoc. Prof. Merih Angın | Koç University

---

## Research Question

How do different psychological coping styles appear in digital discourse during the February 6, 2023 earthquake in Türkiye, and how are they related to blame, help-seeking, and solidarity patterns across time?

---

## Overview

This study operationalizes Lazarus and Folkman's (1984) coping theory at scale in Turkish-language digital crisis discourse. Using over one million tweets posted in the 30 days following the February 6, 2023 earthquake in Türkiye, we train a multi-label BERTurk classifier to detect three coping styles — problem-focused, emotion-focused, and meaning-making — and track their temporal dynamics across four theoretically motivated crisis phases.

The earthquake unfolded in a deeply polarized political context, three months before a national election, with widespread public anger over construction negligence and delayed rescue operations. This makes the dataset an especially productive setting for studying the intersection of collective trauma and political meaning-making.

---

## Model Performance

| Metric | BERTurk | Zero-Shot (mDeBERTa) |
|--------|---------|----------------------|
| Macro F1 | 0.693 | 0.324 |
| Hamming Loss | 0.222 | 0.409 |
| Subset Accuracy | 0.427 | 0.147 |
| F1 Problem-Focused | 0.755 | 0.361 |
| F1 Emotion-Focused | 0.704 | 0.162 |
| F1 Meaning-Making | 0.619 | 0.450 |

BERTurk's advantage over the zero-shot baseline is statistically significant for problem-focused coping detection (McNemar p < .001).

---

## Key Findings

1. Problem-focused coping dominates Phase 1 (mean rate: 0.562) and declines sharply across phases — reflecting the exhaustion of rescue coordination needs
2. Emotion-focused coping rises from Phase 1 (0.320) and stabilizes in Phases 3–4 (0.54–0.57)
3. Meaning-making increases monotonically across all phases (0.12 → 0.24 → 0.35 → 0.37)
4. Anger correlates most strongly with meaning-making (Spearman r = 0.387) — suggesting it functions as a mobilizing force toward blame attribution rather than practical action
5. Demographic variation exists across coping styles: younger users show higher problem-focused rates, older users show higher emotion-focused and meaning-making rates; organizational accounts show markedly lower meaning-making than personal accounts (0.129 vs. 0.228)
6. Coping dynamics are not context-free: the political landscape of the 2023 earthquake shapes the particularly strong rise of meaning-making observed in this dataset

---

## Dataset

- **Source:** Politus corpus — Yörük, E., Hürriyetoğlu, A., Kına, M. F., Duruşan, F., Yardı, M. C., Atsızelti, Ş., et al. (2024). *Politus Dataset: A Political Public Opinion Dataset from Social Media in Turkey Processed with Privacy-Preserving AI.* figshare. https://doi.org/10.6084/m9.figshare.28027961.v3
- **Size:** 1,037,663 Turkish-language original tweets (Feb 6 – Mar 6, 2023)
- **Keywords:** #deprem, #6Şubat, #KahramanmaraşDepremi, #Kahramanmaraş, #Gaziantep, "deprem yardım", "enkaz", "AFAD", "earthquake Turkey"
- **Annotation:** 500 tweets; 50 labeled by 3 human annotators (majority vote gold standard), 450 labeled by GPT-4o (validated by high human-GPT agreement: 96%, 96%, 86%)

---

## Inter-Annotator Agreement (Cohen's κ)

| Category | Mean κ | Interpretation |
|----------|--------|----------------|
| Problem-Focused | 0.766 | Substantial |
| Emotion-Focused | 0.591 | Moderate–Substantial |
| Meaning-Making | 0.449 | Moderate |
| Avoidance | 0.021 | Poor — excluded |

---

## Repository Structure

| File | Description |
|------|-------------|
| `cssm530_coping_crisis.ipynb` | Full analysis pipeline (sampling → annotation → BERTurk → inference → visualization) |
| `demographic_analysis.py` | Demographic variation analysis (gender, age group, account type) |
| `requirements.txt` | Python dependency list |
| **data/** | |
| `data/cssm530_gold_standard.csv` | 500 annotated tweets with gold standard labels |
| `data/cssm530_kappa_results.csv` | Pairwise inter-annotator agreement results |
| `data/cssm530_model_comparison.csv` | BERTurk vs. zero-shot performance comparison |
| `data/cssm530_mcnemar.csv` | McNemar test results per label |
| `data/cssm530_test_results.csv` | Test set evaluation metrics |
| `data/cssm530_gpt_annotated.csv` | GPT-4o annotations for full 500-tweet sample |
| **figures/** | |
| `figures/fig1_temporal_coping.png` | Daily coping style distribution (Feb 6 – Mar 6) |
| `figures/fig2_phase_coping.png` | Mean coping rate by crisis phase |
| `figures/fig3_demographic_coping.png` | Coping styles by gender, age group, and account type |
| `figures/fig4_engagement.png` | Coping styles × engagement metrics (Spearman r) |
| `figures/fig5_emotion_heatmap.png` | Politus emotion scores × coping styles heatmap |

---

## How to Run

This project was developed in Google Colab with GPU support (Tesla T4).

**1. Clone the repository**
