# Data Fest: Sequence-Based Segmentation of Chronic Disease Care Pathways

**Competition:** Data Fest<br>
**Award:** Best Insights Award<br>
**Team:** Team 07V, CUEB<br>
**Members:** Lei Duli, Liu Haoyang, Li Haonan, Guo Jiawei, Chen Chaoran<br>
**Role:** Data Analysis / Pipeline Implementation<br>
**Tools:** Python, pandas, NumPy, scikit-learn, K-Means, PCA, matplotlib, seaborn

## Project Overview

This project investigates whether chronic disease patients with diabetes and hypertension follow distinct longitudinal care pathways. We focused on identifying whether some patients silently disengage from routine care before returning through emergency or inpatient services.

## Data

* 46,714 chronic disease patients
* 3,073,410 unique encounters
* Time range: 2022-2025
* Source: Stormont Vail Health integrated encounter master table
* Key preprocessing step: EncounterKey-level deduplication to avoid inflated visit frequencies and distorted gap metrics

## Methodology

1. Sequence construction

   * routine visit = 1
   * specialist visit = 2
   * ED/inpatient acute event = 3
   * silence marker = 0 when the gap between visits exceeded 90 days

2. Feature engineering

   * total encounters
   * acute-care ratio
   * mean gap
   * gap variability
   * maximum silence duration
   * silence-to-acute transition probability

3. Clustering

   * standardized features
   * K-Means clustering
   * four-cluster solution selected using the elbow method and clinical interpretability

4. Cluster profiling

   * compared mortality, MyChart activation, SDOH risk indicators, and acute-care patterns across clusters

## Key Results

| Result                                    |               Value |
| ----------------------------------------- | ------------------: |
| Award                                     | Best Insights Award |
| Patients analyzed                         |              46,714 |
| Unique encounters                         |           3,073,410 |
| Number of clusters                        |                   4 |
| Acute-unstable cluster size               |                4.7% |
| ED/inpatient ratio in Cluster 2           |               12.5% |
| Mortality rate in Cluster 2               |                7.3% |
| MyChart activation in Cluster 2           |               71.7% |
| Silence-to-acute probability in Cluster 2 |                 59% |

## Best Insight

The highest-risk patients were not simply those with the most visits. Instead, risk concentrated among patients whose care trajectories became unstable after long silent periods. In Cluster 2, nearly six out of ten post-silence transitions ended in ED or inpatient care, suggesting that silence in the clinical record is not a benign gap but a predictive intervention window.

## Implications

The results support a stratified chronic disease management strategy. High-touch patients may require sustained coordination to reduce care burden. Low-engagement patients may benefit from targeted outreach. The acute-unstable cluster should be prioritized for proactive monitoring during silent periods, digital engagement support, and targeted intervention.

## Files

* Project Paper: `files/datafest-team07v-paper.docx`
* Analysis Pipeline: `files/analysis_pipeline.py`
