# Reliability-Aware Knowledge-Guided Sentiment Classification

This repository contains the complete Jupyter Notebook implementation for the study:

**A Reliability-Aware Knowledge-Guided Decision Framework for Imbalanced Social Media Sentiment Classification**

The project evaluates lexicon-based methods, classical machine-learning models, recurrent neural networks, transformer models, imbalance-handling strategies, reliability analysis, explainability, robustness, and multi-criteria model selection on an imbalanced Reddit sentiment dataset.

## Repository Overview

The implementation is provided as a single end-to-end notebook:

```text
Sentiment Analysis for Journal Paper.ipynb
```

The notebook performs:

- Dataset loading, cleaning, auditing, and stratified splitting
- Exact-duplicate removal and near-duplicate leakage analysis
- Lexicon-based sentiment classification
- TF-IDF-based classical machine-learning experiments
- LSTM, BiLSTM, and BiLSTM-Attention experiments
- DistilBERT, BERT-base, RoBERTa-base, and sentiment-specialized RoBERTa fine-tuning
- Random oversampling, class weighting, focal loss, and class-balanced focal loss
- Knowledge-prior and reliability-reweighting ablations
- The proposed LGR-CBFL framework
- Validation-selected soft-voting transformer ensembles
- Five-seed replication of the central transformer comparison
- Bootstrap confidence intervals and paired statistical tests
- Probability calibration and selective-prediction analysis
- Robustness, subgroup, disagreement, and error analysis
- Token-occlusion explanations
- Counterfactual identity-term sensitivity analysis
- TOPSIS-based deployment recommendations
- Publication-ready tables, figures, manifests, and reproducibility archives

## Main Research Objective

The project investigates whether class-imbalance treatment and knowledge-guided learning consistently improve social-media sentiment classification.

The proposed method, **Lexicon-Guided Reliability-Aware Class-Balanced Focal Loss (LGR-CBFL)**, combines:

1. Class-balanced focal supervision
2. VADER and TextBlob sentiment priors
3. Confidence-gated knowledge regularization
4. Agreement-based reliability reweighting

The ablation analysis separates the effects of symbolic knowledge guidance and reliability-aware instance weighting.

## Dataset

The experiments use the **Reddit Sentiment Analysis Dataset for NLP Projects** available from Kaggle.

Dataset source:

```text
https://www.kaggle.com/datasets/alyahmedts13/reddit-sentiment-analysis-dataset-for-nlp-projects
```

Expected input file:

```text
data.csv
```

Required columns:

| Column | Description |
|---|---|
| `text` | Reddit text used for sentiment classification |
| `label` | Sentiment label |

The notebook normalizes labels into:

| Label | ID |
|---|---:|
| Negative | 0 |
| Neutral | 1 |
| Positive | 2 |

### Important Dataset Note

The dataset provider generated the released sentiment targets using CardiffNLP's Twitter RoBERTa model. Therefore, the labels are treated as **model-generated pseudo-labels**, not independently verified human annotations.

The original Reddit text is not included in this repository. Users must obtain the dataset separately and follow the current dataset license and applicable platform conditions.

The dataset provider reports an Apache License 2.0 license. Users should verify the current license information at the source before reuse or redistribution.

## Cleaned Dataset Profile

After cleaning and exact-duplicate removal, the study used 30,840 observations:

| Class | Observations | Percentage |
|---|---:|---:|
| Negative | 3,293 | 10.68% |
| Neutral | 19,037 | 61.73% |
| Positive | 8,510 | 27.59% |
| Total | 30,840 | 100.00% |

The frozen stratified split was:

| Partition | Observations |
|---|---:|
| Training | 21,588 |
| Validation | 4,626 |
| Test | 4,626 |

Random oversampling was applied only to the training partition.

## Evaluated Models

The benchmark contains 16 models from five methodological families.

### Lexicon Models

- VADER
- TextBlob

### Classical Machine-Learning Models

- Logistic Regression
- Linear Support Vector Machine
- SGD Logistic Regression
- Complement Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

### Recurrent Neural Networks

- LSTM
- BiLSTM
- BiLSTM-Attention

### Transformer Models

- DistilBERT
- BERT-base
- RoBERTa-base
- CardiffNLP Twitter RoBERTa Sentiment

### Ensemble

- Validation-selected soft-voting transformer ensemble

## Training Strategies

The notebook evaluates the following strategies where applicable:

- No explicit balancing
- Random oversampling
- Class weighting
- Focal loss
- Class-balanced focal loss
- Knowledge-prior regularization
- Reliability reweighting
- Full LGR-CBFL

Strategy coverage varies by model family because some objectives require differentiable neural probabilities.

## Main Results

In the primary seed-42 experiment, sentiment-specialized RoBERTa with focal loss achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.965 |
| Balanced accuracy | 0.972 |
| Macro-F1 | 0.959 |
| Negative-class F1 | 0.945 |
| Matthews correlation coefficient | 0.936 |

However, among the four strategies included in the five-seed replication, untreated RoBERTa Sentiment achieved the strongest repeated mean performance:

| Metric | Mean |
|---|---:|
| Accuracy | 0.9552 |
| Balanced accuracy | 0.9562 |
| Macro-F1 | 0.9484 |

The results also showed that:

- Focal loss produced the strongest primary single-run predictive result.
- Untreated RoBERTa Sentiment produced the strongest repeated mean performance.
- Untreated RoBERTa Sentiment had substantially better probability calibration.
- Reliability reweighting slightly improved class-balanced focal learning.
- Direct knowledge-prior regularization reduced predictive performance.
- Full LGR-CBFL did not outperform focal loss or the untreated specialized transformer.
- Model selection changed when calibration, stability, robustness, and computational cost were considered together.

These results should be interpreted as agreement with the released pseudo-labels rather than independent validation against human-annotated ground truth.

## System Requirements

The completed journal run used:

- Windows
- Anaconda-managed Python environment
- NVIDIA GeForce RTX 4070 SUPER
- AMD Ryzen 7 7700
- 16 GB DDR5 RAM
- CUDA-enabled PyTorch
- PyTorch CUDA 12.6 reported during execution

A CUDA-capable GPU is strongly recommended for the complete transformer and replication experiments. CPU execution is possible for some sections but will be substantially slower.

## Installation

### 1. Create an environment

Using Conda:

```bash
conda create -n kbs-sentiment python=3.11 -y
conda activate kbs-sentiment
```

### 2. Install the required packages

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install imbalanced-learn openpyxl joblib tqdm psutil tabulate pyarrow
pip install transformers sentencepiece protobuf accelerate safetensors datasets
pip install vaderSentiment textblob wordcloud gdown requests
pip install jupyter notebook
```

Install a PyTorch build compatible with your operating system, GPU, and CUDA environment.

For a basic installation:

```bash
pip install torch torchvision torchaudio
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
KBS_Ready_NLP_Sentiment_Analysis_Enhanced.ipynb
```

## Data Setup

Place the downloaded dataset in the repository root:

```text
repository/
├── KBS_Ready_NLP_Sentiment_Analysis_Enhanced.ipynb
├── README.md
└── data.csv
```

The default dataset configuration is:

```python
CONFIG = {
    "data_path": "data.csv",
    "text_col": "text",
    "label_col": "label",
}
```

The notebook also contains optional Google Drive download settings. Before publishing or sharing the repository, disable or remove any dataset download link that would redistribute the original text without clear permission.

Recommended public-repository setting:

```python
CONFIG["download_from_google_drive"] = False
CONFIG["data_path"] = "data.csv"
```

## Run Profiles

The notebook supports two run profiles.

### Development Profile

Use this profile to verify installation, data loading, and pipeline execution quickly:

```python
RUN_PROFILE = "development"
```

Development mode reduces epochs, statistical iterations, replication seeds, and selected analyses. Development-mode results must not be reported as final research findings.

### Journal Full Profile

Use this profile for the complete experiment:

```python
RUN_PROFILE = "journal_full"
```

The full profile enables:

- Complete cleaned dataset
- Full model registry
- Full configured strategy coverage
- Five-seed replication
- 10,000 bootstrap iterations
- 5,000 permutation iterations
- Near-duplicate auditing
- Robustness analysis
- Token-occlusion explanations
- Counterfactual sensitivity analysis
- Manuscript artifact generation

The full run is computationally expensive. Runtime depends on the GPU, available memory, internet speed, checkpoint availability, and whether completed runs can be resumed.

## Recommended Execution Order

Run the notebook from top to bottom.

The main sections are:

1. Package installation
2. Imports and global configuration
3. Utility functions and evaluation metrics
4. Dataset loading, cleaning, and splitting
5. Near-duplicate leakage audit
6. Dataset descriptives and figures
7. Unified evaluation and artifact saving
8. Lexicon and classical baselines
9. Recurrent models
10. Transformer models and LGR-CBFL
11. Soft-voting ensembles
12. Primary results and sensitivity analysis
13. Calibration and confidence analysis
14. Disagreement and subgroup analysis
15. Statistical testing
16. Five-seed replication
17. TOPSIS decision support
18. Robustness and external validation
19. Token-occlusion explanations
20. Counterfactual sensitivity audit
21. Manuscript tables and figures
22. Reproducibility manifests and validation
23. Final scientific summary

Do not run later analysis cells before the required model predictions and checkpoints have been generated.

## Checkpoint Recovery

The notebook supports interrupted-run recovery.

Relevant settings:

```python
CONFIG["resume_existing_runs"] = True
CONFIG["save_best_checkpoints"] = True
CONFIG["save_latest_checkpoints"] = True
```

Best checkpoints, latest training states, histories, and completion markers are stored in the output directory. When a valid incomplete run is detected, the notebook attempts to continue training rather than starting from the beginning.

Do not delete checkpoint folders during a partially completed full run unless retraining is intended.

## Output Directory

The notebook creates:

```text
kbs_sentiment_framework_outputs_enhanced/
```

Main subdirectories include:

```text
kbs_sentiment_framework_outputs_enhanced/
├── archives/
├── audits/
├── best_models/
├── calibration/
├── checkpoint_manifests/
├── checkpoints/
├── counterfactual_sensitivity/
├── decision_support/
├── explainability/
├── external_evaluation/
├── figures/
├── histories/
├── knowledge_guided_method/
├── logs/
├── manifests/
├── manuscript_tables/
├── model_cards/
├── models/
├── predictions/
├── reports/
├── replications/
├── results/
├── robustness/
├── statistics/
├── submission_figures/
└── summaries/
```

Generated artifacts include:

- Frozen train, validation, and test partitions
- Dataset audit reports
- Predictions and probability matrices
- Classification reports and confusion matrices
- Training histories
- Best and latest checkpoints
- Multi-seed replication outputs
- Bootstrap confidence intervals
- Permutation, McNemar, and Bowker tests
- Calibration and risk-coverage outputs
- Near-duplicate sensitivity results
- Robustness results
- Token-occlusion tables and figures
- Counterfactual sensitivity results
- TOPSIS rankings
- Publication-ready tables and figures
- Model cards
- Package and hardware manifests
- File hashes and integrity checks
- Manuscript artifact archive

## Reproducibility

The notebook applies random seeds to:

- Python
- NumPy
- PyTorch
- CUDA
- Dataset splitting
- Oversampling
- Data-loader shuffling
- Model initialization

The primary random seed is:

```python
42
```

The replication seeds are:

```python
[42, 52, 62, 72, 82]
```

The notebook also records:

- Package versions
- Hardware information
- Configuration values
- Dataset hashes
- Frozen split hashes
- Checkpoint hashes
- Output inventory
- Completion markers

Exact bitwise reproduction across different operating systems, GPUs, drivers, CUDA versions, and library versions is not guaranteed.

## External Evaluation

The notebook includes optional Hugging Face external evaluation using TweetEval sentiment data.

This analysis requires:

- Internet access
- A functioning Hugging Face datasets installation
- Successful access to the external dataset
- Compatible three-class label mapping
- Available saved transformer checkpoints

In the reported experiment, external TweetEval evaluation was not successfully completed because the dataset failed to load in the execution environment. Therefore, the study does not claim successful cross-dataset generalization.

## Human Label Review

The notebook generates a human-review template containing 450 observations.

The template supports later assessment of:

- Label correctness
- Sarcasm
- Mixed sentiment
- Implicit sentiment
- Slang
- Ambiguity
- Topic-specific language
- Other semantic error categories

The human annotation stage was not completed in the reported experiment. No inter-annotator agreement or human-validated semantic error frequencies are claimed.

## Responsible Use and Privacy

The notebook includes privacy-pattern detection for:

- URLs
- Email addresses
- Reddit usernames
- Subreddit references
- Phone-like strings

When enabled, exported review text is redacted using neutral placeholders.

This project is intended for research and comparative evaluation. Before deployment, users should perform:

- Domain-specific validation
- Independent human-label evaluation
- Calibration assessment
- Fairness and harm analysis
- Privacy review
- Monitoring for language and distribution shift

The counterfactual identity-term module is a limited sensitivity diagnostic and must not be treated as a complete fairness evaluation.

## Known Limitations

1. The primary study uses one English-language Reddit dataset.
2. The sentiment labels are model-generated pseudo-labels.
3. The evaluated sentiment-specialized model is related to the model family used by the dataset provider for label generation.
4. The main experiment uses one frozen stratified split.
5. Five-seed replication is limited to selected RoBERTa Sentiment strategies.
6. Focal loss was not included in the completed five-seed replication.
7. External TweetEval evaluation was not successfully completed.
8. The human-label review template was prepared but not annotated.
9. Near-duplicate observations may influence performance estimates.
10. VADER and TextBlob may not reliably capture sarcasm, slang, implicit sentiment, or community-specific meanings.
11. The identity-term audit is based on a limited set of synthetic templates.
12. Results may not generalize to other languages, platforms, domains, or imbalance ratios.

## Scientific Reporting Guidance

When using this code, report clearly:

- Whether labels are human annotations or pseudo-labels
- Which run profile was used
- Which strategies were repeated across seeds
- Whether external validation completed successfully
- Whether the human-review template was annotated
- Which metrics were used for checkpoint selection
- Whether balancing was restricted to training data
- Which configurations produced genuine probabilities
- Calibration and uncertainty results
- Hardware and package versions
- Any failed or incomplete analyses

Do not present development-mode outputs as final results.

## Suggested Repository Structure

```text
repository/
├── KBS_Ready_NLP_Sentiment_Analysis_Enhanced.ipynb
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── data/
│   └── README.md
└── supplementary_material/
    └── README.md
```

The original dataset should not be committed unless redistribution is explicitly permitted.

Large model checkpoints should normally be stored using Git LFS, an external research archive, or a DOI-based repository rather than standard Git history.

## Citation

Until the article is published, cite the manuscript as:

```text
S. Hossain, S. U. Ahmed, M. A. Shahrier, S. I. Prity, and A. Uddin,
"A Reliability-Aware Knowledge-Guided Decision Framework for Imbalanced
Social Media Sentiment Classification," manuscript, 2026.
```

Temporary BibTeX:

```bibtex
@unpublished{hossain2026reliability,
  author = {Hossain, Sakib and Ahmed, Saeef Uddin and Shahrier, Md. Akib and Prity, Subrina Islam and Uddin, Ashraf},
  title = {A Reliability-Aware Knowledge-Guided Decision Framework for Imbalanced Social Media Sentiment Classification},
  year = {2026},
  note = {Manuscript}
}
```

Replace this temporary citation with the final journal citation and DOI after publication.

## Authors

- Sakib Hossain
- Saeef Uddin Ahmed
- Md. Akib Shahrier
- Subrina Islam Prity
- Ashraf Uddin

Corresponding author:

```text
Ashraf Uddin
American International University-Bangladesh
Email: dr.ashraf@aiub.edu
```

## License

The code license and dataset license are separate.

Before public release, add a repository-level `LICENSE` file approved by all code owners. The chosen code license does not grant permission to redistribute the Reddit dataset, pretrained model files, or third-party resources.

Users are responsible for complying with:

- The selected repository code license
- The dataset license
- Reddit platform conditions
- Hugging Face model licenses
- Third-party package licenses
