# Machine Learning Foundations Project: Predicting Corporate Bankruptcy from Financial Statements

**GitHub repository:** https://github.com/mauroreverberi/machine-learning-foundations
(all commits and the `develop` branch as well as the `main` branch are visible there)

## Project Description
I train and evaluate supervised machine learning models that predict from the
financial statement indicators of a company whether it will go bankrupt
within the next three years. I compare a dummy baseline, a hand-written solvency
rule, a logistic regression and a random forest, select the better learned
model on a validation set, and turn its scores into an explicit operating
decision through a threshold analysis. The selected model is evaluated once
on a held-out test set.

The question connects to my earlier capstone projects. In Project 1 I built a
cleaned dataset of Swiss legal entities from the GLEIF register, in Project 2
I analyzed new company registrations from the Swiss commercial gazette. A
later capstone project will build a due diligence agent that looks up a
company and assesses it, and such an agent needs exactly the decision modeled
here. Given the numbers a company publishes, how urgently does a human
analyst need to look at it?

## What Was Built
- A Jupyter notebook (`modeling.ipynb`) with the whole workflow: loading the
  dataset from the original archive, data checks, preparation and
  preprocessing, four models from dummy baseline to random forest, a
  validation comparison with precision-recall analysis, a threshold analysis
  that turns scores into decisions, a final test evaluation and a short
  summary.
- A written report (`Machine_Learning_Analysis_Report.pdf`) that explains the
  problem, dataset, modeling approach, results and limitations for technical
  and non-technical readers.
- A reproducibility file (`requirements.txt`), generated with `pip freeze`
  inside the project's own virtual environment.

## Dataset
Polish Companies Bankruptcy Data, UCI Machine Learning Repository (dataset
365), license CC BY 4.0:
https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

The original archive `polish+companies+bankruptcy+data.zip` is included
unchanged in this repository, I downloaded it on 2026-08-24. The archive
holds five ARFF files, one per forecasting horizon, all on Polish companies
from the manufacturing sector. An ARFF file is a
tabular format, a CSV table with a small attribute header. The notebook uses
the `3year` file (10,503 companies, 64 financial indicators, 495 bankrupt
within three years) and reads it directly from the archive, so no extraction
or conversion step is needed and the data stays exactly as UCI publishes it.
For convenience, `3year.arff` is also included as an extracted, byte-identical
copy of that file, so the data can be opened directly as a plain text table.
The notebook itself reads the file from the archive.

## Main Results
The random forest won the model comparison on validation average precision
(0.4687 against 0.3035 for the logistic regression) and was carried forward
alone. Its scores feed a three-band operating policy with two thresholds
fixed on the validation set, urgent review from a score of 0.160 and
watchlist from 0.047 (approximately 0.04667). On the untouched test set the frozen policy put 99 of
the 2,084 companies into urgent review, 48 of them real future bankruptcies
(48.5%), and the two flagged bands together caught 85.9% of the bankruptcies
in the test set, with a test average precision of 0.5108. The most
surprising finding was that the most important input of the forest is the
missing indicator of one ratio (Attr27), the gaps in the data carry signal
of their own.

## How to Run the Project
1. Clone this repository.
2. Create and activate a virtual environment (Python 3.12 or newer, tested with 3.14):
   ```
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Start Jupyter:
   ```
   jupyter lab
   ```
5. Open `modeling.ipynb` and run all cells
   (Kernel > Restart Kernel and Run All Cells).

The dataset archive is already part of the repository, so no download is
needed to run the notebook. The notebook uses relative paths and a fixed
random seed (42), so a full run reproduces every number in the report.

## Dependencies
`requirements.txt` was created inside the project's own virtual
environment with:
```
pip freeze > requirements.txt
```

## Connection to Future AI Work
I stay in the domain of official company and financial data. Projects 1 and
2 built and analyzed register snapshots, this project adds the modeling
step, from raw financial indicators to a risk score with an explicit
decision policy. The planned due diligence agent of the later capstone
projects needs exactly this building block when it has to decide how
urgently a human analyst should look at a company.
