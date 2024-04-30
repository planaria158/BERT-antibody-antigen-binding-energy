# A-Alpha Bio Machine Learning Technical Assessment

Welcome to the A-Alpha Bio machine learning challenge! 🧬💻👩‍🔬

## Purpose

We issue a homework challenge as means to fairly assess the technical skills of our candidates.

The data provided is a real AlphaSeq data set produced at A-Alpha Bio. The primary task is to fit
a regressor model, which is a routine task for our ML/DS scientists at A-Alpha Bio.
We expect candidates to (as best as possible) demonstrate the depth of their technical skills,
their creativity in approach to machine learning modeling,
their thought process as a scientific researcher,
as well as their professionalism with respect to scientific presentation and communication.

## Instructions

1. Using the data in `alphaseq_data_train.csv`, produce one or more ML regressors to predict the binding affinity `Kd`.
    - At least one of the models should involve deep learning somehow.
2. Selecting a single final regressor model, produce predictions for all data points of the hold out set in `alphaseq_data_hold_out.csv`.
    - Write out the predictions to a CSV file.
3. Prepare a `README.md` file that briefly describes your approach and any instructions on how to use your code.
    - A quick overview of the work done to help us review the submission.
4. Prepare a slide deck (PowerPoint, Google Slides, PDF, etc.) to show your thought process, modeling efforts, and results.
    - Tell us a story about the work and your findings!
5. Submit all source code, predictions, generated data, and the slide deck to the shared Microsoft OneDrive folder that has been provided.
    - Please be mindful to keep the file(s) reasonably sized, not several gigabytes or more.
6. Let A-Alpha Bio know you're done :)

### Additional Notes
- Assuming the submission passes our review, in a following 1-hour interview session, we will schedule you to present your slide deck as well
as a short job talk to an audience at A-Alpha Bio.
- Don't wory: This is not a Kaggle competiton! We do *not* rank candidates by their scores on the hold out set, rather we judge
on the overall quality of the submission (see evaluation rubric below).
We have the true measured AlphaSeq labels for all the data in the hold out set, and we just want to see that the final model performs on par with what
we have seen from our own internal modeling efforts, serving as a fair and consistent data set to do so across candidates' varying approaches.
We also just want to see that the model can be generalized/extended to unseen data points without being subject to information leakage.


## Time

There is *no time limit* on this homework!

Please take as long as needed, but also bear in mind that we are likely evaluating other
candidates concurrently and cannot wait indefinitely. We try to be flexible here. We understand that our candidates are people
with busy lives and that this homework requires devoting a fair amount of time, as does any time spent in an interview process.

We estimate that this homework will take 4-6 hours of focused time for data analysis and modeling, 
and up to 2 hours for preparing the submission and presentation; a total of 8 hours. There is no reward nor penalty for taking shorter or longer,
but we do reccommend placing a time box on yourself. Your time is valuable. There is no perfect model. There is only what can be delivered
in a given time frame.

We will confirm with you that you have recieved the homework. After about two weeks, we will check back in to see if
you are still working on the homework and remain interested in the role.


## Data Dictionary

There are two files of data provided, representing an AlphaSeq experiment to measure the binding affinity
between an scFv variant of Pembrolizumab and the target protein PD-1.

For molecular biology reasons, mutations/indels of the parental scFv are restricted to a window of sequence:
```
TNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFD
```

### `alphaseq_data_train.csv`

AlphaSeq data set to be used in training a regressor for `Kd`. Consists of a site-saturating mutagensis (SSM) scan
and a random selection of double mutants.

Columns:

- `description_a`: A text description of `sequence_a`
- `sequence_a`: The amino acid sequence of the Pembrolizumab scFv variant
- `description_alpha`: A text description of `sequence_alpha`
- `sequence_alpha`: The amino acid sequence of PD-1
- `Kd`: The measured binding affinity, represented as the log10-transformed equilibrium dissociation constant in nM concentration
- `Kd_lower_bound`: The lower bound of the 95% confidence interval of `Kd`
- `Kd_upper_bound`: The upper bound of the 95% confidence interval of `Kd`
- `q_value`: The false discovery rate of the `Kd` measurement, relative to negative controls

### `alphaseq_data_hold_out.csv`

AlphaSeq data set to be used for hold out prediction of `Kd`. The measured `Kd` is omitted for this data set.

Columns:

- `description_a`: A text description of `sequence_a`
- `sequence_a`: The amino acid sequence of the Pembrolizumab scFv variant
- `description_alpha`: A text description of `sequence_alpha`
- `sequence_alpha`: The amino acid sequence of PD-1

Note that some of these sequences involve insertions and deletions relative to the sequences in `alphaseq_data_train.csv`.

## Evaluation Rubric

This is a high-level rubric by which we evaluate the submissions of our candidates. We take into account four major areas:

### 1: Code quality
- Are source code and/or notebooks organized and easily readable?
- Is the code runnable and reproducible?
- Does the code demonstrate strong programming expertise and follow common best practices? 

### 2: Data analysis
- What kind of data wrangling and pre-modeling analysis has been done?
- Are appropriate metrics/statistics/quantities used to interpret the data?
- Does the analysis demonstrate an understanding of working with biomolecular data?

### 3: ML modeling
- Has the candidate approached model training and model selection appropriately?
- Does the modeling demonstrate an understanding of the algorithms and their tradeoffs?
- Are the predictions on the hold out set reasonably accurate?

### 4: Presentation
- Is the presentation of the results organized and easy to follow?
- Are results communicated clearly with visualizations, tables, etc.?
- Are conclusions scientifically sound?


## Integrity

We expect the submissions from candidates to be their own original work, reflective of their individual skills as much as possible.
It's entirely okay to make use of programming libraries, auxiliary data sets, AI generated code, or external models
as long as its source is cited/documented.


## Disclosure

We ask that candidates refrain from redistributing the homework and data sets outside of the interview process with A-Alpha Bio.


## Questions

If you have any questions about the homework, please reach out to Adrian: alange@aalphabio.com


## Cat

This is a cat.

```
    /\_____/\
   /  o   o  \
  ( ==  ^  == )
   )         (
  (           )
 ( (  )   (  ) )
(__(__)___(__)__)
```
# aAlphaBio-Homework
