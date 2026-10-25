# Abstract Classifier: Science, Business or Social Science?

A small machine learning project that reads the abstract of a research paper and guesses its subject: **Science**, **Business** or **Social Science**.

**Try it online:** https://abstractclassifierproject-anywluv2kvfhx6uvgd238t.streamlit.app/

Paste an abstract to see the predicted subject and how sure the model is, or upload a CSV file with a column called `abstract` to predict many at once. English abstracts only.

## Dataset

- Source: [OpenAlex](https://openalex.org), a free index of research papers.
- Labels come from OpenAlex's own subject tags:
  - Science: Chemistry, Physics, Biochemistry and Genetics
  - Business: Business, Management and Accounting
  - Social Science: Social Sciences and Psychology
- We downloaded about 3,050 random papers, then removed 30 duplicates and abstracts with fewer than 30 words.
- Final dataset: **2,818 abstracts** (960 Science, 945 Business, 913 Social Science), saved in `dataset.csv`.
- `raw_dataset.csv` is the data before cleaning.

## What we did

1. **Cleaning the text:** lowercase, remove HTML tags, links and punctuation, remove stop words, split into words (NLTK tokenization), reduce words to their base form (NLTK WordNet lemmatization).
2. **Turning words into numbers:** TF-IDF with 5,000 features, using single words and word pairs.
3. **Train and test split:** 80% for training (2,254 abstracts), 20% for testing (564 abstracts), keeping the same subject mix in both.
4. **Training six models:** Logistic Regression, Decision Tree, Naive Bayes, kNN, SVM and Random Forest.

## Results (on the 564 test abstracts)

| Model | Accuracy | F1 score |
|---|---|---|
| Logistic Regression | 0.8901 | 0.8889 |
| SVM | 0.8865 | 0.8852 |
| Naive Bayes | 0.8812 | 0.8803 |
| kNN | 0.8457 | 0.8441 |
| Random Forest | 0.8404 | 0.8404 |
| Decision Tree | 0.7305 | 0.7299 |

- The top three models are within 0.9 points of each other, so we cannot say one is truly better.
- The Decision Tree got 99.96% on the training papers but only 73.05% on the test papers. It memorised the training data (overfitting).
- Science is the easiest subject. Most mistakes are between Business and Social Science, because both use words like employees, trust and social.
- The online demo uses **Logistic Regression**, the best model.

## Files

| File | What it is |
|---|---|
| `code.ipynb` | The full notebook: data collection, cleaning, training, charts |
| `dataset.csv` | The cleaned, labelled dataset |
| `raw_dataset.csv` | The dataset before cleaning |
| `streamlit_app.py` | The web app |
| `model.joblib`, `tfidf.joblib` | The saved Logistic Regression model and TF-IDF vectorizer |
| `requirements.txt` | Packages the web app needs |

## Run it yourself

Run the web app on your computer:

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To run the notebook, also install `requests`, `matplotlib` and `textblob`, then run the cells from top to bottom.

## Limitations

- Labels come from OpenAlex tags. People did not read each paper and choose the subject.
- Some papers fit two subjects at once, so there is no single right answer.
- A few abstracts in other languages slipped into the data and were not removed.
- We tested on one split of the data. When we re-ran the notebook the results changed a little, so differences of a point or two between models do not mean much.
- The models look at words only. They do not understand meaning.

## Reference

S. Chowdhury and M. P. Schoen, "Research Paper Classification using Supervised Machine Learning Techniques," 2020 Intermountain Engineering, Technology and Computing (IETC), Orem, UT, USA, 2020, pp. 1-6, doi: 10.1109/IETC47856.2020.9249211.
