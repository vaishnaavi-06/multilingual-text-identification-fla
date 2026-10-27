# Multilingual Text Identification Using Finite Automata

## 1. Project Overview
This project identifies the language of a given text using character-level machine learning and explores the role of finite automata in text processing.

## 2. Objectives
- Identify languages from input sentences.
- Explore character patterns in multilingual text.
- Demonstrate finite automata concepts through Unicode writing-system detection.
- Evaluate a machine-learning language classifier.

## 3. Dataset
The project uses a dataset containing 22,000 text samples across 22 languages. The dataset includes two columns: `Text` and `language`.

Dataset source: https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst

## 4. Methodology
1. Load and inspect the dataset.
2. Split the data into training and testing sets.
3. Convert character sequences into TF-IDF features.
4. Train a Linear Support Vector Classifier (LinearSVC).
5. Evaluate the model using accuracy and a classification report.
6. Predict the language of new input sentences.

The project also includes a Unicode-based writing-system detector. This demonstrates character-range scanning related to automata concepts; it is not, by itself, a complete formal DFA implementation.

## 5. Results
The model achieved **98.70% test accuracy** in the current experiment.

## 6. Technologies Used
- Python
- Pandas
- Scikit-learn
- Google Colab
- Visual Studio Code

## 7. How to Run
1. Download or clone this repository.
2. Install the required Python libraries.
3. Place `dataset.csv` in the same folder as `main.py`.
4. Run `python main.py` in the terminal.

For Google Colab, open the notebook and upload `dataset.csv` when required before running the cells.

## 8. Social Impact
The project relates to **SDG 9: Industry, Innovation and Infrastructure** through applied language technology, and **SDG 10: Reduced Inequalities** through the broader goal of supporting multilingual access to digital tools.

## 9. Project Links
- Google Colab Notebook: https://colab.research.google.com/drive/1Wqy8-xHRBgzd77nsHbJ_6Sk1tcEN5YB4?usp=sharing
- Dataset: https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst

## 10. Conclusion
This project demonstrates multilingual language classification using character-level features and machine learning, alongside a Unicode writing-system detector that introduces concepts relevant to text processing and finite automata.
