# Analiza Personalității Clienților (Customer Personality Analysis) - Kaggle

Acest proiect realizează o analiză detaliată a bazei de clienți utilizând setul de date Kaggle [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis). Proiectul conține două componente principale implementate conform cerințelor academice din `CerinteProiecte.pdf`:
1. **A. Clusterizare (Segmentare)**: Gruparea clienților folosind tehnici nesupravegheate (K-Means pe componente PCA).
2. **B. Clasificare**: Predicția comportamentului de consum și a răspunsului la campania de marketing (`Response`) folosind **11 algoritmi de clasificare** diferiți, inclusiv un clasificator Bayesian neparametric personalizat (KDE).

---

## Structura Proiectului

```
kaggle-personality/
├── .gitignore
├── requirements.txt
├── README.md                  # Acest fișier
├── download_dataset.sh        # Script pentru descărcarea setului de date (Kaggle CLI)
├── data/                      # Folder local pentru date
│   └── marketing_campaign.csv # Setul de date brut (tab-separated)
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Încărcare, curățare și inginerie de caracteristici
│   ├── clustering.py          # Pipeline-ul de K-Means și PCA (Segmentare)
│   ├── classification_prep.py # Split Train/Test/App și evaluarea predictorilor (ANOVA / Mutual Info)
│   ├── custom_nb.py           # Estimator custom Bayesian neparametric (KDE + Histograme)
│   ├── classification_runner.py # Rutine de antrenare, evaluare și plotting
│   └── run_pipeline.py        # Execuția cap-la-cap a procesului de clasificare
├── notebooks/
│   ├── eda_and_clustering.ipynb      # Notebook pentru segmentare (K-Means)
│   └── classification_analysis.ipynb # Notebook executat complet pentru clasificare (11 modele)
├── tests/
│   └── test_custom_nb.py      # Teste unitare pentru clasificatorul custom Naive Bayes
└── output/                    # Fișiere de ieșire generate de scripturi și grafice
    ├── predictor_rankings.csv # Clasamentul variabilelor predictor
    ├── model_comparison_results.csv # Rezultatele comparative ale celor 11 modele
    ├── classification_errors.csv # Tabel cu erorile de clasificare în test set
    ├── application_predictions.csv # Predicțiile finale pe setul de aplicare
    ├── *.png                  # Grafice de performanță (ROC, Lift, Cumulative Gain, PCA Errors, Confusion Matrices)
```

---

## Funcționare și Pipeline de Clasificare

Procesul este automatizat și urmează etapele standard de Machine Learning:

1. **Curățare și Feature Engineering**:
   - Eliminarea valorilor lipsă (`Income`) și a valorilor aberante (anul nașterii < 1900, venituri extreme).
   - Generarea de noi trăsături: `Age`, `Total_Spend`, `Children`, `Is_Parent`, `Total_Purchases`, `Customer_Tenure_Days`.
   - Simplificarea variabilelor categoriale (`Living_Status`, `Education_Level`).
2. **Împărțirea Datelor (Data Split)**:
   - **Set de Antrenare (70%)** - folosit pentru potrivirea transformărilor și antrenare.
   - **Set de Testare (20%)** - folosit exclusiv pentru evaluare și comparare.
   - **Set de Aplicare (10%)** - folosit pentru demonstrarea capabilității de predicție a modelului lider pe date noi (fără etichetă vizibilă).
3. **Evaluarea Predictorilor**:
   - Evaluarea relevanței prin **ANOVA F-value** (caracteristici numerice) și **Mutual Information** (relevanță generală / neliniară) pentru a clasifica și a selecta variabilele predictor.
4. **Clasificatorul Bayesian Custom**:
   - Pentru variabilele continue: estimează probabilitățile de clasă condiționată prin **KDE** (Kernel Density Estimation cu nucleu Gaussian).
   - Pentru variabilele discrete (one-hot): calculează frecvențele categoriale cu **netezire Laplace (Laplace smoothing)**.
5. **Antrenare și Evaluare Generală**:
   - Antrenează 11 algoritmi și generează metrici de performanță (Accuracy, Precision, Recall, F1-Score, AUC-ROC).
   - Creează matrici de confuzie și curbele de robustețe: **ROC**, **Cumulative Gain** și **Lift**.
   - Proiectează instanțele clasificate corect/greșit într-un spațiu bidimensional format prin **PCA (2D)**.

---

## Rezultate Obținute

Modelele au fost antrenate pe setul de antrenare și evaluate pe setul de testare independent. Rezultatele comparative (ordonate descrescător după scorul F1-Score datorită dezechilibrului clasei target) sunt următoarele:

| Loc | Model / Algoritm | Acuratețe (Accuracy) | Precizie (Precision) | Senzitivitate (Recall) | Scorul F1 | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Linear Classifier (LDA)** | **90.52%** | **68.52%** | **59.68%** | **0.6379** | **0.9153** |
| 2 | Linear SVM | 90.29% | 68.52% | 59.68% | 0.6126 | 0.9111 |
| 3 | Logistic Regression (Regresia Logistică) | 90.29% | 68.52% | 59.68% | 0.6126 | 0.9086 |
| 4 | AdaBoost | 90.07% | 68.42% | 57.58% | 0.6071 | 0.8833 |
| 5 | Naive Bayes (Gaussian) | 83.75% | 46.00% | 69.70% | 0.5610 | 0.8345 |
| 6 | Gaussian SVM (RBF) | 89.39% | 70.59% | 36.36% | 0.5053 | 0.9082 |
| 7 | Decision Tree (Arbori de Decizie) | 88.49% | 65.00% | 39.39% | 0.5049 | 0.7984 |
| 8 | kNN | 88.26% | 65.00% | 36.36% | 0.4902 | 0.8188 |
| 9 | Naive Bayes (Non-parametric KDE) | 81.04% | 39.39% | 59.09% | 0.4815 | 0.7952 |
| 10 | Bagging | 88.26% | 81.82% | 18.18% | 0.4091 | 0.8916 |
| 11 | Random Forest | 87.58% | 75.00% | 18.18% | 0.3956 | 0.8722 |

### Concluzii Cheie din Analiză
* **Modele Liniere Dominante**: Algoritmii bazați pe separare liniară (**LDA**, **Linear SVM** și **Regresia Logistică**) obțin cele mai mari scoruri F1 (0.61 - 0.64) și un scor AUC-ROC excelent de ~0.91. Acest lucru demonstrează că procesul de preprocesare și normalizare a ordonat eficient datele pentru separare liniară.
* **Naive Bayes non-parametric (KDE)**: Versiunea noastră custom Bayesiană neparametrică oferă performanțe echilibrate (F1: 0.48), depășind modele ansamblu precum Random Forest și Bagging, care sunt puternic afectate de dezechilibrul claselor (Recall scăzut de 18%).
* **Curba Lift**: Analiza curbei Lift demonstrează că o campanie care folosește modelul LDA pentru a selecta top **20%** din clienții cu probabilitatea cea mai mare de cumpărare va reuși să captureze peste **60%** din totalul persoanelor care ar fi răspuns pozitiv, oferind o eficiență de 3.0x mai mare decât selecția aleatorie.
* **Generalizare pe date noi (Setul de Aplicare)**: Evaluat pe setul de aplicare de 10% (date complet nevăzute), modelul lider (LDA) își păstrează stabilitatea, obținând o **acuratețe de 88.29%** și un scor **F1 de 0.5185** (AUC-ROC: 0.8626).

---

## Ghid de Pornire (Getting Started)

### 1. Configurarea Mediului Python

Asigurați-vă că aveți instalat Python 3.10+ și creați un mediu virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Descărcarea Setului de Date
Descărcați manual setul de date de pe Kaggle și plasați fișierul `marketing_campaign.csv` în directorul `data/` cu separatorul original (tab). Alternativ, rulați:
```bash
./download_dataset.sh
```

### 3. Rularea Pipeline-ului de Clasificare
Pentru a genera toate tabelele comparative, tabelele de erori și graficele sub folderul `output/`, rulați:
```bash
PYTHONPATH=. .venv/bin/python3 src/run_pipeline.py
```

### 4. Rularea Testelor Unitare
Pentru a rula testele unitare care validează clasificatorul custom Naive Bayes:
```bash
.venv/bin/python3 -m unittest discover -s tests -p "test_*.py"
```

### 5. Jupyter Notebooks
Puteți inspecta analizele interactive și vizualizările complete în:
* Segmentare clienți: `notebooks/eda_and_clustering.ipynb`
* Comparație modele clasificare: `notebooks/classification_analysis.ipynb` (poate fi deschis și rulat direct din Jupyter Lab).
