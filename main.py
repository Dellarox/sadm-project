import os

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind, ttest_rel, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import friedmanchisquare, shapiro
import scikit_posthocs as sp

from lifelines import KaplanMeierFitter


def plots(data):
    os.makedirs("plots", exist_ok=True)
    columns = {
        "Chol": "Cholesterol level [mg/dL]",
        "Age": "Age [years]",
        "ThalAch": "Max BPS Level [beats/min]",
        "TRestBPs": "Rest BPS Level [beats/min]",
    }

    for index, label in enumerate(columns.keys()):
        plt.figure(figsize=(12, 6))
        to_plot = data[label]

        plt.subplot(1, 2, 1)
        plt.hist(to_plot, bins=20, color="skyblue", edgecolor="black")
        plt.xlabel(f"{columns[label]}")
        plt.title(f"Histogram - {columns[label]}")

        plt.subplot(1, 2, 2)
        plt.boxplot(to_plot, vert=False)
        plt.xlabel(f"{columns[label]}")
        plt.title(f"Boxplot - {columns[label]}")

        plt.tight_layout()
        plt.savefig(f"plots/{label}.png")
        plt.show()


def check_normal_distribution(data):
    height_data = data["Height"]
    weight_data = data["Weight"]

    # Visual Inspection
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.hist(height_data, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Height [inch]")
    plt.title("Histogram - height")

    plt.subplot(2, 2, 2)
    plt.hist(weight_data, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Weight [pounds]")
    plt.title("Histogram - weight")

    plt.subplot(2, 2, 3)
    plt.boxplot(height_data, vert=False)
    plt.xlabel("Height [inch]")
    plt.title("Boxplot - height")

    plt.subplot(2, 2, 4)
    plt.boxplot(weight_data, vert=False)
    plt.xlabel("Weight [pounds]")
    plt.title("Boxplot - weight")

    # Shapiro-Wilk Test for Normality
    height_statistics, height_p_value = shapiro(height_data)
    weight_statistics, weight_p_value = shapiro(weight_data)

    print(f"Shapiro-Wilk p-value for height: {height_p_value:.4f}")
    print(f"Shapiro-Wilk p-value for weight: {weight_p_value:.4f}")

    plt.tight_layout()
    plt.savefig("plots/normal_distribution.png")
    plt.show()


# This function calculates basic descriptive statistics for the given data and saves the results to a file.
def descriptive_statistics(data, output_file="descriptive_statistics.csv"):
    # Calculate basic descriptive statistics
    descriptive_stats = data.describe()

    # Transpose the DataFrame (swap rows with columns)
    descriptive_stats = descriptive_stats.transpose()

    # Round data to 2 decimal places
    descriptive_stats = descriptive_stats.round(2)

    # Save rounded and transposed descriptive statistics to a file
    descriptive_stats.to_csv(output_file)

    # Display rounded and transposed descriptive statistics
    print(f"Descriptive Statistics:")
    print(descriptive_stats)


def student_ttest(data):
    height = data["Height"]
    weight = data["Weight"]

    t_stat, p_value = ttest_ind(height, weight)
    print("t-statistic: ", t_stat)
    print("p-value: ", p_value)

    alpha = 0.05
    if p_value < alpha:
        print(
            "Reject the null hypothesis. There is a significant difference between height and weight levels."
        )
    else:
        print(
            "Fail to reject the null hypothesis. There is no significant difference between height and weight levels."
        )


def paired_ttest(data):
    corr_coefficient, p_value = ttest_rel(data["Weight"], data["Height"])

    # Print the results
    print(f"Pearson correlation coefficient: {corr_coefficient}")
    print(f"P-value: {p_value}")

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print(
            "Reject the null hypothesis. There is a significant difference between Height and Weight."
        )
    else:
        print(
            "Fail to reject the null hypothesis. There is no significant difference between Height and Weight."
        )


def kruskal_test(data):
    kruskal_stat, p_value = kruskal(data["ThalAch"], data["Chol"], data["Age"])

    # Print the results
    print(f"Kruskal statistics: {kruskal_stat}")
    print(f"P-value: {p_value}")

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print(
            "Reject the null hypothesis. There is a significant difference between Height and Weight."
        )
    else:
        print(
            "Fail to reject the null hypothesis. There is no significant difference between Height and Weight."
        )


def mann_whitney_u_test(data):
    mann_whitney_u_stat, p_value = mannwhitneyu(data["Age"], data["Chol"])

    # Print the results
    print(f"Mann Whitney U statistic: {mann_whitney_u_stat}")
    print(f"P-value: {p_value}")

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print(
            "Reject the null hypothesis. There is a significant difference between ThalAch and Sex."
        )
    else:
        print(
            "Fail to reject the null hypothesis. There is no significant difference between ThalAch and Sex."
        )


def make_friedman_test():
    np.random.seed(42)

    # Symulacja wyników pomiarów dla różnych grup
    group1 = np.random.normal(30, 5, 20)
    group2 = np.random.lognormal(3, 0.5, 20)
    group3 = np.random.exponential(5, 20)
    group4 = np.random.uniform(10, 20, 20)

    # Tworzenie DataFrame
    data = {
        'PersonID': np.arange(1, 21),
        'Measurement_A': group1,
        'Measurement_B': group2,
        'Measurement_C': group3,
        'Measurement_D': group4
    }

    df = pd.DataFrame(data)

    # Sprawdzenie, czy różnice między grupami mają różny rozkład od normalnego
    normal = True
    for i in range(len(df.columns) - 1):
        for j in range(i + 1, len(df.columns)):
            _, p_shapiro = shapiro(df.iloc[:, i] - df.iloc[:, j])
            if p_shapiro < 0.05:
                print("Znaleziono rozkład różnic między grupami inny niż normalny.")
                normal = False
                break
        if not normal:
            break

    if normal:
        print("Rozkład różnic między grupami jest zawsze normalny.")
    else:
        # Wykonanie testu Friedmana
        stat, p = friedmanchisquare(df['Measurement_A'], df['Measurement_B'], df['Measurement_C'], df['Measurement_D'])
        print(f'Test Friedmana: p = {p}')

        # Analiza post-hoc za pomocą testu Nemenyi
        if p < 0.05:
            post_hoc_result = sp.posthoc_nemenyi_friedman(df.iloc[:, 1:])
            print(post_hoc_result)
        else:
            print("H0 nie zostało odrzucone, nie ma potrzeby wykonania testu post-hoc.")


def plot_survival_curves():
    # Create a DataFrame from the data
    df = pd.read_csv("datasets/strokes.csv")

    # Tworzenie krzywej Kaplana-Meiera
    kmf = KaplanMeierFitter()

    kmf.fit(durations=df['avg_glucose_level'], event_observed=df['stroke'])
    kmf.plot_survival_function()

    plt.title('Krzywa przeżycia Kaplana-Meiera w zależności od ilości glukozy we krwi')
    plt.xlabel('Ilość glukozy we krwi')
    plt.ylabel('Prawdopodobieństwo niedostania udaru')
    plt.xlim(min(df['avg_glucose_level']) - 10, max(df['avg_glucose_level']) + 10)
    legend = plt.legend()
    legend.get_texts()[0].set_text('Ilość glukozy we krwi')
    plt.show()

    kmf = KaplanMeierFitter()

    df['age_higher_then_40_event'] = df['age'].apply(lambda x: 1 if x > 40 else 0)
    df['age_lower_of_equal_40_event'] = df['age'].apply(lambda x: 1 if x <= 40 else 0)

    kmf.fit(durations=df['age'], event_observed=df['stroke'])
    kmf.plot_survival_function()

    plt.title('Krzywa przeżycia Kaplana-Meiera w zależności od wieku')
    plt.xlabel('Wiek')
    plt.ylabel('Prawdopodobieństwo niedostania udaru')
    plt.xlim(min(df['age']), max(df['age']) + 5)
    legend = plt.legend()
    legend.get_texts()[0].set_text('Wiek')
    plt.show()

def pca_roc():
    data = pd.read_csv("datasets/strokes.csv")



    # Extract relevant columns for X
    selected_features = ["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]

    # Convert categorical variables to numerical or boolean values
    gender_mapping = {'Male': 0, 'Female': 1}
    smoking_status_mapping = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}
    ever_married_mapping = {'No': 0, 'Yes': 1}
    work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
    residence_type_mapping = {'Urban': 0, 'Rural': 1}
    data['gender'] = data['gender'].map(gender_mapping)
    data['smoking_status'] = data['smoking_status'].map(smoking_status_mapping)
    data['ever_married'] = data['ever_married'].map(ever_married_mapping)
    data['work_type'] = data['work_type'].map(work_type_mapping)
    data['Residence_type'] = data['Residence_type'].map(residence_type_mapping)
    data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')

    data = data.dropna(axis="rows")
    X = data[selected_features].dropna(axis="rows", how='all').values
    y = data["stroke"].values

    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    n_components = min(len(selected_features), len(data))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_standardized)
    explained_variance_ratio = pca.explained_variance_ratio_
    plt.plot(range(1, n_components + 1), explained_variance_ratio.cumsum(), marker='o')
    plt.title('Explained Variance Ratio vs. Number of Principal Components')
    plt.xlabel('Ilość cech')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()

    results = []
    auc_res = []

    for i in range(1,11):
        X_train, X_test, y_train, y_test = train_test_split(X_pca[:,:i], y, test_size=0.2)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        auc_res.append( roc_auc_score(y_test, y_scores))
        results.append((fpr, tpr))


    plt.figure(figsize=(8, 6))

    for i,res in enumerate(results):
        fpr, tpr=res
        plt.plot(fpr, tpr, lw=2,label=f"Attributes = {i+1}")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend(loc='lower right')

    plt.savefig("plots/roc.png")

    eigenvalues = pca.explained_variance_
    eigenvalue_diff = np.diff(np.insert(eigenvalues, 10, 0))
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    table_data = {
        "Eigenvalue": eigenvalues,
        "Difference": eigenvalue_diff,
        "Proportion": explained_variance_ratio,
        "Cumulative": cumulative_variance_ratio,
        "AUC": auc_res
    }
    result_table = pd.DataFrame(table_data)
    result_table.index += 1
    result_table.to_csv('pca_results_table.csv', index_label='Index')

def main():
    # Loading normal distributed data - https://www.kaggle.com/code/mysha1rysh/gaussian-normal-distribution/notebook
    normal_distributed_data = pd.read_csv("datasets/height_weight.csv")
    # Loading non-normal distributed data - https://www.kaggle.com/datasets/yasserh/heart-disease-dataset
    medical_data = pd.read_csv("datasets/heart_disease.csv")

    plots(medical_data)

    # Calculate and save descriptive statistics
    descriptive_statistics(
        normal_distributed_data,
        output_file="descriptive_stats/descriptive_statistics_height_weight.csv",
    )
    descriptive_statistics(
        medical_data, output_file="descriptive_stats/descriptive_statistics_medical.csv"
    )

    print("Performing the normal distribution check:")
    # Checking normal distribution of data for tests with normal distribution
    check_normal_distribution(normal_distributed_data)
    print("")

    print("Performing the student t-test:")
    # Perform Student t-test - parametric
    student_ttest(normal_distributed_data)
    print("")

    print("Performing the pair t-test:")
    # Perform paired t-test correlation test - parametric
    paired_ttest(normal_distributed_data)
    print("")

    print("Performing the Kruskal-Wallis test:")
    # Perform Kruskal-Wallis test - non parametric
    kruskal_test(medical_data)
    print("")

    print("Performing the Mann Whitney U test:")
    # Perform Mann Whitney U test - non parametric
    mann_whitney_u_test(medical_data)

    make_friedman_test()

    plot_survival_curves()

    pca_roc()


if __name__ == "__main__":
    main()
