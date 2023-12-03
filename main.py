import pandas as pd
import scipy.stats as stats
from scipy.stats import f_oneway, chi2_contingency, kruskal, pearsonr


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
    print("Descriptive Statistics:")
    print(descriptive_stats)


def tstudent_test(data):
    # The t-test will be performed between the Age and Chol columns
    # The null hypothesis is that there is no significant difference between the means of age and cholesterol levels.
    age = data["Age"]
    cholesterol = data["Chol"]

    t_stat, p_value = stats.ttest_ind(age, cholesterol)
    print("t-statistic: ", t_stat)
    print("p-value: ", p_value)

    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis. There is a significant difference between age and cholesterol levels.')
    else:
        print(
            'Fail to reject the null hypothesis. There is no significant difference between age and cholesterol levels.')


def anove_test(data):
    # Extract 'Age' and 'ExAng' columns
    age = data['Age']
    ex_angina = data['ExAng']

    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(age, ex_angina)

    # Print the results
    print(f'F-statistic: {f_stat}')
    print(f'P-value: {p_value}')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis. There is a significant difference between age and Ex Induced Angina.')
    else:
        print(
            'Fail to reject the null hypothesis. There is no significant difference between age and Ex Induced Angina.')


def pearson_correlation(data):
    # Perform Pearson correlation test
    corr_coefficient, p_value = pearsonr(data['Age'], data['ExAng'])

    # Print the results
    print(f'Pearson correlation coefficient: {corr_coefficient}')
    print(f'P-value: {p_value}')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis. There is a significant correlation between Age and ExAng.')
    else:
        print('Fail to reject the null hypothesis. There is no significant correlation between Age and ExAng.')


def chi_square_test(data):
    # Create a contingency table
    contingency_table = pd.crosstab(data['Age'], data['Sex'])

    # Perform chi-square test
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

    # Print the results
    print(f'Chi-square statistic: {chi2_stat}')
    print(f'P-value: {p_value}')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis. There is a significant association between Age and Sex.')
    else:
        print('Fail to reject the null hypothesis. There is no significant association between Age and Sex.')


def kruskal_wallis_test(data):
    # Perform Kruskal-Wallis test
    kruskal_stat, p_value = kruskal(data['ThalAch'], data['Sex'])

    # Print the results
    print(f'Kruskal-Wallis statistic: {kruskal_stat}')
    print(f'P-value: {p_value}')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis. There is a significant difference between ThalAch and Sex.')
    else:
        print('Fail to reject the null hypothesis. There is no significant difference between ThalAch and Sex.')


def main():
    # Load data
    data = pd.read_csv("heart_disease.csv")

    # Calculate and save descriptive statistics
    descriptive_statistics(data)

    # In our case, the normal distribution is not necessary, because of a lot of samples.
    tstudent_test(data)

    # Perform one-way ANOVA
    anove_test(data)

    # Perform Pearson correlation test
    pearson_correlation(data)

    # Perform chi-square test
    chi_square_test(data)

    # Perform Kruskal-Wallis test
    kruskal_wallis_test(data)


if __name__ == "__main__":
    main()