import pandas as pd
from scipy.stats import shapiro, ttest_ind, ttest_rel, mannwhitneyu, kruskal
import matplotlib.pyplot as plt


def check_normal_distribution(data):
    height_data = data["Height"]
    weight_data = data["Weight"]

    # Visual Inspection
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.hist(height_data, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram')

    plt.subplot(2, 2, 2)
    plt.hist(weight_data, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram')

    plt.subplot(2, 2, 3)
    plt.boxplot(height_data, vert=False)
    plt.title('Boxplot')

    plt.subplot(2, 2, 4)
    plt.boxplot(weight_data, vert=False)
    plt.title('Boxplot')

    # Shapiro-Wilk Test for Normality
    height_statistics, height_p_value = shapiro(height_data)
    weight_statistics, weight_p_value = shapiro(weight_data)

    print(f'Shapiro-Wilk p-value for height: {height_p_value:.4f}')
    print(f'Shapiro-Wilk p-value for weight: {weight_p_value:.4f}')

    plt.tight_layout()
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
        print('Reject the null hypothesis. There is a significant difference between height and weight levels.')
    else:
        print(
            'Fail to reject the null hypothesis. There is no significant difference between height and weight levels.')


def paired_ttest(data):
    corr_coefficient, p_value = ttest_rel(data['Weight'], data['Height'])

    # Print the results
    print(f'Pearson correlation coefficient: {corr_coefficient}')
    print(f'P-value: {p_value}')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis. There is a significant correlation between Height and Weight.')
    else:
        print('Fail to reject the null hypothesis. There is no significant correlation between Height and Weight.')


def kruskal_test(data):
    kruskal_stat, p_value = kruskal(data['Age'], data['Chol'], data['ThalAch'])

    # Print the results
    print(f'Kruskal statistics: {kruskal_stat}')
    print(f'P-value: {p_value}')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis. There is a significant correlation between Height and Weight.')
    else:
        print('Fail to reject the null hypothesis. There is no significant correlation between Height and Weight.')


def mann_whitney_u_test(data):
    mann_whitney_u_stat, p_value = mannwhitneyu(data['ThalAch'], data['Sex'])

    # Print the results
    print(f'Mann Whitney U statistic: {mann_whitney_u_stat}')
    print(f'P-value: {p_value}')

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis. There is a significant difference between ThalAch and Sex.')
    else:
        print('Fail to reject the null hypothesis. There is no significant difference between ThalAch and Sex.')


def main():
    # Loading normal distributed data
    normal_distributed_data = pd.read_csv("datasets/height_weight.csv")
    # Loading non-normal distributed data
    medical_data = pd.read_csv("heart_disease.csv")

    # Calculate and save descriptive statistics
    descriptive_statistics(normal_distributed_data, output_file="descriptive_stats/descriptive_statistics_height_weight.csv")
    descriptive_statistics(medical_data, output_file="descriptive_stats/descriptive_statistics_medical.csv")

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


if __name__ == "__main__":
    main()