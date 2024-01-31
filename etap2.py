import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, shapiro
import scikit_posthocs as sp

# Generowanie syntetycznego datasetu
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
