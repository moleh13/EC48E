import numpy as np
import matplotlib.pyplot as plt

# Distribution functions
def uniform_distribution(n):
    return np.random.uniform(0, 1, n)

def exponential_distribution(n):
    return np.random.exponential(1, n)

def beta_distribution(n):
    return np.random.beta(2, 5, n)

def normal_distribution(n):
    return np.random.normal(0, 1, n)

def poisson_distribution(n):
    return np.random.poisson(3, n)

# Function to generate sample stats
def generate_sample_stats(distribution_func, sample_sizes, num_samples):
    sample_means = {}
    sample_variances = {}
    for n in sample_sizes:
        means = []
        variances = []
        for _ in range(num_samples):
            sample = distribution_func(n)
            means.append(np.mean(sample))
            variances.append(np.var(sample, ddof=1))  # Use ddof=1 for unbiased estimate
        sample_means[n] = means
        # Correct variance calculation: Variance of the means, not means of variances
        sample_variances[n] = np.var(means, ddof=1)
    return sample_means, sample_variances

# Main distribution parameters
main_distribution = {
    "Uniform": {"mean": 0.5, "variance": 1 / 12},
    "Exponential": {"mean": 1, "variance": 1},
    "Beta": {"mean": 0.2857, "variance": 0.01905},
    "Normal": {"mean": 0, "variance": 1},
    "Poisson": {"mean": 3, "variance": 3}
}

# Sample sizes and number of samples
sample_sizes = [10, 100, 1000]
num_samples = 1000

# Generate sample means and variances for each distribution
for distribution_name, params in main_distribution.items():
    distribution_func = globals()[f"{distribution_name.lower()}_distribution"]
    actual_mean = params["mean"]
    actual_variance = params["variance"]

    sample_means, sample_variances = generate_sample_stats(distribution_func, sample_sizes, num_samples)

    # Plotting histograms for the means of sample sizes 10, 100, 1000
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Histograms of Sample Means for {distribution_name}')

    for ax, (n, means) in zip(axs, sample_means.items()):
        ax.hist(means, bins=30, density=True, alpha=0.6, color='g')
        ax.axvline(actual_mean, color='r', linestyle='dashed', linewidth=1)
        ax.set_title(f'Sample Size: {n}\nMean: {np.mean(means):.4f}, Variance: {np.var(means, ddof=1):.4f}\nActual Mean: {actual_mean}, Actual Variance: {actual_variance / n:.4f}')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()








# Step 2: Define a function to generate sample means and variances
def generate_sample_stats(distribution_func, sample_sizes, num_samples):
    sample_means = []
    sample_variances = []
    for n in sample_sizes:
        means = []
        variances = []
        for _ in range(num_samples):
            sample = distribution_func(n)
            means.append(np.mean(sample))
            variances.append(np.var(sample, ddof=1))  # Use ddof=1 for unbiased estimate
        sample_means.append(np.mean(means))
        # Correct variance calculation: Variance of the means, not means of variances
        sample_variances.append(np.var(means, ddof=1))
    return sample_means, sample_variances


# Define main distribution parameters
main_distribution = {
    "Uniform": {"mean": 0.5, "variance": 1 / 12},
    "Exponential": {"mean": 1, "variance": 1},
    "Beta": {"mean": 0.2857, "variance": 0.01905},
    "Normal": {"mean": 0, "variance": 1},
    "Poisson": {"mean": 3, "variance": 3}
}

# Define sample sizes and number of samples
sample_sizes = [1, 10, 20, 50, 100, 200, 500, 1000]
num_samples = 1000

# Generate sample means and variances for each distribution, and plot the results
for distribution_name, params in main_distribution.items():
    distribution_func = globals()[f"{distribution_name.lower()}_distribution"]
    actual_mean = params["mean"]
    actual_variance = params["variance"]

    sample_means, sample_variances = generate_sample_stats(distribution_func, sample_sizes, num_samples)

    # Plotting Mean and Variance changes
    plt.figure(figsize=(14, 7))

    # Plot for Means
    plt.subplot(1, 2, 1)
    plt.plot(sample_sizes, [actual_mean] * len(sample_sizes), 'r--', label='Actual Mean')
    plt.plot(sample_sizes, sample_means, 'g-o', label='Sample Mean')
    plt.xlabel('Sample Size')
    plt.ylabel('Mean')
    plt.title(f'Mean Comparison for {distribution_name}')
    plt.legend()

    # Plot for Variance of Sample Means
    plt.subplot(1, 2, 2)
    expected_variance = [actual_variance / n for n in sample_sizes]  # Expected decrease as per CLT
    plt.plot(sample_sizes, expected_variance, 'r--', label='Expected Variance of Sample Means')
    plt.plot(sample_sizes, sample_variances, 'b-o', label='Observed Variance of Sample Means')
    plt.xlabel('Sample Size')
    plt.ylabel('Variance of Sample Means')
    plt.title(f'Variance of Sample Means for {distribution_name}')
    plt.legend()

    plt.tight_layout()
    plt.show()
# Repeat the plotting process for other distributions as needed