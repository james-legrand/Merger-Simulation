import numpy as np
import matplotlib.pyplot as plt

# Define vectors and matrices
a = np.full((3, 1), 10)
B = np.full((3, 3), 0.3)
np.fill_diagonal(B, -2)
delta = np.array([[1,1,0],
                  [1,1,0],
                  [0,0,1]])
c_pre = 1  # Pre-merger cost
gamma_values = np.linspace(-0.75, 0.75, 150)  # Range of cost savings (-50% to +50%)

prices_firm1 = []
prices_firm3 = []
profits_firm1 = []
profits_firm3 = []

# Loop over different cost savings (gamma)
for gamma in gamma_values:
    c_post = np.full((3, 1), c_pre, dtype=float)
    c_post[0] = c_post[1] = c_pre + gamma  # Adjust cost for firms 1 & 3 post-merger

    # Compute new equilibrium price
    Delta_Hadamard_B = delta * B
    inverse_term = np.linalg.inv(B.T + Delta_Hadamard_B)
    price = inverse_term @ (-a + (Delta_Hadamard_B @ c_post))

    # Compute demand
    D_p = a + B.T @ price

    # Compute profits
    profits = delta.T @ ((price - c_post) * D_p)

    # Store results
    prices_firm1.append(price[0, 0])
    prices_firm3.append(price[2, 0])
    profits_firm1.append(profits[0, 0])
    profits_firm3.append(profits[2, 0])

# Define plotting function
def plot_results(ax, results, results_key, title, ylabel, ylim, firms, colors):
    for idx, firm in enumerate(firms):
        ax.plot(gamma_values, 
                results[results_key][:, idx], 
                label=firm, color=colors[idx])
    ax.set_title(title)
    ax.set_xlabel('Cost Change (γ)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.25)
    if results_key == "price":
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.25)

# Store computed results for plotting
results = {
    "price": np.column_stack((prices_firm1, prices_firm3)),
    "profit": np.column_stack((profits_firm1, profits_firm3))
}

# Firms and colors
firms = ["Firm 1: Merged", "Firm 2: Non-Merged"]
colors = ['#add8e6', '#00008b']  # Light blue for Firm 1, Dark blue for Firm 3

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Generate plots
plot_results(axs[0], results, "price", "Price vs Cost Change", "Price", (min(prices_firm1 + prices_firm3) - 0.5, max(prices_firm1 + prices_firm3) + 0.5), firms, colors)
plot_results(axs[1], results, "profit", "Profit vs Cost Change", "Profit", (min(profits_firm1 + profits_firm3) - 0.5, max(profits_firm1 + profits_firm3) + 0.5), firms, colors)

# Add legend
fig.legend(labels=firms, loc='lower center', ncol=2)

plt.tight_layout(rect=[0, 0.05, 1, 0.9])
plt.show()

