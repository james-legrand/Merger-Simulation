from math import log
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

#%% Setup pre-merger parameters

firms = ['firm1', 'firm2', 'firm3']
market_share = {'firm1': 0.2,
                'firm2': 0.3,
                'firm3': 0.5}
elasticity = {('firm1', 'firm1'): -3.0}
market_elasticity = -1

coefficient = {}
coefficient['firm1', 'firm1'] = market_share['firm1'] * (elasticity['firm1', 'firm1'] + 1 - market_share['firm1'] * (market_elasticity + 1))

coefficient.update({
    (firm, firm): market_share[firm] * (1 - market_share[firm]) / (market_share['firm1'] * (1 - market_share['firm1'])) * coefficient['firm1', 'firm1']
    for firm in firms
})

coefficient.update({
    (one_firm, other_firm): -market_share[one_firm] / (1 - market_share[other_firm]) * coefficient[other_firm, other_firm]
    for one_firm in firms for other_firm in firms if one_firm != other_firm
})

elasticity.update({
    (one_firm, other_firm): (
        (-1 + coefficient[one_firm, one_firm] / market_share[one_firm] + market_share[one_firm] * (market_elasticity + 1))
        if one_firm == other_firm
        else (coefficient[one_firm, other_firm] / market_share[one_firm] + market_share[other_firm] * (market_elasticity + 1))
    )
    for one_firm in firms for other_firm in firms
})

margin = {firm: -1 / elasticity[firm, firm] for firm in firms}

merging_firms = ['firm1', 'firm2']
nonmerging_firms = list(set(firms) - {'firm1', 'firm2'})
efficiency_gains = {}
for firm in merging_firms:
    efficiency_gains[firm] = 0.0
for firm in nonmerging_firms:
    efficiency_gains[firm] = 0.0
    
#%% Define post-merger system to solve

post_marketshare, post_elasticity, post_margin, price_change = {}, {}, {}, {}

def vector_function(post_marketshare, post_elasticity, post_margin, price_change):
    eq = []
    # Equations for all firms
    for firm in firms:
        # Equation (1)
        eq.append(
            post_marketshare[firm] - (market_share[firm] + sum(coefficient[firm, other_firm] * log(1 + price_change[other_firm]) for other_firm in firms))
        )
        # Equation (2)
        eq.append(
            post_margin[firm] - (1 - (1 + efficiency_gains[firm]) / (1 + price_change[firm]) * (1 - margin[firm]))
        )
    # Equations for non-merging firms
    for firm in nonmerging_firms:
        # Equation (3)
        eq.append(
            post_margin[firm] + 1 / post_elasticity[firm, firm]
        )
        # Equation (5)
        eq.append(
            post_elasticity[firm, firm] - (-1 + coefficient[firm, firm] / post_marketshare[firm] + post_marketshare[firm] * (market_elasticity + 1))
        )
    # Equations for merging firms
    for firm in merging_firms:
        # Equation (4)
        eq.append(
            post_marketshare[firm] + sum(post_elasticity[other_firm, firm] * post_marketshare[other_firm] * post_margin[other_firm] for other_firm in merging_firms)
        )
        # Equation (6)
        for other_firm in merging_firms:
            eq.append(post_elasticity[firm, other_firm] - (-(firm == other_firm) + coefficient[firm, other_firm] / post_marketshare[firm] + post_marketshare[other_firm] * (market_elasticity + 1))
            )
    return eq

#%% Prepare functions for solver

def wrapper_function(x):
    count = 0
    for d in [post_marketshare, post_margin, price_change]:
        for firm in firms:
            d[firm] = x[count]
            count += 1
    for firm in nonmerging_firms:
        post_elasticity[firm, firm] = x[count]
        count += 1
    for firm in merging_firms:
        for of in merging_firms:
            post_elasticity[firm, of] = x[count]
            count += 1
    return vector_function(post_marketshare, post_elasticity, post_margin, price_change)

def unwrap(x):
    count = 0
    for d in [post_marketshare, post_margin, price_change]:
        for firm in firms:
            d[firm] = x[count]
            count += 1
    for firm in nonmerging_firms:
        post_elasticity[firm, firm] = x[count]
        count += 1
    for firm in merging_firms:
        for of in merging_firms:
            post_elasticity[firm, of] = x[count]
            count += 1
    return [post_marketshare, post_elasticity, post_margin, price_change]

def initial_value():
    x = [market_share[firm] for firm in firms] + [margin[firm] for firm in firms] + [0.0] * len(firms)
    x += [elasticity[firm, firm] for firm in nonmerging_firms]
    x += [elasticity[firm, of] for firm in merging_firms for of in merging_firms]
    return x
#%% Perform baseline simulation
outcome = unwrap(optimize.fsolve(wrapper_function, initial_value()))

#%% Perform comparative statics wrt efficiency gains

efficiency_gain_values = np.arange(-0.74, 0.75, 0.01)
results = {
    "market_share": {},
    "margin": {},
    "price_change": {}
}

# Function to calculate outcomes for a given efficiency gain
def calculate_outcomes(efficiency_gain, firms, merging_firms, efficiency_gains, initial_value, wrapper_function):
    for firm in merging_firms:
        efficiency_gains[firm] = efficiency_gain
    outcome = unwrap(optimize.fsolve(wrapper_function, initial_value()))
    return {
        "market_share": [outcome[0][firm] for firm in firms],
        "margin": [outcome[2][firm] for firm in firms],
        "price_change": [outcome[3][firm] for firm in firms]
    }

# Calculate and store results
for efficiency_gain in efficiency_gain_values:
    outcomes = calculate_outcomes(efficiency_gain, firms, merging_firms, efficiency_gains, initial_value, wrapper_function)
    for key in results:
        results[key][efficiency_gain] = outcomes[key]
        
#%% Plot results of comparative statics

# Define plotting function
def plot_results(ax, results_key, title, ylabel, ylim, firms, colors):
    for idx, firm in enumerate(firms):
        ax.plot(efficiency_gain_values, 
                [results[results_key][gain][idx] for gain in efficiency_gain_values], 
                label=firm, color=colors[idx])
    ax.set_title(title)
    ax.set_xlabel('Efficiency Gains')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.25)
    if results_key == "price_change":
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.25)

# Plot settings
colors = ['#add8e6', '#4682b4', '#00008b']
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Generate plots
plot_results(axs[0], "market_share", "Market Share vs Efficiency Gains", "Market Share", (0, 0.8), firms, colors)
plot_results(axs[1], "margin", "Margin vs Efficiency Gains", "Margin", (0, 0.6), firms, colors)
plot_results(axs[2], "price_change", "Price Change vs Efficiency Gains", "Price Change", (-0.65, 0.65), firms, colors)

# Add legend
fig.legend(labels=['Firm 1: Merged', 'Firm 2: Merged', 'Firm 3: Non-Merged'], loc='lower center', ncol=len(firms))

plt.tight_layout(rect=[0, 0.05, 1, 0.9])
plt.show()
