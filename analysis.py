# Import libraries
import numpy as np
import scipy.stats as stats
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
from collections import defaultdict


mask = 2**16 - 1
alpha, beta, gamma = 1, 8, 2
SIMON_ROUNDS = 10


# Define the NMTS class to store the differentials
class NMTS:
    def __init__(self, dx=None, dy=None, dz=None, wt=None):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.wt = wt
        

# function to load the differentilas and probabilities from text files
def highwaylist():
    hw = []
    with open('alph.txt') as ip1, open('bet.txt') as ip2, open('gam.txt') as ip3, open('pro.txt') as ip4:
        for i1, i2, i3, i4 in zip(ip1, ip2, ip3, ip4):
            hw.append(NMTS(int(i1.strip()), int(i2.strip()), int(i3.strip()), float(i4.strip())))
    return hw


# Differential calculation functions
def ROR(x, r, mask):
    return ((x << (16 - r)) + (x >> r)) & mask

def ROL(x, r, mask):
    return ((x >> (16 - r)) + (x << r)) & mask

# calculate the hamming weight
def weightAND(alpha1, beta1, gamma1, mask):
    s = (gamma1 & (~(alpha1 ^ beta1))) & mask
    temp = bin((alpha1 ^ beta1) & mask)
    wt = temp[1:].count("1")
    return (-200, 200) if s != 0 else (wt, 2**(-wt))


# Find the differential path
def find_diff_path(st1, st0, SIMON_ROUNDS, mask, alpha, beta, gamma):
    tempdec_list = []
    temp_wt = 0
    for _ in range(SIMON_ROUNDS):
        A = ROL(st1, alpha, mask)
        B = ROL(st1, beta, mask)
        C = ROL(st1, gamma, mask)
        while True:
            op=hw[random.randint(0,len(hw)-1)].dz
            wt, _ = weightAND(A, B, op, mask)
            if wt != -200:
                break
        D = op ^ C
        E = st0 ^ D
        tempdec_list.append(NMTS(st1, st0, op, wt))
        temp_wt += tempdec_list[-1].wt
        st0 = st1
        st1 = E
    return tempdec_list, temp_wt



# Find best path
def find_best_path(st1, st0, SIMON_ROUNDS, mask, alpha, beta, gamma):
    best_wt = float('inf')
    for _ in range(10):  # Number of trials for pathfinding
        tempdec_list, temp_wt = find_diff_path(st1, st0, SIMON_ROUNDS, mask, alpha, beta, gamma)
        if temp_wt < best_wt:
            best_wt = temp_wt
            best_path = tempdec_list
    return best_wt, best_path



# run experiemnt function
def run_experiment(differential, num_trials=10):
    
    results = []

    for _ in range(num_trials):
        st1 = differential.dx
        st0 = differential.dy
        best_wt, _ = find_best_path(st1, st0, SIMON_ROUNDS, mask, alpha, beta, gamma)
        results.append(best_wt)
    return results


# main initial body

# Read data into NMTS objects
hw = highwaylist()

# Perform experiments

# Create empty array for experiment results
experiment_results = []
for diff in hw:
    if diff.wt >= 0.5:
        try:
            result = run_experiment(diff)
            experiment_results.extend(result)
        except Exception as e:
            print(f"Error processing differential: {diff} - {e}")
            
# Convert results to a numpy array for analysis
experiment_results = np.array(experiment_results)

baseline_mean = 0.25 # Insignificnt differentials

# Calculate baseline mean from all probability values
probabilities = [diff.wt for diff in hw] # Collect all probability (pro) values
mean_baseline = np.mean(probabilities)

t_stat, p_value = stats.ttest_1samp(experiment_results, mean_baseline)

print(f"T-test: t-statistic = {t_stat}, P-value = {p_value}")



# Calculate the mean of all probabilities

baseline_means = [0.125, 0.25, 0.5, 1]
for baseline in baseline_means:
    t_stat, p_value = stats.ttest_1samp(experiment_results, baseline)
    print(f"Baseline mean = {baseline}: t-stat-stic = {t_stat}, P-value = {p_value}")
    
# Print the mean of all probabilities to serve as a reference
print(f"Mean of all probabilities: {mean_baseline}")




# Filter differentials with a probability >= 0.5
significant_differentials = [diff for diff in hw if diff.wt >= 0.5]

# Analyse characteristics
dx_values = [diff.dx for diff in significant_differentials]
dy_values = [diff.dy for diff in significant_differentials]
dz_values = [diff.dz for diff in significant_differentials]
probabilities = [diff.wt for diff in significant_differentials]

# Plot a histogram of the differentials
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.hist(dx_values, bins=30, edgecolor="k", alpha=0.7)
plt.title("Histogram of △X")
plt.xlabel('Differential')
plt.ylabel('Frequency')

plt.subplot(3, 1, 2)
plt.hist(dy_values, bins=30, edgecolor="k", alpha=0.7)
plt.title("Histogram of △Y")
plt.xlabel('Differential')
plt.ylabel('Frequency')

plt.subplot(3, 1, 3)
plt.hist(dz_values, bins=30, edgecolor="k", alpha=0.7)
plt.title("Histogram of △Z")
plt.xlabel('Differential')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



# Create DataFrame for pairwise analysis
data = pd.DataFrame({
    '△X': dx_values,
    '△Y': dy_values,
    '△Z': dz_values,
    'Probability': probabilities
})

# Pairplot to visualize relationships between dx, dy, dz
sns.pairplot(data)
plt.show()






# Perform K-means clustering
kmeans = KMeans(n_clusters=10)  # 10 clusters
data['cluster'] = kmeans.fit_predict(data[['△X', '△Y', '△Z']])

# Plot clusters
sns.scatterplot(x='△X', y='△Y', hue='cluster', data=data, palette='viridis', s=100)
plt.title('Clustering of Differentials')
plt.show()




data_corr = pd.DataFrame({
    '△X': dx_values,
    '△Y': dy_values,
    '△Z': dz_values
})


correlation_matrix = data_corr.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Differentials')
plt.show()



# Extract significant differentials

# Define a threshold for significance. eg. Probabilities greater than 0.5
significant_differentials = [diff for diff in hw if diff.wt >= 0.5]

# Extract values for further testing
dx_values_significant = [diff.dx for diff in significant_differentials]
dy_values_significant = [diff.dy for diff in significant_differentials]
dz_values_significant = [diff.dz for diff in significant_differentials]



# Perform experiments using significant differentials. Employ the run_experiment() function defined earlier
experiment_results_significant = [run_experiment(diff) for diff in significant_differentials]

# Convert results to a numpy array for analysis
experiment_results_significant = np.array([item for sublist in experiment_results_significant for item in sublist])




with open("experiment_results_significant.txt", "a") as file:
    for item in experiment_results_significant:
        file.write(str(item) + "\n")

print("Variables saved to experiment_results_significant.txt")


def sample_data(hw):
    # Set the sample size as a decimal percentage
    samepl_size = 0.1
    
    # group differentials by the dz attribute values
    grouped_data = defaultdict(list)
    
    for item in hw:
        grouped_data[item.dz].append(item) # store the whole item
    
    # create an empty list
    final_nmts_list = []
    
    # Sample from each differential based on dz samples
    for dz, group in grouped_data.items():
        group_size = len(group)
        num_samples = max(1, math.ceil(group_size * samepl_size)) # Ensures at least 1 of each differential if it is too small
        
        for item in group[:num_samples]:
            nmts_instance = NMTS(dx=item.dx, dy=item.dy, dz=item.dz, wt=item.wt)
            final_nmts_list.append(nmts_instance)
    
   
    print(len(final_nmts_list))
    
    
    
    
    return final_nmts_list



non_significant_differentials = [diff for diff in hw if diff.wt < 0.5]
sampled_non_significant_differentials =  sample_data(hw)


print(len(sampled_non_significant_differentials))

# Extract results for the sampled non-significant differentials
experiment_results_non_significant = [run_experiment(diff) for diff in sampled_non_significant_differentials]


experiment_results_non_significant = np.array([item for sublist in experiment_results_non_significant for item in sublist])

# Statistical comparison
# t_stat, p_value = stats.ttest_ind(experiment_results_significant, experiment_results_non_significant)
# print(f"Comparison T-test: t-statistic = {t_stat}, P-value = {p_value}")

with open("experiment_results_non_significant.txt", "a") as file:
    for item in experiment_results_non_significant:
        file.write(str(item) + "\n")

print("Variables saved to experiment_results_non_significant.txt")


file_path = 'experiment_results_non_significant.txt'  # Replace with the actual path to your file

lines = []
with open(file_path, 'r') as file:
    for line in file:
        lines.append(line.strip())
        
# lines

# experiment_results_significant = lines


plt.figure(figsize=(8, 6))

#plt.subplot(1, 2, 1)
# plt.hist(experiment_results_significant, bins=30, edgecolor='k', alpha=0.7, label='Significant')
# Create histogram with KDE overlay
sns.histplot(experiment_results_significant, kde=True, bins=30, line_kws={'linestyle': '--', 'linewidth': 3})
#plt.title('Distribution of Significant \nDifferentials Results', fontsize=22)
plt.xlabel('Calculated Hamming Weight', fontsize=22)
plt.ylabel('Distribution', fontsize=22)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.show()


# plt.figure(figsize=(8, 6))
# #plt.subplot(1, 2, 2)
# plt.hist(experiment_results_non_significant, bins=30, edgecolor='k', alpha=0.7, label='Non-Significant')
# #plt.title('Distribution of Non-Significant \nDifferentials Results', fontsize=22)
# plt.xlabel('Calculated Hamming Weight', fontsize=22)
# plt.ylabel('Distribution', fontsize=22)
# plt.tick_params(axis="both", which="major", labelsize=16)


# #plt.tight_layout()
# plt.show()


file_path = 'experiment_results_significant.txt'  # Replace with the actual path to your file

lines_sig = []
with open(file_path, 'r') as file:
    for line in file:
        lines_sig.append(line.strip())
        
# lines_sig

# experiment_results_significant = lines


plt.figure(figsize=(8, 6))
#plt.subplot(1, 2, 2)
# plt.hist(experiment_results_non_significant, bins=[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90], edgecolor='k', alpha=0.7, label='Non-Significant')
#plt.title('Distribution of Non-Significant \nDifferentials Results', fontsize=22)
sns.histplot(experiment_results_non_significant, kde=True, bins=30, line_kws={'linestyle': '--', 'linewidth': 3}, kde_kws={'bw_adjust': 2.5})

plt.xlabel('Calculated Hamming Weight', fontsize=22)
plt.ylabel('Distribution', fontsize=22)
plt.tick_params(axis="both", which="major", labelsize=16)


#plt.tight_layout()
plt.show()


# Combine results into DataFrame for plotting
df = pd.DataFrame({
    'Hamming Weights': np.concatenate([experiment_results_significant, experiment_results_non_significant]),
    'Differential Group': ['Significant Differentials'] * len(experiment_results_significant) + ['Non-Significant Differentials'] * len(experiment_results_non_significant)
})

# Box plot for comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='Differential Group', y='Hamming Weights', data=df)
plt.xlabel('Differential Group', fontsize=22)
plt.ylabel('Hamming Weights', fontsize=22)
plt.tick_params(axis="both", which="major", labelsize=16)
#plt.title('Comparison of Experiment Results')
plt.show()


%%time
# Get all significant differentials with a weight of 0
significant_differentials_with_zero_weight = []
for index, diff in enumerate(significant_differentials):
    if diff.wt >= 0.5:  # Check if the differential is significant
        # Calculate the hamming weight from the differential
        st1 = diff.dx
        st0 = diff.dy
        op = diff.dz
        
        # Calculate the Hamming weight
        wt, _ = weightAND(st1, st0, op, mask)  # Assuming mask is 0xFFFF
        
        if wt == 0:
            significant_differentials_with_zero_weight.append((index, diff.dx, diff.dy, diff.dz))
            
# Output the significant differentials with Hamming weight of 0
for idx, dx, dy, dz in significant_differentials_with_zero_weight:
    print(f"Index: {idx}, Input Differentials: dx={dx}, dy={dy}, Output Differential: dz={dz}, Hamming Weight: 0")


# Print the number of significant differentials with 0 weight
number_sig_diffs_zero = len(significant_differentials_with_zero_weight)
print(number_sig_diffs_zero)


%%time
# Simulate rounds
def simulate_rounds(dx, dy, dz, SIMON_ROUNDS):
    
    # Define empty lists
    final_differentials = []
    final_probabilities = []
    final_weight = []
    log2p = []

    for _ in range(SIMON_ROUNDS):
        # Use SIMON cipher round function
        temp_dx = dx
        dx = dy ^ (ROR(dx, alpha, mask) & ROR(dx, beta, mask)) ^ ROR(dx, gamma, mask)
        dy = temp_dx
        
        # Calculate the Hamming weight
        wt = bin(dx ^ temp_dx).count('1')
        
        # Track the new differentials and their associated probability
        final_differentials.append((hex(dx), hex(dy), hex(dz)))
        final_probabilities.append(2**(-wt))  # Probability
        final_weight.append(wt)
        #log2p.append(math.log2(2**(-wt)) if (2**(-wt)) > 0 else float('-inf'))
        log2p.append(-wt)
        
    return final_differentials, final_probabilities, final_weight, log2p



%%time
SIMON_ROUNDS = 32

for dif in significant_differentials_with_zero_weight:
    differentials, probabilities, final_weight, log2p = simulate_rounds(dif[1], dif[2], dif[3], SIMON_ROUNDS)
    log2psum = 0
    prob_mean = 0
    csv_table = "Round, dx, dy, dz, wt, probability, log2p\n"
    for i, (diff, prob, wt, log2p) in enumerate(zip(differentials, probabilities, final_weight, log2p)):
        dx, dy, dz = diff

        
        csv_table += f"{i}, {dx},{dy},{dz},{wt},{prob:.8f},{log2p}\n"
        #print(f"Round {i}: {dx}, {dy}, {dz}, Wt= {wt}, Probability = {prob}, log2p = {log2p}")
        log2psum = log2psum + (log2p)
        prob_mean += np.mean(prob)
    csv_table += f" ,  , , , ,{(prob_mean/SIMON_ROUNDS):.8f}, {log2psum}\n"
    print(csv_table)



most_promising_differentials = []
for idx, dx, dy, dz in significant_differentials_with_zero_weight:
    final_differentials, final_probabilities, final_wt, log2p = simulate_rounds(dx, dy, dz, SIMON_ROUNDS) 
    avg_probability = np.mean(final_probabilities)
    most_promising_differentials.append((idx, dx, dy, dz, avg_probability))

# Sort by average probability to find the most promising differentials
most_promising_differentials.sort(key=lambda x: x[4], reverse=True)

for idx, dx, dy, dz, avg_prob in most_promising_differentials:
    print(f"Index: {idx}, dx: {dx}, dy: {dy}, dz: {dz}, Average Probability: {avg_prob}")


most_promising_differentials_log2p = []
for idx, dx, dy, dz in significant_differentials_with_zero_weight:
    final_differentials, final_probabilities, final_wt, log2p = simulate_rounds(dx, dy, dz, SIMON_ROUNDS) 
    log2p_probability = np.sum(log2p)
    most_promising_differentials_log2p.append((idx, dx, dy, dz, log2p_probability))

# Sort by average probability to find the most promising differentials
most_promising_differentials_log2p.sort(key=lambda x: x[4], reverse=True)

for idx, dx, dy, dz, log2p_prob in most_promising_differentials_log2p:
    print(f"Index: {idx}, dx: {dx}, dy: {dy}, dz: {dz}, Sum Log2: {log2p_prob}")



#count how many 0 dz values
diffs_zero = [diff for diff in hw if diff.dz == 0]

total_zero = len(diffs_zero)

print(f"Total output differentials with a zero value: {total_zero}")

percent_significant = (number_sig_diffs_zero / total_zero) * 100
print(f"Percentage of promising differentials: {percent_significant:.2f}%")


for diff in diffs_zero:
    print(f"dx: {diff.dx}, dy: {diff.dy}, dz: {diff.dz}, pro: {diff.wt}")



# Check power of 2
def is_power_of_two(n):
    return n == 0 or (n > 0 and (n & (n - 1)) == 0)


dx_count = 0
dy_count = 0
dz_count = 0
for idx, dx, dy, dz in significant_differentials_with_zero_weight:
    print(idx)
    print(f"dx {dx} {is_power_of_two(dx)}")
    if is_power_of_two(dx):
        dx_count += 1
    if is_power_of_two(dy):
        dy_count += 1
    if is_power_of_two(dz):
        dz_count += 1
    print(f"dx {dy} {is_power_of_two(dy)}")
    zero = False
    if dz == 0:
        zero = True
    print(f"dx {dz} {is_power_of_two(dz)} {zero}")
    
print(f"The number of dx power of 2 is: {dx_count}")
print(f"The number of dy power of 2 is: {dy_count}")
print(f"The number of dz power of 2 or 0 is: {dz_count}")


df = pd.DataFrame(sampled_non_significant_differentials, columns=["dx", "dy", "dz", "wt"])
df['experiments'] = experiment_results_non_significant



piv_tb = df.pivot_table(values="experiments", index="dy", columns="dx", aggfunc=="mean")

plt.figure(figsize=(10,8))
sns.heatmap(piv_tb, annot=True, cmap="viridis", fmt=".1f")
plt.show()

exr = np.array(experiment_results_non_significant)
unique_diffs, counts = np.unique(exr,return_counts=True)
print(unique_diffs, counts)
plt.figure(figsize=(8,10))

heatmap = sns.heatmap([counts],  annot=False, fmt="d", cmap="viridis", xticklabels=unique_diffs, yticklabels=["Count"])
plt.xlabel("Vals")
plt.ylabel('Freq')
plt.show()



uniform_data = np.random.rand(10, 12)
uniform_data

res_arr = np.array(experiment_results_non_significant)
res_arr
print(len(res_arr))

non_sig_dz_values = [diff.dz for diff in sampled_non_significant_differentials]
print(len(non_sig_dz_values))
print(len(experiment_results_non_significant))

# joined = np.stack((non_sig_dz_values, experiment_results_non_significant), axis=1)
# print(joined)
non_sig_dz_values_arr = np.array(non_sig_dz_values)

non_sig_dz_values_reshaped = non_sig_dz_values_arr.reshape(-1, 1)
non_sig_dz_values_tiled = np.tile(non_sig_dz_values_reshaped, 2)

non_sig_dz_values
myset = set(non_sig_dz_values)
print(myset)
# combined = np.stack((non_sig_dz_values_arr,experiment_results_non_significant), axis = 1)
# combined = np.vstack((experiment_results_non_significant, non_sig_dz_values_reshaped))
# print(combined)

joined = np.asarray(joined).reshape(373,1)
ax = sns.heatmap(joined, linewidth=0.5)

plt.show()


uniform_data = [non_sig_dz_values, res_arr]
ax = sns.heatmap(uniform_data, linewidth=0.5)
plt.show()

myarray = np.asarray(sampled_non_significant_differentials)
myarray
