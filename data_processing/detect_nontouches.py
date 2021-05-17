from parse_touchpad_data import TouchpadData
from sklearn.mixture import BayesianGaussianMixture

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# Load Data
legal = np.array(TouchpadData('touchpad_capture/real_data/legal').data)
illegal = np.array(TouchpadData('touchpad_capture/real_data/illegal').data)
nontouch = np.array(TouchpadData('touchpad_capture/real_data/nontouches').data)


# Create DataFrames for features
legal_features = pd.DataFrame({
    'vars': np.var(legal, axis=(1,2)), 
    'max_values': np.max(legal, axis=(1,2))
    })

illegal_features = pd.DataFrame({
    'vars': np.var(illegal, axis=(1,2)), 
    'max_values': np.max(illegal, axis=(1,2))
    })

nontouch_features = pd.DataFrame({
    'vars': np.var(nontouch, axis=(1,2)), 
    'max_values': np.max(nontouch, axis=(1,2))
    })


# Train Gaussian Mixture Model
bgm = BayesianGaussianMixture().fit(nontouch_features)

# Calculate logarithmic likelihoods
legal_scores = pd.Series(bgm.score_samples(legal_features))
illegal_scores = pd.Series(bgm.score_samples(illegal_features))

# Eliminate high scores 
legal_scores = legal_scores[legal_scores < -500]
illegal_scores = illegal_scores[illegal_scores < -500]

# Save the rest as PNG
for index in legal_scores.index:
    plt.imsave('data_processing/out/legal/{}.png'.format(index), legal[index], cmap='gray', vmin=-10, vmax=245)
for index in illegal_scores.index:
    plt.imsave('data_processing/out/illegal/{}.png'.format(index), illegal[index], cmap='gray', vmin=-10, vmax=245)