"""
Regression using Artificial Neural Networks
Urban Entropy Prediction

Predicts normalised street network entropy based on
graph topology and circuity features.

Usage
-----
    python src/neural_network/train_ann.py
"""

# Import standard Libraries
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, FIGURES_DIR

OUT_DIR = FIGURES_DIR / 'neural_network'
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.float_format', '{:.4f}'.format)
sns.set(rc={'figure.figsize': (10, 10)})
print("imports ok")


# ── Load Data ──────────────────────────────────────────────────────
data = pd.read_csv(PROCESSED_DIR / 'patch_training_data_full.csv')
pd.options.display.max_columns = None

print(data)
print(data.info())

for colname, col in data.items():
    print(colname, "min_val", col.min(), "max_val", col.max())

data.describe()


# ── Select features and target ─────────────────────────────────────
FEATURES = [
    'n_4way', 'n_3way', 'n_deadend',
    'proportion_4way', 'proportion_3way', 'proportion_deadend',
    'mean_degree', 'mean_edge_length', 'total_edge_length',
    'meshedness', 'intersection_density', 'street_density',
    'circuity',
]

data_features = data[FEATURES + ['entropy_normalised']].dropna()

data_numerical = data_features[['entropy_normalised', 'circuity',
                                 'proportion_4way', 'meshedness', 'mean_degree']]
sns.pairplot(data_numerical)
plt.savefig(OUT_DIR / 'pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved -> pairplot.png")


# ── Prepare Data ───────────────────────────────────────────────────

# declare features
X = data_features[FEATURES]

# Load and instantiate a StandardScaler
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()

# Apply the scaler to our X-features
X_scaled = scalerX.fit_transform(X)

print(X_scaled.shape)

# declare regression target
import numpy as np
y = data_features['entropy_normalised'].to_numpy()
y = y.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
scalerY = MinMaxScaler()

# entropy_normalised is already between 0 and 1, MinMax locks it in that range
y_scaled = scalerY.fit_transform(y)

print(y_scaled.shape)


# ── Split into Train and Test ──────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=21
)

print("TRAIN", "input", X_train.shape, "output", y_train.shape)
print("TEST",  "input", X_test.shape,  "output", y_test.shape)


# ── Build Model ────────────────────────────────────────────────────
# Regression between 0 and 1:
#   activation = relu for hidden layers / sigmoid for final layer
#   loss = mean squared error
#   optimizer = adam
#   input = 13 features
#   output = 1 value (entropy_normalised)

model = tf.keras.models.Sequential()
n_cols = X_scaled.shape[1]

model.add(tf.keras.layers.Dense(4, input_shape=(n_cols,), activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()


# ── Train Model ────────────────────────────────────────────────────
history = model.fit(X_train, y_train, epochs=500, validation_split=0.2)

# plot loss curve
final_train_mse = history.history['loss'][-1]
final_val_mse   = history.history['val_loss'][-1]
plt.plot(history.history['loss'],     label=f'Train  (final MSE={final_train_mse:.4f})')
plt.plot(history.history['val_loss'], label=f'Val    (final MSE={final_val_mse:.4f})')
plt.title('Neural Network — Loss over 500 Epochs')
plt.ylabel('MSE (mean squared error)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig(OUT_DIR / 'loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved -> loss_curve.png")


# ── Evaluate on Test Data ──────────────────────────────────────────
loss_test = model.evaluate(X_test, y_test)
print('mse_test:', loss_test)


# ── Plot Predictions ───────────────────────────────────────────────
y_pred  = scalerY.inverse_transform(model.predict(X_test))
y_truth = scalerY.inverse_transform(y_test)

from sklearn.metrics import r2_score, mean_squared_error
r2   = r2_score(y_truth, y_pred)
rmse = np.sqrt(mean_squared_error(y_truth, y_pred))

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_truth, y_pred, alpha=0.4, color='steelblue', label='test patches')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='perfect prediction')
ax.set_ylim((0, 1))
ax.set_xlim((0, 1))
ax.set_xlabel('Actual entropy (entropy_normalised)', fontsize=12)
ax.set_ylabel('Predicted entropy (entropy_normalised)', fontsize=12)
ax.set_title('Neural Network — Predicted vs Actual Entropy', fontsize=13)
ax.text(0.05, 0.92, f'R² = {r2:.3f}\nRMSE = {rmse:.4f}',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / 'scatter_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved -> scatter_pred_vs_actual.png  (R²={r2:.3f}, RMSE={rmse:.4f})")


def plot_comparison(x_val, pred, truth, xlab, ylab, xlabel_full):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(x_val, truth, color='tomato',    label='Actual entropy',
             linestyle='None', marker='o', markersize=5)
    ax1.plot(x_val, pred,  color='steelblue', label='Predicted entropy',
             linestyle='None', marker='o', markersize=4, alpha=0.6)
    ax1.set_xlabel(xlabel_full, fontsize=12)
    ax1.set_ylabel('entropy_normalised', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.set_title(f'Prediction Comparison — {xlabel_full}', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'comparison_{xlab}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> comparison_{xlab}.png")


# plot predictions against the two most important features (from SHAP analysis)
circuity_test    = scalerX.inverse_transform(X_test)[:, FEATURES.index('circuity')]
prop_4way_test   = scalerX.inverse_transform(X_test)[:, FEATURES.index('proportion_4way')]

plot_comparison(circuity_test,  y_pred, y_truth, 'circuity',        'entropy_normalised', 'Circuity (network distance / straight-line distance)')
plot_comparison(prop_4way_test, y_pred, y_truth, 'proportion_4way', 'entropy_normalised', 'Proportion of 4-way intersections')

error = y_pred - y_truth
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [entropy units]')
plt.ylabel('Count')
plt.title('Error Distribution')
plt.savefig(OUT_DIR / 'error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved -> error_distribution.png")


# ── Per-city error plot ────────────────────────────────────────────
# recover city codes for the test samples using the same split
df_full = pd.read_csv(PROCESSED_DIR / 'patch_training_data_full.csv')
df_model = df_full[FEATURES + ['entropy_normalised', 'city_code']].dropna(
    subset=FEATURES + ['entropy_normalised']
)
_, df_test = train_test_split(df_model, test_size=0.2, random_state=21)
city_codes = df_test['city_code'].values

abs_errors = np.abs(y_pred.flatten() - y_truth.flatten())
city_df = pd.DataFrame({'city': city_codes, 'abs_error': abs_errors})

# median absolute error per city, sorted worst to best
city_summary = city_df.groupby('city')['abs_error'].median().sort_values(ascending=False)
n_cities = len(city_summary)

# split into two panels: worst half and best half
mid = n_cities // 2
worst = city_summary.iloc[:mid]
best  = city_summary.iloc[mid:]

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, max(8, mid * 0.35)))
fig.suptitle('Neural Network — Median Absolute Error per City', fontsize=14)

worst.plot(kind='barh', ax=ax_left, color='tomato')
ax_left.invert_yaxis()
ax_left.set_xlabel('Median absolute error')
ax_left.set_title(f'Worst predicted ({mid} cities)')
ax_left.axvline(abs_errors.mean(), color='black', linestyle='--', linewidth=1, label='overall mean')
ax_left.legend(fontsize=9)

best.plot(kind='barh', ax=ax_right, color='steelblue')
ax_right.invert_yaxis()
ax_right.set_xlabel('Median absolute error')
ax_right.set_title(f'Best predicted ({n_cities - mid} cities)')
ax_right.axvline(abs_errors.mean(), color='black', linestyle='--', linewidth=1, label='overall mean')
ax_right.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / 'error_by_city.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved -> error_by_city.png")
