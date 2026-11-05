# %% [markdown]
# # Fashion-MNIST — Parts 0 and 1
#
# Two classifiers on the same images: a linear classifier (SGD with logistic
# loss) and a random forest. The linear model learns a weight for each pixel;
# the forest votes across decision trees. Both work on a CPU.
#
# Run from the repo with `python notebooks/part0_part1.py`, or open this `.py`
# file as a notebook in VS Code/Jupytext. Install `requirements.txt` first.
# The four official Fashion-MNIST archives and fixed splits are already in the
# repo. No results or figure files are written by this notebook.

# %%
from pathlib import Path
import os
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Works both as a .py script and from a notebook kernel in the repo.
start_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in (start_dir, *start_dir.parents) if (p / "splits/indices.npz").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Run from the repo containing splits/indices.npz")
sys.path.insert(0, str(ROOT))
from src.fashion_mnist import CLASS_NAMES, load_images, load_labels
from src.splits import load_splits
from src.timing import hardware_info

SEED = 708
print("Hardware:", hardware_info())

# %% [markdown]
# ## Part 0 — fixed data
#
# Fashion-MNIST has 60,000 official training and 10,000 test images (28×28,
# greyscale, 10 classes). The 50k train / 10k validation split and all subsets
# were stratified with seed 708 and are saved in `splits/` with checksums.
# We use the permitted 10k training subset to keep this run quick. Test *labels*
# are read only to verify the existing split manifest; test *images* are never
# opened. For clustering later, use the saved 3k/1k/1k indices for every
# representation, and never use labels for clustering distances or assignments.

# %%
raw = ROOT / "data/raw"
y_all = load_labels(raw, "train")
y_test_for_check = load_labels(raw, "test")   # checks frozen indices, NOT evaluation
splits, manifest = load_splits(ROOT / "splits", y_all, y_test_for_check)
assert manifest["seed"] == SEED
for name, idx in splits.items():
    print(f"{name:20s} {len(idx):6,d}   per class: {manifest['indices'][name]['class_counts']}")
train_ids, val_ids = splits["fit_train"], splits["validation"]
y_train, y_val = y_all[train_ids], y_all[val_ids]
images = load_images(raw, "train")  # never load_images(raw, "test") here

# Downsample 28x28 -> 14x14 (196 values) to keep both models CPU-friendly.
# Extract ONCE, then reuse these two arrays for every model and setting.
def make_features(pixels):
    x = pixels.astype(np.float32) / 255.0
    return x.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196)

features = {}
extract_time = {}
for name, ids in (("train", train_ids), ("validation", val_ids)):
    t = perf_counter()
    features[name] = make_features(images[ids])
    extract_time[name] = perf_counter() - t
X_train, X_val = features["train"], features["validation"]
print(f"Feature extraction: train {extract_time['train']:.3f}s, val {extract_time['validation']:.3f}s (cached in memory)")

# %% [markdown]
# ## Part 1.1 and 1.4 — models and complexity
#
# Logistic SGD is a simple linear baseline: it learns pixel weights and uses
# regularisation `alpha` (larger = simpler). A random forest can learn nonlinear
# pixel interactions; `max_depth` limits tree complexity. We vary one complexity
# setting per model and compare train with validation accuracy. Higher training
# accuracy without a matching validation gain suggests overfitting.

# %%
def fit_and_measure(model, label):
    t = perf_counter()
    model.fit(X_train, y_train)
    train_sec = perf_counter() - t
    train_acc = accuracy_score(y_train, model.predict(X_train))
    t = perf_counter()
    val_pred = model.predict(X_val)
    pred_sec = perf_counter() - t
    return dict(label=label, model=model, train=train_acc,
                val=accuracy_score(y_val, val_pred), preds=val_pred,
                fit_sec=train_sec, predict_sec=pred_sec)

linear_runs = []
for alpha in (0.00001, 0.0001, 0.001, 0.01):
    model = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=300,
                          tol=1e-3, random_state=SEED)
    linear_runs.append(fit_and_measure(model, alpha))

forest_runs = []
for depth in (4, 8, 12, None):
    model = RandomForestClassifier(n_estimators=50, max_depth=depth,
                                   min_samples_leaf=2, random_state=SEED,
                                   n_jobs=min(os.cpu_count() or 1, 4))
    forest_runs.append(fit_and_measure(model, depth))

for name, runs in (("Logistic SGD (alpha)", linear_runs), ("Random forest (max_depth)", forest_runs)):
    print(f"\n{name:32s} setting     train    val    gap    fit(s)  inference(s)")
    for r in runs:
        print(f"{'':32s} {str(r['label']):8s} {r['train']:.3f}   {r['val']:.3f}  "
              f"{r['train'] - r['val']:+.3f}   {r['fit_sec']:.2f}    {r['predict_sec']:.3f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, runs, title in zip(axs, (linear_runs, forest_runs), ("Logistic SGD: alpha", "Forest: max depth")):
    xx = np.arange(len(runs))
    ax.plot(xx, [r["train"] for r in runs], "o-", label="train")
    ax.plot(xx, [r["val"] for r in runs], "s-", label="validation")
    ax.set_xticks(xx, [str(r["label"]) for r in runs])
    ax.set(xlabel="Complexity setting", ylabel="Accuracy", title=title, ylim=(0, 1))
    ax.legend()
fig.tight_layout()
plt.show()

# Select using validation, never the official test set.
linear = max(linear_runs, key=lambda r: r["val"])
forest = max(forest_runs, key=lambda r: r["val"])
print(f"Chosen settings: linear alpha={linear['label']}, forest depth={forest['label']}")
for name, runs in (("Linear", linear_runs), ("Forest", forest_runs)):
    biggest_gap = max(runs, key=lambda r: r["train"] - r["val"])
    print(f"{name}: largest train–validation gap {biggest_gap['train'] - biggest_gap['val']:.3f} "
          f"at setting {biggest_gap['label']}. A bigger gap suggests more overfitting.")

# %% [markdown]
# ## Part 1.2 — class-level results
#
# All rows below use the *same 10,000 validation images*. Support is the number
# of actual examples of that class. Recall is the fraction correctly found;
# precision is the fraction of predictions for that class that were correct.

# %%
reports = {}
for title, result in (("Linear", linear), ("Forest", forest)):
    print(f"{title} overall validation accuracy: {result['val']:.4f}")
    reports[title] = classification_report(y_val, result["preds"], labels=np.arange(10),
                                            target_names=CLASS_NAMES, zero_division=0, output_dict=True)
print(f"{'Class':14s} {'support':>7s} {'lin prec':>8s} {'lin recall':>10s} {'lin F1':>7s} "
      f"{'RF prec':>8s} {'RF recall':>9s} {'RF F1':>7s}")
for cls in CLASS_NAMES:
    a, b = reports["Linear"][cls], reports["Forest"][cls]
    print(f"{cls:14s} {int(a['support']):7d} {a['precision']:8.3f} {a['recall']:10.3f} {a['f1-score']:7.3f} "
          f"{b['precision']:8.3f} {b['recall']:9.3f} {b['f1-score']:7.3f}")
for title, result in (("Linear", linear), ("Forest", forest)):
    recalls = reports[title]
    ranked = sorted(CLASS_NAMES, key=lambda c: recalls[c]["recall"])
    cm = confusion_matrix(y_val, result["preds"], labels=np.arange(10))
    np.fill_diagonal(cm, 0)
    if cm.sum():
        true, predicted = np.unravel_index(np.argmax(cm), cm.shape)
        mistake = f"Most common error: {CLASS_NAMES[true]} -> {CLASS_NAMES[predicted]} ({cm[true, predicted]} images)."
    else:
        mistake = "No mistakes on this validation set."
    print(f"{title}: easiest by recall = {', '.join(ranked[-2:])}; hardest = {', '.join(ranked[:2])}. {mistake}")

# %% [markdown]
# ## Part 1.3 — same-image error analysis
#
# Check where both models agree and where only one gets the label right. Fine
# differences between tops (shirt/T-shirt/coat) can be lost by 2×2 pooling;
# footwear often has a clearer shape. The actual hardest classes are printed
# above rather than assumed in advance.

# %%
a_ok = linear["preds"] == y_val
b_ok = forest["preds"] == y_val
groups = {
    "both right": np.flatnonzero(a_ok & b_ok),
    "linear only": np.flatnonzero(a_ok & ~b_ok),
    "forest only": np.flatnonzero(~a_ok & b_ok),
    "both wrong": np.flatnonzero(~a_ok & ~b_ok),
}
print("Correctness on the same validation images:", {k: len(v) for k, v in groups.items()})
rng = np.random.default_rng(SEED)
fig, axs = plt.subplots(4, 3, figsize=(9, 11))
for row, (group_name, positions) in enumerate(groups.items()):
    picked = rng.choice(positions, size=min(3, len(positions)), replace=False)
    for col, ax in enumerate(axs[row]):
        ax.axis("off")
        if col >= len(picked):
            continue
        i = int(picked[col])
        ax.imshow(images[val_ids[i]], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{group_name} | row {val_ids[i]}\n"
                     f"true: {CLASS_NAMES[y_val[i]]}\n"
                     f"linear: {CLASS_NAMES[linear['preds'][i]]} | RF: {CLASS_NAMES[forest['preds'][i]]}",
                     fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Part 1.5 — decision threshold and costs
#
# Binary task: **is it footwear?** (Sandal, Sneaker, Ankle boot). For a simple
# stock check, missing footwear (false negative) costs 4 units; falsely
# flagging non-footwear (false positive) costs 1. We sum the *linear model's*
# three footwear probabilities. Changing the threshold trades FP against FN.
# The choice below uses validation only; test data is still held back.

# %%
footwear = np.isin(y_val, [5, 7, 9])
footwear_cols = np.isin(linear["model"].classes_, [5, 7, 9])
score = linear["model"].predict_proba(X_val)[:, footwear_cols].sum(axis=1)
thresholds = np.arange(0.05, 1.0, 0.05)
tradeoff = []
for threshold in thresholds:
    positive = score >= threshold
    fp = int(np.sum(positive & ~footwear))
    fn = int(np.sum(~positive & footwear))
    tradeoff.append((threshold, fp, fn, fp + 4 * fn))
print(f"Validation: {footwear.sum()} footwear / {len(footwear)} total. Cost = FP + 4×FN")
print("Threshold   FP   FN   Total cost")
for t, fp, fn, cost in tradeoff:
    print(f"{t:8.2f} {fp:5d} {fn:5d} {cost:10d}")
best_t, fp, fn, cost = min(tradeoff, key=lambda r: r[3])
print(f"Lowest validation cost at threshold {best_t:.2f}: FP={fp}, FN={fn}, cost={cost}.")
print("Raising the threshold tends to reduce FP but increase FN; the cost weights decide which matters more.")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(thresholds, [r[1] for r in tradeoff], "o-", label="false positives")
ax.plot(thresholds, [r[2] for r in tradeoff], "o-", label="false negatives")
ax.axvline(best_t, color="black", linestyle="--", label=f"min cost: {best_t:.2f}")
ax.set(xlabel="Footwear threshold", ylabel="Validation images (count)",
       title="Binary errors as threshold changes")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Notes for the viva:** linear SGD learns ten sets of pixel weights by
# minimising logistic loss; alpha penalises large weights. The forest fits
# many trees to random bootstrap samples, splits on pixel values, and averages
# their votes; depth controls its flexibility. Increasing complexity can help
# training accuracy while hurting validation if the model overfits. This
# notebook only uses training and validation data; test evaluation comes last.
