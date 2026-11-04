"""Classical regression baseline using the PCA eigenvalues of each stream.

Fits the target (the raw file's pid, log10(M*) for prog_mass) as a function of the three eigenvalues (λ1 ≥ λ2 ≥ λ3) of each stream's spatial covariance matrix.
The fit is least squares on the train split (with polynomial features if --degree > 1), and --variant auto keeps the variant with the lower val MSE.
Two feature variants:

  linear   : features = [√λ1, √λ2, √λ3]  (linear in eigenvalue space)
  power_law: features = [log10(√λ1), log10(√λ2), log10(√λ3)]
             → equivalent to M* ∝ λ1^a · λ2^b · λ3^c

Output:
  {output_dir}/results/loss_history.csv   (same format as run_average.py)
  {output_dir}/results/fit_summary.txt    (variant, formula, all MSEs)
  with --eval_input_raw: the same two files plus predictions.npz in {output_dir}/shared_test<eval_dataset_size>/ instead of results/

Usage:
    python src/eigenvalue_regression.py \\
        --input_raw raw_data/prog_mass_reg_dataset_10000.h5 \\
        --output_dir prog_mass_3d_x_y_z_10000/eigenvalue_regression/run_0

    # force a specific variant:
    python src/eigenvalue_regression.py ... --variant power_law

    # all sizes from repo root:
    for size in 100 1000 10000; do
        python src/eigenvalue_regression.py \\
            --input_raw raw_data/prog_mass_reg_dataset_${size}.h5 \\
            --output_dir prog_mass_3d_x_y_z_${size}/eigenvalue_regression/run_0
    done
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# ---------------------------------------------------------------------------
# Eigenvalue extraction (mirrors visualization._extract_sigmas)
# ---------------------------------------------------------------------------

def _extract_sigmas(cart_ds: np.ndarray) -> np.ndarray:
    """Computes √λ of each stream's covariance matrix (of the x, y, z of its particles).

    Returns an (N, 3) array, largest first, in the coordinates' units (kpc).

    `cart_ds`: the raw Cartesian_data, (N, n_particles, n_features) with x, y, z first.
    """
    N = cart_ds.shape[0]
    sigmas = np.zeros((N, 3))
    for i in range(N):
        if i % 1000 == 0:
            print(f"  Extracting eigenvalues {i}/{N}...", flush=True)
        pts = cart_ds[i, :, 0:3]
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[np.argsort(eigvals)[::-1]]
        sigmas[i] = np.sqrt(np.maximum(eigvals, 0))
    return sigmas


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _make_pipeline(degree: int) -> Pipeline:
    """Linear regression, after PolynomialFeatures if degree > 1."""
    steps = []
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("lr", LinearRegression()))
    return Pipeline(steps)


def fit_variant(X_train, y_train, X_val, y_val, X_test, y_test,
                degree: int, label: str):
    """Fits one feature variant on the train split and scores it on all three splits.

    Returns (train_mse, val_mse, test_mse, coef, intercept, pipe), the MSEs in raw target units; also prints them and, for degree 1, the fitted formula.

    `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`: the features and targets of each split.
    `degree`: the polynomial degree of the features (1 = plain linear).
    `label`: 'linear' or 'power_law', for the printout.
    """
    pipe = _make_pipeline(degree)
    pipe.fit(X_train, y_train)

    train_mse = float(np.mean((pipe.predict(X_train) - y_train) ** 2))
    val_mse   = float(np.mean((pipe.predict(X_val)   - y_val)   ** 2))
    test_mse  = float(np.mean((pipe.predict(X_test)  - y_test)  ** 2))

    lr = pipe.named_steps["lr"]
    coef, intercept = lr.coef_, float(lr.intercept_)

    print(f"\n  [{label}]  train={train_mse:.6f}  val={val_mse:.6f}  test={test_mse:.6f}")
    if degree == 1:
        feat_names = (["log10(√λ1)", "log10(√λ2)", "log10(√λ3)"]
                      if label == "power_law" else ["√λ1", "√λ2", "√λ3"])
        terms = "  +  ".join(f"{c:.4f}·{n}" for c, n in zip(coef, feat_names))
        print(f"  formula: log10(M*) = {terms}  +  {intercept:.4f}")

    return train_mse, val_mse, test_mse, coef, intercept, pipe


def _formula_str(variant: str, coef, intercept: float, degree: int) -> str:
    """The fitted formula for fit_summary.txt, e.g. 'log10(M*) = a·√λ1 + b·√λ2 + c·√λ3 + d'; for degree > 1 only the degree and the intercept."""
    if degree > 1:
        return f"polynomial degree={degree}, intercept={intercept:.4f}"
    if variant == "power_law":
        names = ["log10(√λ1)", "log10(√λ2)", "log10(√λ3)"]
        terms = " + ".join(f"{c:.4f}·{n}" for c, n in zip(coef, names))
        return f"log10(M*) = {terms} + {intercept:.4f}"
    else:
        names = ["√λ1", "√λ2", "√λ3"]
        terms = " + ".join(f"{c:.4f}·{n}" for c, n in zip(coef, names))
        return f"log10(M*) = {terms} + {intercept:.4f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Reads the input file, output folder, variant and shared-test options from the command line.

    Returns the argparse namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_raw", required=True,
                    help="Path to raw HDF5 dataset (prog_mass_reg_dataset_*.h5)")
    ap.add_argument("--output_dir", required=True,
                    help="Directory for results (loss_history.csv and fit_summary.txt)")
    ap.add_argument("--variant", choices=["linear", "power_law", "auto"], default="auto",
                    help="Feature variant: linear (√λ), power_law (log10(√λ)), "
                         "auto=pick best val MSE (default)")
    ap.add_argument("--degree", type=int, default=1,
                    help="Polynomial degree for features (default: 1)")
    ap.add_argument("--eval_input_raw", default=None,
                    help="If set, score the already-selected (best-val) variant against "
                         "this file's test split instead of --input_raw's own. Requires "
                         "--eval_dataset_size. Writes to "
                         "shared_test<eval_dataset_size>/loss_history.csv.")
    ap.add_argument("--eval_dataset_size", type=int, default=None,
                    help="Dataset size label for --eval_input_raw's output subdir naming.")
    return ap.parse_args()


def main():
    """Fits the baseline on --input_raw and writes its results.

    Returns nothing; writes loss_history.csv and fit_summary.txt (plus predictions.npz with --eval_input_raw), see the module docstring.
    """
    args = parse_args()

    h5_path = Path(args.input_raw)
    if not h5_path.exists():
        sys.exit(f"Error: {h5_path} not found.")

    # --- Load data ---
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        cart_ds = f["Cartesian_data"][:]
        y       = f["pid"][:].ravel()    # the target, log10(M*/M_sun) for prog_mass

    N = len(y)
    train_end = int(N * 0.6)
    val_end   = int(N * 0.8)

    # --- Extract eigenvalues ---
    sigmas = _extract_sigmas(cart_ds)    # (N, 3)

    # --- Split ---
    sig_tr, y_tr = sigmas[:train_end],        y[:train_end]
    sig_va, y_va = sigmas[train_end:val_end], y[train_end:val_end]
    sig_te, y_te = sigmas[val_end:],          y[val_end:]

    log_sig_tr = np.log10(np.maximum(sig_tr, 1e-9))
    log_sig_va = np.log10(np.maximum(sig_va, 1e-9))
    log_sig_te = np.log10(np.maximum(sig_te, 1e-9))

    # --- Average baseline for reference ---
    mean_train   = y_tr.mean()
    avg_test_mse = float(np.mean((y_te - mean_train) ** 2))
    print(f"\nAverage baseline  test MSE: {avg_test_mse:.6f}")

    # --- Fit variants ---
    r_lin = r_pow = None

    if args.variant in ("linear", "auto"):
        print("\n--- Linear features [√λ1, √λ2, √λ3] ---")
        r_lin = fit_variant(sig_tr, y_tr, sig_va, y_va, sig_te, y_te,
                            args.degree, "linear")

    if args.variant in ("power_law", "auto"):
        print("\n--- Power-law features [log10(√λ1), log10(√λ2), log10(√λ3)] ---")
        r_pow = fit_variant(log_sig_tr, y_tr, log_sig_va, y_va, log_sig_te, y_te,
                            args.degree, "power_law")

    # --- Select best ---
    if args.variant == "auto":
        if r_lin[1] <= r_pow[1]:
            best_variant, best = "linear", r_lin
        else:
            best_variant, best = "power_law", r_pow
    elif args.variant == "linear":
        best_variant, best = "linear", r_lin
    else:
        best_variant, best = "power_law", r_pow

    train_mse, val_mse, test_mse, coef, intercept, pipe = best

    # --- Optional: score the already-selected (best-val) variant on a shared test set instead of --input_raw's own. ---
    # The variant and its coefficients stay as fitted above; only the reported test MSE (and the average baseline's) changes.
    out_dir = Path(args.output_dir) / "results"
    if args.eval_input_raw:
        if args.eval_dataset_size is None:
            sys.exit("Error: --eval_dataset_size is required with --eval_input_raw")
        eval_path = Path(args.eval_input_raw)
        if not eval_path.exists():
            sys.exit(f"Error: {eval_path} not found.")
        with h5py.File(eval_path, "r") as f:
            eval_val_end = int(f["pid"].shape[0] * 0.8)
            eval_cart = f["Cartesian_data"][eval_val_end:]
            eval_y = f["pid"][eval_val_end:].ravel()
        print(f"\nExtracting eigenvalues for shared test set ({eval_path}, {eval_cart.shape[0]} streams)...")
        eval_sigmas = _extract_sigmas(eval_cart)
        eval_X = (np.log10(np.maximum(eval_sigmas, 1e-9)) if best_variant == "power_law"
                  else eval_sigmas)
        eval_pred = pipe.predict(eval_X)
        test_mse = float(np.mean((eval_pred - eval_y) ** 2))
        avg_test_mse = float(np.mean((eval_y - mean_train) ** 2))
        out_dir = Path(args.output_dir) / f"shared_test{args.eval_dataset_size}"

    improvement = (avg_test_mse - test_mse) / avg_test_mse * 100

    print(f"\n==> Saved variant: {best_variant}")
    print(f"    train={train_mse:.6f}  val={val_mse:.6f}  test={test_mse:.6f}")
    print(f"    vs average baseline: {avg_test_mse:.6f}  (improvement: {improvement:.1f}%)")

    # --- Save results ---
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "Epoch":      [1],
        "Train_Loss": [train_mse],
        "Val_Loss":   [val_mse],
        "Test_Loss":  [test_mse],
    }).to_csv(out_dir / "loss_history.csv", index=False)

    if args.eval_input_raw:
        # Per-stream predictions in raw units, under the same keys as OmniLearned's outputs__streams_*.npz (pid is the raw target), read downstream by compute_bootstrap_shared_test.py.
        np.savez(out_dir / "predictions.npz", prediction=eval_pred, pid=eval_y)

    formula = _formula_str(best_variant, coef, intercept, args.degree)
    coef_str = "  ".join(f"λ{i+1}={c:.4f}" for i, c in enumerate(coef))
    summary = (
        f"variant: {best_variant}\n"
        f"degree:  {args.degree}\n"
        f"formula: {formula}\n"
        f"coefficients: {coef_str}  intercept={intercept:.4f}\n"
        f"train_mse: {train_mse:.6f}\n"
        f"val_mse:   {val_mse:.6f}\n"
        f"test_mse:  {test_mse:.6f}\n"
        f"avg_baseline_test_mse: {avg_test_mse:.6f}\n"
        f"improvement_vs_avg: {improvement:.1f}%\n"
    )
    (out_dir / "fit_summary.txt").write_text(summary)

    print(f"\nSaved {out_dir / 'loss_history.csv'}")
    print(f"Saved {out_dir / 'fit_summary.txt'}")


if __name__ == "__main__":
    main()
