import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.compute as pc
from tail_calibration.flucast_utils import build_evaluation_df, uniquely_identified


s3 = fs.S3FileSystem(anonymous=True)

mo = ds.dataset(
    "cdcepi-flusight-forecast-hub/model-output/",
    filesystem=s3,
    format="parquet"
)

with s3.open_input_file("cdcepi-flusight-forecast-hub/target-data/time-series.csv") as f:
    time_series = pd.read_csv(f)

oracle_asof = time_series.loc[
    time_series["target"].isin(['wk inc flu hosp']) 
].copy()
oracle_asof[["target_end_date", "as_of"]] = oracle_asof[["target_end_date", "as_of"]].apply(pd.to_datetime)

with s3.open_input_file("cdcepi-flusight-forecast-hub/target-data/oracle-output.csv") as f:
    oracle = pd.read_csv(f)

forecasts = mo.to_table(
    filter=(
        (pc.field("target").isin(['wk inc flu hosp'])) &
        (pc.field("output_type") == "quantile")  & 
        (pc.field("horizon") >= 0) &
        (pc.field("model_id").isin(["FluSight-ensemble"]))
    )
).to_pandas()
forecasts[["target_end_date", "reference_date"]] = forecasts[["target_end_date", "reference_date"]].apply(pd.to_datetime)

date_ranges = {
    ("2023-10-14", "2024-04-27"): "2024-04-27",
    ("2024-10-14", "2025-06-21"): "2025-07-23"
}


result = build_evaluation_df(forecasts, oracle_asof, date_ranges)
assert uniquely_identified(result, by=["target", "reference_date", "location", "horizon", "model_id"])

# ── Filter (your existing code) ───────────────────────────────────────────────
result_mask = (
    (result['target']   == 'wk inc flu hosp') &
    (result['location'] == 'US') &
    (result['horizon']  == 0) &
    (result['model_id'] == 'FluSight-ensemble')
)
result_filtered = result[result_mask].copy()
result_filtered['cover_80'] = (
    (result_filtered['observation'] >= result_filtered['quantile_0.1']) &
    (result_filtered['observation'] <= result_filtered['quantile_0.9'])
)

# ── Sort by date if you have one (adjust column name as needed) ───────────────
date_col = 'reference_date'   # <-- change if your date column has a different name
if date_col in result_filtered.columns:
    result_filtered = result_filtered.sort_values(date_col)
    labels = result_filtered[date_col].astype(str).tolist()
else:
    labels = list(range(len(result_filtered)))

observations = result_filtered['observation'].tolist()
covered      = result_filtered['cover_80'].tolist()

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_HIT  = '#185FA5'   # blue  — covered
COLOR_MISS = '#993C1D'   # coral — not covered

bar_colors = [COLOR_HIT if c else COLOR_MISS for c in covered]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
ax.set_facecolor('white')

bars = ax.bar(range(len(observations)), observations, color=bar_colors,
              width=0.75, zorder=2)

# ── Axes formatting ───────────────────────────────────────────────────────────
tick_every = max(1, len(labels) // 20)   # show ~20 labels max
ax.set_xticks(range(0, len(labels), tick_every))
ax.set_xticklabels(
    [labels[i] for i in range(0, len(labels), tick_every)], #type:ignore
    rotation=45, ha='right', fontsize=9, color='#555'
)
ax.set_ylabel('Weekly Incidence of Flu Hospitalisations', fontsize=13, color='#555')
ax.tick_params(axis='y', colors='#555')
ax.grid(axis='y', color='#111111', alpha=0.08, linewidth=0.6, zorder=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlim(-0.6, len(observations) - 0.4)

# ── Coverage rate annotation ──────────────────────────────────────────────────
n_covered = sum(covered)
n_total   = len(covered)
rate      = n_covered / n_total
ax.text(0.01, 0.97,
        f'Empirical 80% coverage: {n_covered}/{n_total} $\\approx {rate:.2f}$',
        transform=ax.transAxes, fontsize=13, color='#333',
        va='top', ha='left')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=COLOR_HIT,  label='Within 80% interval'),
    mpatches.Patch(facecolor=COLOR_MISS, label='Outside 80% interval'),
]
ax.legend(handles=legend_handles, fontsize=10, frameon=False,
          loc='upper right', bbox_to_anchor=(1.0, 1.02))

plt.tight_layout()
plt.savefig('./images/flu_coverage_bar.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()