import pickle, bentoml, numpy as np, pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from common import split

ref, _, _ = split()
X_ref = ref.drop(columns="class")
log = pd.read_json("logs/preds.jsonl", lines=True)

# 모델이 본 102열 중요도 → 원본 14열로 합산
model = bentoml.sklearn.load_model("income_drill:latest")
cols = pickle.load(open("column_list.pkl", "rb"))
imp = pd.Series(model.feature_importances_, index=cols)
raw_imp = imp.groupby(lambda c: next(r for r in X_ref.columns if c == r or c.startswith(r + "_"))).sum()
top = raw_imp.sort_values(ascending=False)
print("원본 열 중요도 top5:", ", ".join(f"{k}={v:.2f}" for k, v in top.head(5).items()), "\n")
TOP3 = list(top.index[:3])


def drifted(cur, columns):
    r = Report(metrics=[DataDriftPreset(columns=columns, stattest="psi", stattest_threshold=0.1, drift_share=0.5)])
    r.run(reference_data=X_ref, current_data=cur[X_ref.columns])
    m = r.as_dict()["metrics"]
    return m[0]["result"], [c for c, v in m[1]["result"]["drift_by_columns"].items() if v["drift_detected"]]


all14 = list(X_ref.columns)
no_nc = [c for c in all14 if c != "native-country"]
print(f"{'win':<4}{'p':>5} | {'A:14열 50%':<12}| {'B:13열(nc제외) 50%':<20}| C:top3 중 2개+ ")
for t in range(15):
    cur = log.iloc[t*200:(t+1)*200]
    a, _ = drifted(cur, all14)
    b, _ = drifted(cur, no_nc)
    _, dc = drifted(cur, TOP3)
    p = min(1, max(0, (t-3)/8))
    fa = lambda d: f"{d['number_of_drifted_columns']}/{d['number_of_columns']} {'ALERT' if d['dataset_drift'] else '':5}"
    print(f"w{t:<3}{p:>5.2f} | {fa(a):<12}| {fa(b):<20}| {len(dc)}/3 {'ALERT' if len(dc) >= 2 else ''} {dc}")

# 라벨 엿보기 (Q5 예고): 0% 예측은 고장인가?
print("\n[라벨 엿보기] 실제 >50K 비율 vs 모델 예측 1 비율 vs 재현율(실제 >50K 중 맞힌 비율)")
for t in [0, 5, 8, 14]:
    y = (pd.read_csv(f"labels/w{t}.csv")["class"] == ">50K").astype(int).values
    yhat = log.prediction.iloc[t*200:(t+1)*200].values
    rec = (yhat[y == 1] == 1).mean() if y.sum() else float("nan")
    print(f"  w{t:<2} 실제={y.mean():.3f} ({y.sum()}명)  예측={yhat.mean():.3f}  재현율={rec:.2f}  정확도={(y == yhat).mean():.2f}")
