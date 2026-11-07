import pickle, bentoml, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import split

COST, GAIN = 2_000, 150_000
P_STAR = COST / (COST + GAIN)

ref, _, _ = split()
cols = pickle.load(open("column_list.pkl", "rb"))
enc = lambda d: pd.get_dummies(d).reindex(columns=cols, fill_value=0)
X_ref, y_ref = enc(ref.drop(columns="class")), (ref["class"] == ">50K").astype(int)

log = pd.read_json("logs/preds.jsonl", lines=True)
labels = pd.concat([pd.read_csv(f"labels/w{t}.csv") for t in range(15)], ignore_index=True)
log["y"] = (labels["class"] == ">50K").astype(int)
log["w"] = log.index // 200
X_log = enc(log[ref.drop(columns="class").columns])

old = bentoml.sklearn.load_model("income_drill:latest")
log["p_old"] = old.predict_proba(X_log)[:, 1]          # 원본 14열 로그 → 확률 재계산


def profit(y, p, th):
    send = p >= th
    return int((send & (y == 1)).sum() * GAIN - send.sum() * COST), int(send.sum())


def table(title, seg, pcol):
    print(f"\n[{title}] {len(seg)}건, 실제 >50K {seg.y.sum()}명, AUC={roc_auc_score(seg.y, seg[pcol]):.3f}")
    for name, th in [("50%", .5), ("30%", .3), ("10%", .1), (f"p*={P_STAR:.1%}", P_STAR), ("전원", -1)]:
        pr, n = profit(seg.y.values, seg[pcol].values, th)
        tp = int(((seg[pcol] >= th) & (seg.y == 1)).sum())
        print(f"  {name:<9} 발송 {n:>4}명  잡은 >50K {tp:>3}/{seg.y.sum()}  순이익 {pr/1e4:>7,.0f}만")


normal, drift = log[log.w <= 3], log[log.w >= 8]
print("=== 실험 D: 판정 기준만 바꾸기 (모델 그대로) ===")
table("정상 w0-3", normal, "p_old")
table("드리프트 w8-14", drift, "p_old")

# === 가중치 재학습: 라벨 없는 서빙 입력만으로 ===
# 1) 도메인 분류기: "이 샘플은 ref에서 왔나(0), 최근 서빙(w8-14)에서 왔나(1)?"
X_new = X_log[log.w >= 8]
dom = LogisticRegression(max_iter=2000).fit(
    pd.concat([X_ref, X_new]).astype(float), np.r_[np.zeros(len(X_ref)), np.ones(len(X_new))])
p_new = dom.predict_proba(X_ref.astype(float))[:, 1]
# 2) 가중치 = p(new|x)/p(ref|x) × (n_ref/n_new)  → 새 분포에 흔한 ref 샘플일수록 큼
wts = np.clip(p_new / (1 - p_new) * len(X_ref) / len(X_new), 0.05, 20)
ess = wts.sum() ** 2 / (wts ** 2).sum()
print(f"\n=== 가중치 재학습 (새 라벨 0건) ===\n가중치 min={wts.min():.2f} median={np.median(wts):.2f} max={wts.max():.2f}, "
      f"유효 샘플 수 {ess:.0f}/{len(wts)}")
top = ref.assign(wt=wts).sort_values("wt", ascending=False)
print("가중치 top3:", top[["age", "hours-per-week", "marital-status", "wt"]].head(3).to_dict("records"))
print("가중치 low3:", top[["age", "hours-per-week", "marital-status", "wt"]].tail(3).to_dict("records"))
# 3) 같은 ref 라벨로, sample_weight만 줘서 재학습
new = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(X_ref, y_ref, sample_weight=wts)
log["p_new"] = new.predict_proba(X_log)[:, 1]
table("드리프트 w8-14 · 가중치 모델", log[log.w >= 8], "p_new")
table("정상 w0-3 · 가중치 모델", log[log.w <= 3], "p_new")
