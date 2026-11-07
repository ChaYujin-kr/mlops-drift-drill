"""Q5: 고객 ID로 지연 라벨 연결 → 구모델 vs 신모델(지연 라벨로 재학습) AUC·순이익 비교."""
import pickle, bentoml, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from common import split

COST, GAIN = 2_000, 150_000
ref, _, _ = split()
FEATS = list(ref.drop(columns="class").columns)
cols = pickle.load(open("column_list.pkl", "rb"))
enc = lambda d: pd.get_dummies(d[FEATS]).reindex(columns=cols, fill_value=0)

# 1) 첫 스트림 + 3개월 뒤 도착한 라벨 (구 로그엔 ID가 없어 순서로 연결: 레거시)
s1 = pd.read_json("logs/preds.jsonl", lines=True)
lab1 = pd.concat([pd.read_csv(f"labels/w{t}.csv", index_col=0) for t in range(15)])
s1["class"], s1["customer_id"] = lab1["class"].values, [f"C{i:06d}" for i in lab1.index]

# 2) 신모델: ref + 지연 라벨 3,000건으로 재학습
train = pd.concat([ref, s1[FEATS + ["class"]]])
new = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(
    enc(train), (train["class"] == ">50K").astype(int))

# 3) 새 스트림 로그 ↔ 라벨을 customer_id 로 조인
log = pd.read_json("logs/preds_v2.jsonl", lines=True)
log = pd.concat([log.drop(columns="row"), pd.DataFrame(log["row"].tolist())], axis=1)
lab2 = pd.concat([pd.read_csv(f"labels2/w{t}.csv") for t in range(6, 12)])
n_log = len(log)
ev = log.drop_duplicates("customer_id").merge(lab2.drop_duplicates("customer_id"), on="customer_id", how="inner")
leak = ev.customer_id.isin(s1.customer_id)
print(f"로그 {n_log}건 → 고객ID 조인 {len(ev)}건 → 학습에 쓰인 고객 {leak.sum()}명 제외 → 평가 {(~leak).sum()}건")
ev = ev[~leak]
y = (ev["class"] == ">50K").astype(int).values
print(f"모델 버전(로그): {ev.model_version.unique()}, 실제 >50K {y.sum()}명\n")

p_old = ev["probability"].values                        # 서빙 때 로그에 남긴 확률 그대로
p_new = new.predict_proba(enc(ev))[:, 1]


def profit(p, th):
    send = p >= th
    return (send & (y == 1)).sum() * GAIN - send.sum() * COST, send.sum(), (send & (y == 1)).sum()


print(f"{'':<16}{'AUC':>6} | {'50% 기준':^22} | {'10% 기준':^22} | {'p*=1.3%':^22}")
for name, p in [("구모델", p_old), ("신모델(재학습)", p_new)]:
    cells = [f"{n:>4}명 {tp:>2}/{y.sum()} {pr/1e4:>5,.0f}만" for pr, n, tp in (profit(p, th) for th in (.5, .1, COST/(COST+GAIN)))]
    print(f"{name:<14}{roc_auc_score(y, p):>8.3f} | " + " | ".join(f"{c:^22}" for c in cells))
print("\n(칸 형식: 발송 인원, 잡은 >50K / 전체 >50K, 순이익)")
