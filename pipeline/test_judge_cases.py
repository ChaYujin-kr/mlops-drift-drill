"""judge_logic.decide 를 일부러 망가뜨린 경우로 검증 (11-01, 11-06 데이터)."""
import json
import boto3, mlflow, numpy as np, pandas as pd
from mlflow import MlflowClient
from judge_logic import decide

mlflow.set_tracking_uri("http://localhost:5000")
c = MlflowClient()
champ = c.get_model_version_by_alias("income-classifier", "champion")
baseline, threshold = c.get_run(champ.run_id).data.metrics, float(champ.tags["threshold"])
model = mlflow.sklearn.load_model("models:/income-classifier@champion")
s3 = boto3.client("s3", endpoint_url="http://localhost:9000")
get = lambda k: s3.get_object(Bucket="drift-drill", Key=k)["Body"].read().decode()


def load(date):
    log = pd.DataFrame([json.loads(l) for l in get(f"serving/date={date}/log.jsonl").splitlines()])
    X = pd.DataFrame(log["row"].tolist()).assign(customer_id=log["customer_id"])
    lab = pd.read_csv(pd.io.common.StringIO(get(f"labels_sample/date={date}/labels.csv")))
    ev = X.drop_duplicates("customer_id").merge(lab.drop_duplicates("customer_id"), on="customer_id")
    return (ev["class"] == ">50K").astype(int).values, model.predict_proba(ev.drop(columns=["customer_id", "class"]))[:, 1]


rng = np.random.default_rng(0)
y, p = load("2026-11-01")
y6, p6 = load("2026-11-06")
cases = {
    "정상 11-01": (y, p),
    "모델 고장 (확률 무작위)": (y, rng.permutation(p)),
    "개념 드리프트 (중간 확률 고객 절반이 >50K로)": (np.where((p >= .05) & (p < .3) & (rng.random(len(y)) < .5), 1, y), p),
    "양성 부족 11-06": (y6, p6),
}
for name, (yy, pp) in cases.items():
    d, m = decide(yy, pp, threshold, baseline)
    print(f"{name:<30} → {d:<12} 양성 {m['n_positives']:>2}명  AUC drop {m['auc_drop']:+.3f}  효율 drop {m['efficiency_drop_pct']:+.1%}  보정비율 {m['calibration_ratio']:.2f}")
