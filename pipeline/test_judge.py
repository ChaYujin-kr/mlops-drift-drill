"""judge_logic.decide 로컬 검증: 날짜별 소량 라벨 200건 × 챔피언 확률."""
import json
import boto3, mlflow, numpy as np, pandas as pd
from mlflow import MlflowClient
from judge_logic import decide

mlflow.set_tracking_uri("http://localhost:5000")
c = MlflowClient()
champ = c.get_model_version_by_alias("income-classifier", "champion")
baseline = c.get_run(champ.run_id).data.metrics
threshold = float(champ.tags["threshold"])
model = mlflow.sklearn.load_model("models:/income-classifier@champion")
s3 = boto3.client("s3", endpoint_url="http://localhost:9000")
get = lambda k: s3.get_object(Bucket="drift-drill", Key=k)["Body"].read().decode()

print(f"champion v{champ.version}, threshold={threshold}, baseline_auc={baseline['baseline_auc']:.3f}, "
      f"baseline_profit/인={baseline['baseline_profit_per_customer']:,.0f}원\n")
for d in range(1, 7):
    date = f"2026-11-{d:02d}"
    log = pd.DataFrame([json.loads(l) for l in get(f"serving/date={date}/log.jsonl").splitlines()])
    X = pd.DataFrame(log["row"].tolist()).assign(customer_id=log["customer_id"])
    lab = pd.read_csv(pd.io.common.StringIO(get(f"labels_sample/date={date}/labels.csv")))
    ev = X.drop_duplicates("customer_id").merge(lab.drop_duplicates("customer_id"), on="customer_id")
    y = (ev["class"] == ">50K").astype(int).values
    p = model.predict_proba(ev.drop(columns=["customer_id", "class"]))[:, 1]
    decision, m = decide(y, p, threshold, baseline)
    print(f"{date}  >50K={y.mean():5.1%}  AUC={m['auc']:.3f} (drop {m['auc_drop']:+.3f})  "
          f"효율={m['efficiency']:.1%} (기준 {m['baseline_efficiency']:.1%}, drop {m['efficiency_drop_pct']:+.1%})  "
          f"양성 {m['n_positives']:>2}명  보정비율 {m['calibration_ratio']:.2f}  → {decision}")
