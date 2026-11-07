"""1회성 준비: MinIO에 원본 데이터 업로드 + 챔피언(전처리 포함 Pipeline) MLflow 등록."""
import io, json, os, sys
import boto3, mlflow, numpy as np, pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "..")
from common import split

BUCKET, COST, GAIN, THRESHOLD = "drift-drill", 2_000, 150_000, 0.10
NUM = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
s3 = boto3.client("s3", endpoint_url="http://localhost:9000")
if BUCKET not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
    s3.create_bucket(Bucket=BUCKET)
put = lambda key, body: s3.put_object(Bucket=BUCKET, Key=key, Body=body)

# 1) 레퍼런스 (학습 + 드리프트 기준)
ref, _, _ = split()
ref = ref.assign(customer_id=[f"C{i:06d}" for i in ref.index])
put("reference/reference.csv", ref.to_csv(index=False))

# 2) 지연 라벨 이력: 첫 스트림 3,000건 (구 로그는 ID가 없어 순서로 연결)
s1 = pd.read_json("../logs/preds.jsonl", lines=True)[NUM + CAT]
lab1 = pd.concat([pd.read_csv(f"../labels/w{t}.csv", index_col=0) for t in range(15)])
s1 = s1.assign(**{"class": lab1["class"].values, "customer_id": [f"C{i:06d}" for i in lab1.index]})
put("history/labeled_stream1.csv", s1.to_csv(index=False))

# 3) 서빙 로그와 소량 라벨: 두 번째 스트림 w6~w11 → 날짜 파티션
logs = open("../logs/preds_v2.jsonl").read().splitlines()
for i, t in enumerate(range(6, 12)):
    date = f"2026-11-{i + 1:02d}"
    put(f"serving/date={date}/log.jsonl", "\n".join(logs[i * 200:(i + 1) * 200]) + "\n")
    put(f"labels_sample/date={date}/labels.csv", open(f"../labels2/w{t}.csv").read())
print("uploaded:", sorted({o["Key"].split("/")[0] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}),
      "| serving dates:", 6)

# 4) 챔피언: 전처리 포함 Pipeline
model = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")),
    ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)),
]).fit(ref[NUM + CAT], (ref["class"] == ">50K").astype(int))

# 기준 시점 지표: 정상 구간(첫 스트림 w0~3, 800건)
base = s1.iloc[:800]
y, p = (base["class"] == ">50K").astype(int).values, model.predict_proba(base[NUM + CAT])[:, 1]
send = p >= THRESHOLD
metrics = {"baseline_auc": roc_auc_score(y, p),
           "baseline_profit_per_customer": ((send & (y == 1)).sum() * GAIN - send.sum() * COST) / len(y),
           "baseline_positive_rate": y.mean(), "baseline_pred_rate": send.mean()}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("income-classifier")
with mlflow.start_run(run_name="champion-bootstrap") as run:
    mlflow.log_params({"threshold": THRESHOLD, "cost_fp": COST, "gain_tp": GAIN,
                       "train_data": "reference/reference.csv", "baseline_data": "history/labeled_stream1.csv[:800]"})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model", registered_model_name="income-classifier",
                             input_example=ref[NUM + CAT].head(2))
c = MlflowClient()
v = c.get_latest_versions("income-classifier")[0].version
c.set_registered_model_alias("income-classifier", "champion", v)
c.set_model_version_tag("income-classifier", v, "threshold", str(THRESHOLD))
print(f"champion = income-classifier v{v}", {k: round(x, 3) for k, x in metrics.items()})
