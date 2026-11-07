import json, pickle, time, uuid
from pathlib import Path
import bentoml, pandas as pd
from bentoml.io import JSON

COLUMN_LIST = pickle.load(open(Path(__file__).parent / "column_list.pkl", "rb"))
LOG = Path(__file__).parent / "logs" / "preds_v2.jsonl"
THRESHOLD = 0.10                                   # 실험 D 결과: 드리프트 구간 순이익 최대
model_ref = bentoml.sklearn.get("income_drill:latest")
MODEL_TAG = str(model_ref.tag)
runner = model_ref.to_runner()
svc = bentoml.Service("income_drill", runners=[runner])


@svc.api(input=JSON(), output=JSON())
def predict(row: dict) -> list:
    row = dict(row)
    customer_id = row.pop("customer_id", None)        # 실무 고객 ID: 피처가 아니므로 분리
    X = pd.get_dummies(pd.DataFrame([row])).reindex(columns=COLUMN_LIST, fill_value=0)
    proba = float(runner.predict_proba.run(X)[0, 1])   # >50K 확률
    pred = int(proba >= THRESHOLD)

    # 지연 라벨은 customer_id 로 연결, 모델 교체 후에도 재평가 가능하도록 확률·버전·기준 저장
    record = {
        "customer_id": customer_id,
        "row": row,
        "prediction": pred,
        "probability": proba,
        "model_version": MODEL_TAG,
        "threshold": THRESHOLD,
        "request_id": uuid.uuid4().hex,
        "request_time": time.time()
    }

    with LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return [pred]
