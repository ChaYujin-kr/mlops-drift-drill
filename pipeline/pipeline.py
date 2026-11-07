"""드리프트 경보 → 판단 → (재학습 | 판정 기준 재계산 | 보류 | 정상) KFP 파이프라인.

설계: ../PIPELINE_DESIGN.md 10장
판정 로직: judge_logic.py (이미지 local/drift-pipeline:v2 에 포함, 로컬 검증본과 같은 파일)
"""
from kfp import compiler, dsl, kubernetes
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

IMAGE = "local/drift-pipeline:v2"
MINIO = "http://minio-service.kubeflow.svc.cluster.local:9000"
MLFLOW = "http://mlflow.mlflow.svc.cluster.local:5000"
MINIO_SECRET = {"accesskey": "AWS_ACCESS_KEY_ID", "secretkey": "AWS_SECRET_ACCESS_KEY"}


@dsl.component(base_image=IMAGE)
def load_data(bucket: str, date: str, minio_endpoint: str,
              window: Output[Dataset], labels: Output[Dataset]):
    """그날의 서빙 로그(원본 입력 + 고객 ID)와 소량 라벨을 불러온다."""
    import json
    import boto3, pandas as pd

    s3 = boto3.client("s3", endpoint_url=minio_endpoint)
    body = s3.get_object(Bucket=bucket, Key=f"serving/date={date}/log.jsonl")["Body"].read().decode()
    log = pd.DataFrame([json.loads(l) for l in body.splitlines()])
    X = pd.DataFrame(log["row"].tolist()).assign(customer_id=log["customer_id"])
    X.to_csv(window.path, index=False)
    s3.download_file(bucket, f"labels_sample/date={date}/labels.csv", labels.path)
    window.metadata.update({"rows": len(X), "source": f"s3://{bucket}/serving/date={date}/"})


@dsl.component(base_image=IMAGE)
def check_drift(bucket: str, minio_endpoint: str, window: Input[Dataset],
                important_features: str, min_drifted: int, drift_metrics: Output[Metrics]) -> bool:
    """중요 피처의 PSI 를 레퍼런스와 비교. min_drifted 개 이상 드리프트면 True."""
    import io
    import boto3, pandas as pd
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    s3 = boto3.client("s3", endpoint_url=minio_endpoint)
    ref = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key="reference/reference.csv")["Body"].read()))
    cur = pd.read_csv(window.path)
    feats = [f.strip() for f in important_features.split(",")]

    report = Report(metrics=[DataDriftPreset(columns=feats, stattest="psi", stattest_threshold=0.1)])
    report.run(reference_data=ref[feats], current_data=cur[feats])
    table = report.as_dict()["metrics"][1]["result"]["drift_by_columns"]

    n_drifted = sum(v["drift_detected"] for v in table.values())
    for col, v in table.items():
        drift_metrics.log_metric(f"psi_{col.replace('-', '_')}", float(v["drift_score"]))
    drift_metrics.log_metric("n_drifted", n_drifted)
    return n_drifted >= min_drifted


@dsl.component(base_image=IMAGE)
def judge(window: Input[Dataset], labels: Input[Dataset], mlflow_uri: str,
          override_decision: str, judge_metrics: Output[Metrics]) -> str:
    """챔피언으로 소량 라벨을 평가해 insufficient / degraded / recalibrate / ok 판정."""
    import math
    import mlflow, pandas as pd
    from mlflow import MlflowClient
    from judge_logic import decide

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    champ = client.get_model_version_by_alias("income-classifier", "champion")
    baseline = client.get_run(champ.run_id).data.metrics
    model = mlflow.sklearn.load_model("models:/income-classifier@champion")

    win, lab = pd.read_csv(window.path), pd.read_csv(labels.path)
    ev = win.drop_duplicates("customer_id").merge(lab.drop_duplicates("customer_id"), on="customer_id")
    y = (ev["class"] == ">50K").astype(int).values
    p = model.predict_proba(ev[list(model.feature_names_in_)])[:, 1]

    decision, metrics = decide(y, p, float(champ.tags["threshold"]), baseline)
    for k, v in metrics.items():
        if not math.isnan(float(v)):                   # 계산 불가(NaN)는 기록하지 않음
            judge_metrics.log_metric(k, float(v))
    judge_metrics.log_metric("champion_version", int(champ.version))

    if override_decision:                              # 분기 배선 테스트 전용
        print(f"override: {decision} -> {override_decision}")
        decision = override_decision
    return decision


@dsl.component(base_image=IMAGE)
def recompute_threshold(window: Input[Dataset], labels: Input[Dataset], mlflow_uri: str,
                        cost: float, gain: float, threshold_out: Output[Artifact],
                        th_metrics: Output[Metrics]) -> str:
    """후보 기준(p*, 5%, 10%, 20%, 30%) 중 소량 라벨에서 순이익이 가장 큰 값을 챔피언 태그로 제안."""
    import json
    import mlflow, pandas as pd
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    champ = client.get_model_version_by_alias("income-classifier", "champion")
    model = mlflow.sklearn.load_model("models:/income-classifier@champion")

    win, lab = pd.read_csv(window.path), pd.read_csv(labels.path)
    ev = win.drop_duplicates("customer_id").merge(lab.drop_duplicates("customer_id"), on="customer_id")
    y = (ev["class"] == ">50K").astype(int).values
    p = model.predict_proba(ev[list(model.feature_names_in_)])[:, 1]

    p_star = round(cost / (cost + gain), 4)
    profit = {t: float(((p >= t) & (y == 1)).sum() * gain - (p >= t).sum() * cost)
              for t in sorted({p_star, 0.05, 0.10, 0.20, 0.30})}
    best = max(profit, key=profit.get)
    for t, v in profit.items():
        th_metrics.log_metric(f"profit_at_{t:g}".replace(".", "_"), v)
    th_metrics.log_metric("best_threshold", best)

    current = float(champ.tags["threshold"])
    json.dump({"current": current, "proposed": best, "profit": profit}, open(threshold_out.path, "w"))
    # 바로 바꾸지 않고 제안만 남김: 판정 기준 변경은 사람이 승인 (설계 문서 7장)
    client.set_model_version_tag("income-classifier", champ.version, "threshold_proposed", str(best))
    return f"현재 기준 {current} → 제안 {best} (소량 라벨 순이익 {profit[current]:,.0f} → {profit[best]:,.0f}원)"


@dsl.component(base_image=IMAGE)
def retrain(bucket: str, minio_endpoint: str, mlflow_uri: str,
            model_out: Output[Model], train_metrics: Output[Metrics]) -> str:
    """레퍼런스 + 지연 라벨 이력으로 재학습, 챔피언과 같은 홀드아웃에서 비교 후 challenger 로 등록."""
    import io
    import boto3, joblib, mlflow, pandas as pd
    from mlflow import MlflowClient
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    s3 = boto3.client("s3", endpoint_url=minio_endpoint)
    read = lambda k: pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=k)["Body"].read()))
    ref, hist = read("reference/reference.csv"), read("history/labeled_stream1.csv")
    hist_train, hist_test = train_test_split(hist, test_size=0.2, random_state=0)   # 홀드아웃: 학습에 안 쓴 고객

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    champ_ver = client.get_model_version_by_alias("income-classifier", "champion")
    champion = mlflow.sklearn.load_model("models:/income-classifier@champion")
    feats = list(champion.feature_names_in_)
    cat = [c for c in feats if ref[c].dtype == object]

    train = pd.concat([ref, hist_train])
    challenger = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")),
        ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)),
    ]).fit(train[feats], (train["class"] == ">50K").astype(int))

    y = (hist_test["class"] == ">50K").astype(int)
    auc_champ = roc_auc_score(y, champion.predict_proba(hist_test[feats])[:, 1])
    auc_chal = roc_auc_score(y, challenger.predict_proba(hist_test[feats])[:, 1])
    for k, v in {"auc_champion": auc_champ, "auc_challenger": auc_chal,
                 "n_train": len(train), "n_holdout": len(hist_test)}.items():
        train_metrics.log_metric(k, float(v))
    joblib.dump(challenger, model_out.path)

    mlflow.set_experiment("income-classifier")
    with mlflow.start_run(run_name="retrain-kfp"):
        mlflow.log_params({"train_data": "reference + history[80%]", "holdout": "history[20%]",
                           "threshold": champ_ver.tags["threshold"]})
        mlflow.log_metrics({"holdout_auc": auc_chal, "holdout_auc_champion": auc_champ})
        info = mlflow.sklearn.log_model(challenger, "model", registered_model_name="income-classifier",
                                        input_example=train[feats].head(2))
    version = info.registered_model_version
    client.set_registered_model_alias("income-classifier", "challenger", version)
    client.set_model_version_tag("income-classifier", version, "threshold", champ_ver.tags["threshold"])
    # 승격(champion 교체)은 섀도우·A/B 후 수동 (설계 문서 8–9장)
    return f"challenger v{version} 등록 · 홀드아웃 AUC 챔피언 {auc_champ:.3f} vs 후보 {auc_chal:.3f}"


@dsl.component(base_image=IMAGE)
def notify(stage: str, date: str, detail: str):
    """실습에선 로그 출력으로 알림을 대신한다."""
    print(f"[ALERT] income-classifier · {date} · {stage} · {detail}")


@dsl.pipeline(name="drift-judge-retrain",
              description="드리프트 경보 → 소량 라벨 판단 → 재학습 / 판정 기준 재계산 / 보류 / 정상")
def drift_pipeline(date: str = "2026-11-01", bucket: str = "drift-drill",
                   important_features: str = "age,marital-status,occupation", min_drifted: int = 2,
                   cost: float = 2000.0, gain: float = 150000.0, override_decision: str = ""):
    data = load_data(bucket=bucket, date=date, minio_endpoint=MINIO)
    drift = check_drift(bucket=bucket, minio_endpoint=MINIO, window=data.outputs["window"],
                        important_features=important_features, min_drifted=min_drifted)
    for task in (data, drift):
        kubernetes.use_secret_as_env(task, secret_name="mlpipeline-minio-artifact", secret_key_to_env=MINIO_SECRET)

    with dsl.If(drift.outputs["Output"] == True, name="drift-detected"):
        verdict = judge(window=data.outputs["window"], labels=data.outputs["labels"],
                        mlflow_uri=MLFLOW, override_decision=override_decision)

        with dsl.If(verdict.outputs["Output"] == "degraded", name="degraded"):
            new = retrain(bucket=bucket, minio_endpoint=MINIO, mlflow_uri=MLFLOW)
            kubernetes.use_secret_as_env(new, secret_name="mlpipeline-minio-artifact", secret_key_to_env=MINIO_SECRET)
            notify(stage="degraded → 재학습", date=date, detail=new.outputs["Output"])
        with dsl.Elif(verdict.outputs["Output"] == "recalibrate", name="recalibrate"):
            th = recompute_threshold(window=data.outputs["window"], labels=data.outputs["labels"],
                                     mlflow_uri=MLFLOW, cost=cost, gain=gain)
            notify(stage="recalibrate → 판정 기준 제안", date=date, detail=th.outputs["Output"])
        with dsl.Elif(verdict.outputs["Output"] == "insufficient", name="insufficient"):
            notify(stage="insufficient → 판정 보류", date=date, detail="소량 라벨 양성 20명 미만: 라벨 추가 요청")
        with dsl.Else(name="ok"):
            notify(stage="ok", date=date, detail="드리프트는 있으나 성능 정상: 조치 없음")

    with dsl.Else(name="no-drift"):
        notify(stage="no-drift", date=date, detail="중요 피처 드리프트 없음")


if __name__ == "__main__":
    compiler.Compiler().compile(drift_pipeline, "drift_pipeline.yaml")
    print("compiled → drift_pipeline.yaml")
