# mlops-drift-drill

소득 예측 모델(UCI Adult, 연소득 50K 초과 여부)에 **신규 채널 고객이 서서히 늘어나는 드리프트**를 흘려 넣고, 서빙 → 드리프트 감시 → 판단 → 재학습/판정 기준 재계산까지 직접 만들어 본 실습 코드입니다.

글로 정리한 시리즈: [MLOps 실습 시리즈](https://chayujin-kr.github.io/posts/mlops-drift-00-intro/)

## 핵심 결론

- 드리프트 경보는 "입력이 바뀌었다"는 신호이지 "모델이 틀린다"는 신호가 아니다. 예측 1 비율이 0%가 된 구간의 정확도는 0.99였다.
- 경보 후 재학습 효과는 AUC 0.950 → 0.952. 판정 기준을 50% → 10%로 바꾼 효과는 순이익 764만 → 1,216만 원.
- 그래서 파이프라인은 경보 뒤에 **판단 단계**(insufficient > degraded > recalibrate > ok)를 둔다.

## 구조

| 경로 | 내용 |
|---|---|
| `common.py`, `train.py`, `service.py` | 데이터 로더, 학습, BentoML 서비스 (원본 14열 → 원핫 102열) |
| `stream.py`, `stream2.py` | 드리프트 스트림 생성기 (2번은 고객 ID 포함) |
| `drift.py`, `drift_exp.py` | Evidently 윈도우별 드리프트 판정, 규칙 비교 실험 |
| `exp_threshold.py`, `q5.py` | 판정 기준 실험, 지연 라벨로 구모델 vs 신모델 비교 |
| `mlflow/` | kind 클러스터용 MLflow (SQLite + PVC, MinIO 아티팩트) |
| `pipeline/` | KFP 파이프라인, 판정 로직(`judge_logic.py`)과 로컬 테스트 |
| `PIPELINE_DESIGN.md` | 운영 파이프라인 설계 문서 |
| `blog/` | 실습 정리 글 원본 |

## 실행 환경

- Python 3.10, BentoML 1.1.10, Evidently 0.5.0 (옛 API), scikit-learn 1.7.2
- kind 클러스터 + Kubeflow Pipelines 2.4 standalone + MinIO
- KFP SDK 2.9.0, kfp-kubernetes 1.3.0, MLflow 2.17.2

## 실행 순서 (요약)

```bash
# 1. 데이터: UCI에서 받아 data/ 에 풀기 (openml 서버가 느리면 이쪽이 빠름)
curl -L -o data/adult.zip https://archive.ics.uci.edu/static/public/2/adult.zip

# 2. 로컬 실습 (Q1~Q5)
python train.py && bentoml serve service:svc --port 3000 --api-workers 1
python stream.py && python drift.py

# 3. MLflow 설치
docker build -t local/mlflow-server:2.17.2 mlflow/ && kind load docker-image local/mlflow-server:2.17.2 --name <cluster>
kubectl apply -f mlflow/mlflow.yaml   # minio-cred Secret 은 kubeflow 의 mlpipeline-minio-artifact 에서 복사

# 4. KFP 파이프라인
cd pipeline
python bootstrap.py                  # MinIO 업로드 + 챔피언 등록 (port-forward 5000, 9000 필요)
./run_tests.sh && ./run_tests.sh cases   # 판정 로직 로컬 검증
docker build -t local/drift-pipeline:v2 . && kind load docker-image local/drift-pipeline:v2 --name <cluster>
python pipeline.py && python run.py natural-1101 date=2026-11-01   # port-forward 8080 필요
```

접속 정보는 코드에 넣지 않고, 실행 시 K8s Secret에서 읽습니다.
