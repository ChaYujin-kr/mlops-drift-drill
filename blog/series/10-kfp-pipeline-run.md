# [MLOps 실습 시리즈] 10. KFP로 판단 파이프라인 돌려보기

7편에서 만든 판정 규칙을 Kubeflow Pipelines로 자동화했다. 핵심은 **경보가 울리면 바로 재학습하지 않고, 판정 결과에 따라 네 갈래로 나뉘는 것**이다.

## 파이프라인 구조

```
load_data ──→ check_drift
                 ├ 드리프트 없음 ──→ 알림
                 └ 드리프트 ──→ judge
                                 ├ degraded     ──→ retrain ──→ 알림
                                 ├ recalibrate  ──→ recompute_threshold ──→ 알림
                                 ├ insufficient ──→ 알림 (라벨 추가 요청)
                                 └ ok           ──→ 알림
```

| 컴포넌트 | 하는 일 |
|---|---|
| `load_data` | MinIO에서 그날의 서빙 로그와 소량 라벨을 불러온다 |
| `check_drift` | 중요 피처 3개(나이, 혼인 상태, 직업) 중 2개 이상 PSI > 0.1이면 경보 |
| `judge` | 챔피언 모델로 소량 라벨을 평가해 판정 (7편의 규칙) |
| `retrain` | 재학습 후 MLflow에 후보(challenger)로 등록 |
| `recompute_threshold` | 순이익이 가장 큰 판정 기준을 제안 |
| `notify` | 알림 (실습에선 로그 출력) |

## 분기는 `dsl.If` / `Elif` / `Else`

```python
with dsl.If(drift.outputs["Output"] == True):
    verdict = judge(...)
    with dsl.If(verdict.outputs["Output"] == "degraded"):
        new = retrain(...)
    with dsl.Elif(verdict.outputs["Output"] == "recalibrate"):
        th = recompute_threshold(...)
    with dsl.Elif(verdict.outputs["Output"] == "insufficient"):
        notify(...)
    with dsl.Else():
        notify(...)
```

처음엔 컴파일부터 실패했다.
```
AttributeError: The task has multiple outputs. Please reference the output by its name.
```
반환값(`-> str`)과 `Output[Metrics]`를 함께 가진 컴포넌트는 출력이 두 개라서, `.output` 대신 `.outputs["Output"]`으로 이름을 지정해야 한다.

## 다섯 갈래를 모두 돌려봤다

| 실행 | 결과 |
|---|---|
| 11-01 (자연 상태) | `ok` · 드리프트는 있으나 성능 정상 |
| 11-06 (자연 상태) | `insufficient` · 양성 20명 미만, 라벨 추가 요청 |
| 판정을 `degraded`로 강제 | 재학습 → challenger v2 등록 (홀드아웃 AUC 0.943 → 0.950) |
| 판정을 `recalibrate`로 강제 | 판정 기준 0.10 → 0.05 제안 |
| 드리프트 기준을 불가능하게 | `no-drift` |

자연 데이터로는 `degraded`와 `recalibrate`가 나오지 않아서, 테스트 전용 파라미터 `override_decision`으로 판정을 강제해 **분기 배선**만 확인했다. 판정 로직 자체는 7편에서 로컬로 검증했다.

## 자동으로 하지 않은 것

일부러 사람 손에 남겨둔 단계가 있다.

- **챔피언 교체:** 재학습 모델은 challenger로 등록만 한다. 섀도우 배포와 A/B를 거쳐 사람이 승인한다.
- **판정 기준 변경:** 새 기준은 모델 태그(`threshold_proposed`)에 제안만 남긴다. 비용 구조는 마케팅, 재무와 합의할 일이다.

파이프라인이 끝난 뒤 MLflow 상태:
```
aliases: champion = v1, challenger = v2
v1 tags: threshold = 0.1, threshold_proposed = 0.05
```

## 신경 쓴 것

- **판정 코드는 복사하지 않았다.** 로컬에서 검증한 `judge_logic.py`를 이미지에 넣어 컴포넌트가 그대로 불러온다. 로컬과 파이프라인의 판정이 어긋날 일이 없다.
- **비밀번호는 파라미터로 넘기지 않았다.** `kfp-kubernetes`로 K8s Secret을 환경 변수로 연결했다.
- **재학습 평가는 홀드아웃으로 했다.** 학습에 쓰지 않은 고객 20%로 챔피언과 후보를 같은 조건에서 비교했다.

## 한 줄 정리

자동화의 핵심은 모든 걸 자동으로 하는 게 아니라, 어디까지 자동으로 하고 어디서 사람이 판단할지 정하는 것이다.
