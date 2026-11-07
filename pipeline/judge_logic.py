"""judge 컴포넌트의 판정 로직. KFP에 올리기 전에 로컬에서 test_judge.py 로 검증한다."""
import numpy as np
from sklearn.metrics import roc_auc_score

MIN_POSITIVES = 20                                         # 소량 라벨 최소 양성 수 (2026-11-08 결정)


def profit_per_customer(y, p, threshold, cost, gain):
    """판정 기준 이상이면 안내 발송. (잡은 >50K × 이익 − 발송 수 × 비용) / 전체 인원"""
    send = p >= threshold
    return ((send & (y == 1)).sum() * gain - send.sum() * cost) / len(y)


def decide(y: np.ndarray, p: np.ndarray, threshold: float, baseline: dict,
           cost: float = 2_000, gain: float = 150_000) -> tuple[str, dict]:
    """소량 라벨(y)과 챔피언 확률(p)로 성능 저하 여부를 판정한다.

    baseline: MLflow 챔피언 run 의 지표
        {"baseline_auc", "baseline_profit_per_customer", "baseline_positive_rate", ...}
    반환: ("degraded" | "ok", 판정 근거 지표 dict — KFP Metrics 용이라 값은 모두 숫자)

    설계 문서 6장 (2026-11-08 확정, 같은 날 순이익 규칙 수정):
        아래 중 하나라도 해당하면 저하
        - AUC 가 기준 시점 대비 0.03 이상 하락
        - 순이익 효율(실제 순이익 / 그날 최대 순이익)이 기준 시점 대비 20% 이상 하락
          · 1인당 순이익을 그대로 비교하면 >50K 비율 변화(고객층 변화)까지 저하로 잡힌다
    추가 규칙 (같은 날 결정, 우선순위 순):
        - insufficient: 양성 20명 미만 → 판정 보류
        - recalibrate: 실제 양성 비율 ÷ 예측 확률 평균 ≥ 1.3 또는 ≤ 0.7 → 판정 기준 재계산
    """
    margin = gain - cost                                   # >50K 한 명에게 보냈을 때 남는 돈

    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")   # 한 클래스뿐이면 계산 불가
    profit = profit_per_customer(y, p, threshold, cost, gain)
    max_profit = y.mean() * margin                         # 완벽한 모델: >50K에게만 발송
    efficiency = profit / max_profit if max_profit > 0 else float("nan")

    base_eff = baseline["baseline_profit_per_customer"] / (baseline["baseline_positive_rate"] * margin)
    auc_drop = baseline["baseline_auc"] - auc
    eff_drop_pct = (base_eff - efficiency) / base_eff

    rule_auc = int(auc_drop >= 0.03)                       # NaN 비교는 False → 판정 불가면 규칙 미적용
    rule_profit = int(eff_drop_pct >= 0.20)

    n_pos = int(y.sum())
    calib_ratio = y.mean() / p.mean() if p.mean() > 0 else float("nan")   # 실제 >50K 비율 ÷ 예측 확률 평균
    rule_insufficient = int(n_pos < MIN_POSITIVES)
    rule_calib = int(calib_ratio >= 1.3 or calib_ratio <= 0.7)

    # 우선순위: 근거가 부족하면 다른 규칙도 믿을 수 없으므로 보류가 먼저 (2026-11-08 결정)
    if rule_insufficient:
        decision = "insufficient"                          # 판정 보류, 라벨 추가 요청
    elif rule_auc or rule_profit:
        decision = "degraded"                              # 재학습 트리거
    elif rule_calib:
        decision = "recalibrate"                           # 판정 기준 재계산
    else:
        decision = "ok"

    metrics = {
        "n_samples": len(y), "n_positives": n_pos, "positive_rate": float(y.mean()),
        "auc": auc, "auc_drop": auc_drop,
        "profit_per_customer": profit, "efficiency": efficiency, "efficiency_drop_pct": eff_drop_pct,
        "mean_pred_proba": float(p.mean()), "calibration_ratio": calib_ratio,
        "baseline_auc": baseline["baseline_auc"], "baseline_efficiency": base_eff, "threshold": threshold,
        "rule_insufficient": rule_insufficient, "rule_auc": rule_auc,
        "rule_profit": rule_profit, "rule_calibration": rule_calib,
    }
    return decision, metrics
