import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from common import split

ref, _, _ = split()
ref = ref.drop(columns="class")
log = pd.read_json("logs/preds.jsonl", lines=True)[ref.columns]   # 원본 14열만

for t in range(len(log) // 200):
    cur = log.iloc[t * 200:(t + 1) * 200]
    r = Report(metrics=[DataDriftPreset(stattest="psi", stattest_threshold=0.1, drift_share=0.5)])
    r.run(reference_data=ref, current_data=cur)          # 0.5.0: 키워드 인자
    m = r.as_dict()["metrics"]
    ds, table = m[0]["result"], m[1]["result"]["drift_by_columns"]
    drifted = [c for c, v in table.items() if v["drift_detected"]]
    print(f"w{t:<2} share={ds['share_of_drifted_columns']:.2f} "
          f"ALERT={'Y' if ds['dataset_drift'] else '-'}  {drifted}")
