"""두 번째 스트림: 드리프트 구간(w6~w11), 고객 ID 포함. 라벨은 고객 ID 기준으로 따로 적재."""
import numpy as np, pandas as pd, requests
from common import split

ref, pool, young = split()
s = requests.Session()
rng = np.random.default_rng(1)
for t in range(6, 12):
    p = min(1.0, max(0.0, (t - 3) / 8))
    k = rng.binomial(200, p)
    batch = pd.concat([young.sample(k, random_state=1000 + t),
                       pool.sample(200 - k, random_state=2000 + t)]).sample(frac=1, random_state=t)
    batch.insert(0, "customer_id", [f"C{i:06d}" for i in batch.index])
    for _, row in batch.drop(columns="class").iterrows():
        s.post("http://localhost:3000/predict",
               json=row.astype(object).where(row.notna(), None).to_dict()).raise_for_status()
    batch[["customer_id", "class"]].to_csv(f"labels2/w{t}.csv", index=False)
    print(f"w{t} p={p:.2f} sent")
