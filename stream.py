import numpy as np, pandas as pd, requests
from common import split

ref, pool, young = split()


def stream(n_windows=15, w=200):
    rng = np.random.default_rng(0)
    for t in range(n_windows):
        p = min(1.0, max(0.0, (t - 3) / 8))      # 0~3: 정상, 이후 램프업
        k = rng.binomial(w, p)
        batch = pd.concat([young.sample(k, random_state=t),
                           pool.sample(w - k, random_state=100 + t)])
        yield t, p, batch.sample(frac=1, random_state=t)


s = requests.Session()
for t, p, batch in stream():
    for _, row in batch.drop(columns="class").iterrows():
        s.post("http://localhost:3000/predict", json=row.astype(object).where(row.notna(), None).to_dict()).raise_for_status()
    batch[["class"]].to_csv(f"labels/w{t}.csv")   # 지연 라벨: 따로 적재
    print(f"w{t} p={p:.2f} sent")
