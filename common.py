import numpy as np, pandas as pd
from pathlib import Path

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "class"]
DATA = Path(__file__).parent / "data"


def load_df():
    # UCI 원본: 헤더 없음, 앞 공백, 결측 '?', test 첫 줄 주석 + 라벨 끝 '.'
    kw = dict(names=COLS, skipinitialspace=True, na_values="?")
    df = pd.concat([pd.read_csv(DATA / "adult.data", **kw),
                    pd.read_csv(DATA / "adult.test", skiprows=1, **kw)], ignore_index=True)
    df["class"] = df["class"].str.rstrip(".")
    return df


def split():
    df = load_df()
    ref = df.sample(5000, random_state=0)            # 학습 + 레퍼런스
    pool = df.drop(ref.index)
    young = pool[(pool.age < 30) & (pool["hours-per-week"] < 35)]
    return ref, pool, young
