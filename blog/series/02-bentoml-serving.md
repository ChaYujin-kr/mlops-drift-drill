# [MLOps 실습 시리즈] 2. BentoML로 모델 서빙하기

학습한 모델을 API로 만드는 단계다. 개념은 단순한데, 요청이 **한 건씩** 들어온다는 점에서 함정이 하나 있었다.

## Bento는 이미지가 아니다

- **Bento**는 모델, 서비스 코드, 의존성(`bentofile.yaml`)을 묶은 **소스 묶음(tar)**이다. 컨테이너 이미지가 아니다.
- 이미지는 `bentoml containerize`로 따로 만든다.
- 흐름: **MLflow**(실험과 승인 버전 관리) → **Bento**(배포 단위로 고정) → **이미지** → K8s 배포

## API 서버와 러너를 나누는 이유

- **API 서버**: 요청을 받는 프론트엔드
- **러너**: 모델을 메모리에 올려두고 추론하는 백엔드

나누는 이유는 **따로 늘릴 수 있어서**다. 요청이 많으면 API 서버를, 추론이 무거우면 러너를 늘린다.

## 함정: 한 건만 들어오면 열이 14개뿐이다

학습할 때는 5,000건을 한꺼번에 원핫 인코딩해서 **102열**이 나왔다. 그런데 요청은 한 건씩 온다. 같은 방법으로 원핫 하면 그 사람 값에 해당하는 **14열**만 생긴다.

자동완성이 제안한 코드는 그럴듯했지만 실행하니 터졌다.

```python
X[COLUMN_LIST]   # KeyError: 없는 열 88개
```

해결은 학습 때 열 목록에 맞추고, 없는 열은 0으로 채우는 것이다.

```python
X = pd.get_dummies(pd.DataFrame([row])).reindex(columns=COLUMN_LIST, fill_value=0)
```

- `columns`: 결과가 가질 열 목록과 순서. 없는 열은 만들고, 남는 열은 버린다.
- `fill_value=0`: 새로 만든 열을 0으로. 원핫에서 "해당 없음"은 0이기 때문이다.

## 더 나은 방법: 전처리를 모델 안에 넣기

열 목록 파일을 따로 관리하면 학습과 서빙이 언젠가 어긋난다. 그래서 나중에 이렇게 바꿨다.

```python
model = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                               remainder="passthrough")),
    ("rf", RandomForestClassifier()),
])
```

모델 파일 하나에 전처리까지 들어가니 서빙 코드에 `reindex`가 필요 없어지고, 모델을 되돌릴 때 전처리도 함께 되돌아간다.

## 참고: Yatai는 쓰지 않았다

책에서는 Bento를 K8s에 배포할 때 Yatai를 쓴다. 하지만 yatai-chart 저장소는 아카이브됐고 업데이트도 사실상 멈췄다. 대신 `bentoml build` → `bentoml containerize` → KServe(또는 일반 Deployment)로 배포하는 방식을 택했다.

## 한 줄 정리

서빙에서 가장 흔한 사고는 학습과 서빙의 변환이 어긋나는 것이다. 전처리는 모델과 한 몸으로 저장하자.
