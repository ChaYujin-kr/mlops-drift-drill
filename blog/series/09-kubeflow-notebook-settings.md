# [MLOps 실습 시리즈] 9. Kubeflow 노트북 설정: PodDefault, 어피니티, 톨러레이션

Kubeflow에서 노트북 서버를 띄울 때 나오는 옵션 세 가지. 이름은 어렵지만 비유하면 쉽다.

## PodDefault: 파드 기본값

네임스페이스에 등록해 두면, **라벨이 맞는 파드에 환경 변수와 볼륨을 자동으로 넣어준다.**

```yaml
kind: PodDefault
metadata:
  name: add-proxy-vars
spec:
  selector:
    matchLabels:
      add-proxy-vars: "true"
  env:
    - name: MLFLOW_TRACKING_URI
      value: http://mlflow.mlflow.svc.cluster.local:5000
```

노트북마다 MLflow 주소를 일일이 적지 않아도 된다.

## 어피니티: 끌어당기기

파드를 **원하는 라벨의 노드로 끌어당긴다.** 예를 들어 `compute: small` 노드에 기본으로 배치한다.

## 톨러레이션: 출입증

노드에 **테인트**(출입 제한)가 걸려 있으면, **톨러레이션**(출입증)이 있는 파드만 들어갈 수 있다. GPU 노드에 아무 파드나 올라가지 않게 막을 때 쓴다.

| | 방향 | 비유 |
|---|---|---|
| 어피니티 | 파드 → 노드 | "나는 저 노드로 가고 싶다" |
| 테인트 + 톨러레이션 | 노드 → 파드 | "출입증 있는 파드만 들어와라" |

## 그 밖에

- 노트북 메뉴 옵션은 `jupyter-web-app-config` ConfigMap에서 바꾼다.
- 작업 공간 볼륨은 `/home/jovyan`에 자동으로 붙고, 기존 PVC를 데이터 볼륨으로 추가할 수 있다.

## 한 줄 정리

PodDefault는 넣어주고, 어피니티는 끌어당기고, 톨러레이션은 통과시킨다.
