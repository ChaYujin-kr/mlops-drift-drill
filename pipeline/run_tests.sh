#!/usr/bin/env bash
# 사용법: ./run_tests.sh           (6일치 판정)
#         ./run_tests.sh cases     (일부러 망가뜨린 경우)
# 전제: MLflow(5000), MinIO(9000) port-forward 가 떠 있어야 함
cd "$(dirname "$0")"
eval "$(kubectl -n kubeflow get secret mlpipeline-minio-artifact -o json | python3 -c "
import sys,json,base64; d=json.load(sys.stdin)['data']
print('export AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s' % (base64.b64decode(d['accesskey']).decode(), base64.b64decode(d['secretkey']).decode()))")"
script=test_judge.py; [ "$1" = cases ] && script=test_judge_cases.py
../.venv-pipeline/bin/python "$script" 2>&1 | grep -vE 'warn|Warn|^\s*$|Downloading'
