FROM python:3.10-slim-bookworm
RUN pip install --no-cache-dir mlflow==2.17.2 boto3==1.35.36 scikit-learn==1.7.2 pandas==2.2.3 evidently==0.5.0 kfp==2.9.0
# 로컬에서 검증한 판정 로직을 그대로 사용 (복사본을 만들지 않음)
COPY judge_logic.py /app/judge_logic.py
ENV PYTHONPATH=/app
