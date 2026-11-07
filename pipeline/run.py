"""파이프라인 실행 후 완료를 기다리고, notify 알림과 판정 지표를 출력한다.
사용법: run.py <name> key=value ...   (port-forward: ml-pipeline-ui 8080, 이미 떠 있어야 함)"""
import subprocess, sys
import kfp

name, args = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
for k in ("min_drifted",):
    if k in args: args[k] = int(args[k])
client = kfp.Client(host="http://localhost:8080")
run = client.create_run_from_pipeline_package("drift_pipeline.yaml", arguments=args, run_name=name,
                                              experiment_name="drift-judge", enable_caching=False)
res = client.wait_for_run_completion(run.run_id, timeout=900)
print(f"[{name}] state={res.state}  run_id={run.run_id[:8]}")
pods = subprocess.run(["kubectl", "-n", "kubeflow", "get", "pods", "-l", f"pipeline/runid={run.run_id}",
                       "-o", "jsonpath={range .items[*]}{.metadata.name}{' '}{.status.phase}{'\\n'}{end}"],
                      capture_output=True, text=True).stdout.split("\n")
for line in filter(None, pods):
    pod, phase = line.split()
    if phase == "Failed":
        print("  FAILED pod:", pod)
    logs = subprocess.run(["kubectl", "-n", "kubeflow", "logs", pod, "-c", "main"], capture_output=True, text=True).stdout
    for l in logs.splitlines():
        if l.startswith(("[ALERT]", "override:")) or (phase == "Failed" and ("Error" in l or "Traceback" in l)):
            print("  " + l)
