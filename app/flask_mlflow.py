from flask import Flask, request
from kubernetes import client, config
import yaml
import os
import mlflow
import mlflow.pytorch

app = Flask(__name__)

def create_kubernetes_job(data_file_path, file_script):
    # Kubernetes config 로드
    config.load_kube_config()

    # Kubernetes Job을 정의
    job_yaml = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "ml-job"
        },
        "spec": {
            "backoffLimit": 4,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "ml-job"
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "ml-job",
                        "image": "soohh/mlflow-suhyeon:v1.0",
                        "command": ["python", file_script, "--data", data_file_path],
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": 1
                            }
                        },
                        "volumeMounts": [{
                            "name": "mlflow-volume",
                            "mountPath": "/user"
                        }]
                    }],
                    "restartPolicy": "Never",
                    "volumes": [{
                        "name": "mlflow-volume",
                        "hostPath": {
                            "path": "/home/sooh/final-proj/user"
                        }
                    }],
                }
            }
        }
    }
    
    # 생성된 YAML 파일을 쿠버네티스 API 서버에 적용합니다.
    # Kubernetes API 클라이언트 생성
    k8s_api = client.BatchV1Api()
    # Job 객체 생성
    k8s_api.create_namespaced_job(
        namespace="default",
        body=job_yaml
    )


# Flask 웹 서버를 실행하며, HTTP POST 요청을 '/submit' 경로로 받아 처리
@app.route('/submit', methods=['POST'])
def submit():
    # 클라이언트로부터 받은 POST 요청의 본문을 JSON 형식으로 파싱하여 Python 딕셔너리로 반환
    data = request.get_json()
    file_script = data.get('ml_file', 'user/mnist_mlflow.py')
    data_file_path = data.get('data_file_path', 'user/data')
    
    # 스크립트 파일이나 데이터 경로가 제공되지 않은 경우 error 발생
    if not file_script or not data_file_path:
        return "Error: file_script or data_file_path not provided", 400

    try:
        create_kubernetes_job(data_file_path, file_script)
        return 'Job created', 200
    except Exception as e:
        # 예외 발생 시 스택 트레이스를 로깅
        app.logger.error('An error occurred while creating job', exc_info=True)
        return f"Error in creating job: {str(e)}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

