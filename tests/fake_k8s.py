"""A fake Kubernetes cluster for tests: answers the same calls as the real client,
using data shaped exactly like real Kubernetes API replies (no cluster needed)."""

import json

from kubernetes import client
from kubernetes.client.rest import ApiException

_parser = client.ApiClient()


class _Reply:
    def __init__(self, data):
        self.data = json.dumps(data)


def _typed(data, kind):
    """Turn raw API JSON into the same typed objects the real client returns."""
    return _parser._ApiClient__deserialize(data, kind)


def _api_error(status, message):
    e = ApiException(status=status, reason="error")
    e.body = json.dumps({"message": message})
    return e


def pod(name, ready=True, restarts=0, waiting=None, last_exit=None, image="cancer-api:1.0.0"):
    state = {"waiting": {"reason": waiting, "message": f"back-off restarting {name}"}} \
        if waiting else {"running": {"startedAt": "2026-11-14T10:00:05Z"}}
    status = {"name": "api", "ready": ready, "restartCount": restarts, "image": image,
              "imageID": "", "state": state}
    if last_exit is not None:
        status["lastState"] = {"terminated": {"exitCode": last_exit, "reason": "Error",
                                              "finishedAt": "2026-11-14T10:03:00Z"}}
    return {
        "metadata": {"name": name, "namespace": "mlops", "labels": {"app": "cancer-api"}},
        "spec": {"nodeName": "docker-desktop", "containers": [{
            "name": "api", "image": image,
            "envFrom": [{"configMapRef": {"name": "cancer-api-config"}}],
            "readinessProbe": {"httpGet": {"path": "/ready", "port": 8000},
                               "periodSeconds": 5, "failureThreshold": 3},
            "livenessProbe": {"httpGet": {"path": "/health", "port": 8000},
                              "periodSeconds": 10, "failureThreshold": 3},
            "resources": {"requests": {"cpu": "100m", "memory": "256Mi"},
                          "limits": {"cpu": "500m", "memory": "512Mi"}},
        }]},
        "status": {"phase": "Running", "startTime": "2026-11-14T10:00:00Z",
                   "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
                   "containerStatuses": [status]},
    }


def event(obj, reason, message, kind="Warning", count=1):
    return {"metadata": {"name": f"{obj}.{reason}", "namespace": "mlops"},
            "involvedObject": {"kind": "Pod", "name": obj, "namespace": "mlops"},
            "type": kind, "reason": reason, "message": message, "count": count,
            "lastTimestamp": "2026-11-14T10:03:10Z"}


class FakeCluster:
    """Holds the pretend cluster state. healthy=False gives two crash-looping pods."""

    def __init__(self, healthy=True):
        self.patches = []  # every change request is recorded here
        self.config = {"APP_VERSION": "1.0.0", "MODEL_PATH": "/app/model/model.joblib"}
        if healthy:
            self.pods = [pod("cancer-api-7d9f-aaaaa"), pod("cancer-api-7d9f-bbbbb")]
            self.events = [event("cancer-api-7d9f-aaaaa", "Started", "Started container api",
                                 kind="Normal")]
            self.logs = "INFO model loaded: cancer-classifier version 4\nINFO GET /ready 200"
        else:
            self.config["MODEL_PATH"] = "/app/model/missing-model.joblib"
            self.pods = [
                pod("cancer-api-5c6b-ccccc", ready=False, restarts=4,
                    waiting="CrashLoopBackOff", last_exit=3),
                pod("cancer-api-5c6b-ddddd", ready=False, restarts=4,
                    waiting="CrashLoopBackOff", last_exit=3),
            ]
            self.events = [
                event(p["metadata"]["name"], "BackOff",
                      "Back-off restarting failed container api", count=12)
                for p in self.pods
            ]
            self.logs = ("ERROR FAILED to load model from /app/model/missing-model.joblib: "
                         "[Errno 2] No such file or directory\n"
                         "ERROR:    Application startup failed. Exiting.")

    @property
    def ready(self):
        return sum(1 for p in self.pods if p["status"]["containerStatuses"][0]["ready"])

    # ---- CoreV1Api ----
    def list_namespaced_pod(self, namespace, **kw):
        return _typed({"items": self.pods}, "V1PodList")

    def read_namespaced_pod(self, name, namespace, **kw):
        for p in self.pods:
            if p["metadata"]["name"] == name:
                return _typed(p, "V1Pod")
        raise _api_error(404, f'pods "{name}" not found')

    def read_namespaced_pod_log(self, name, namespace, **kw):
        self.read_namespaced_pod(name, namespace)
        return self.logs

    def list_namespaced_event(self, namespace, field_selector=None, **kw):
        items = self.events
        if field_selector:
            wanted = field_selector.split("=", 1)[1]
            items = [e for e in items if e["involvedObject"]["name"] == wanted]
        return _typed({"items": items}, "CoreV1EventList")

    def list_namespaced_service(self, namespace, **kw):
        return _typed({"items": [{
            "metadata": {"name": "cancer-api", "namespace": "mlops"},
            "spec": {"type": "ClusterIP", "clusterIP": "10.96.12.34",
                     "selector": {"app": "cancer-api"},
                     "ports": [{"name": "http", "port": 80, "targetPort": 8000,
                                "protocol": "TCP"}]},
        }]}, "V1ServiceList")

    def read_namespaced_config_map(self, name, namespace, **kw):
        return _typed({"metadata": {"name": name}, "data": self.config}, "V1ConfigMap")

    def connect_get_namespaced_service_proxy_with_path(self, name, namespace, path, **kw):
        return _Reply({"app_version": self.config["APP_VERSION"],
                       "model": {"model_name": "cancer-classifier", "model_version": "4"}})

    # ---- AppsV1Api ----
    def _deployment(self):
        return {
            "metadata": {"name": "cancer-api", "namespace": "mlops",
                         "labels": {"app": "cancer-api", "app.kubernetes.io/version": "1.0.0"}},
            "spec": {"replicas": 2, "selector": {"matchLabels": {"app": "cancer-api"}},
                     "template": {"metadata": {"labels": {"app": "cancer-api"}},
                                  "spec": self.pods[0]["spec"]}},
            "status": {"replicas": 2, "readyReplicas": self.ready,
                       "updatedReplicas": self.ready,
                       "availableReplicas": self.ready,
                       "unavailableReplicas": 2 - self.ready},
        }

    def list_namespaced_deployment(self, namespace, **kw):
        return _typed({"items": [self._deployment()]}, "V1DeploymentList")

    def read_namespaced_deployment(self, name, namespace, **kw):
        if name != "cancer-api":
            raise _api_error(404, f'deployments.apps "{name}" not found')
        return _typed(self._deployment(), "V1Deployment")

    def patch_namespaced_deployment(self, name, namespace, body, **kw):
        self.patches.append({"name": name, "namespace": namespace, "body": body})

    # ---- DiscoveryV1Api ----
    def list_namespaced_endpoint_slice(self, namespace, **kw):
        return _typed({"items": [{
            "metadata": {"name": "cancer-api-abc"}, "addressType": "IPv4",
            "endpoints": [{"addresses": [f"10.1.0.{i}"],
                           "conditions": {"ready": p["status"]["containerStatuses"][0]["ready"]}}
                          for i, p in enumerate(self.pods)],
        }]}, "V1EndpointSliceList")


def install(monkeypatch_target, cluster):
    """Point kubernetes_tools at the fake cluster instead of a real one."""
    monkeypatch_target._apis.clear()
    monkeypatch_target._apis.update(core=cluster, apps=cluster, discovery=cluster)
