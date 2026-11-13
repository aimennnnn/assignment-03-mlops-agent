"""Kubernetes tools against a fake cluster: safety limits and failure diagnosis."""

from agent.tools import kubernetes_tools as k8s


def test_unauthorized_kubernetes_modification_is_rejected(healthy_cluster):
    """A change outside the approved namespace is refused, even with confirm=True."""
    result = k8s.restart_deployment(namespace="kube-system", confirm=True)

    assert result["ok"] is False
    assert "not approved" in result["error"]
    assert healthy_cluster.patches == []  # nothing was sent to the cluster


def test_reading_another_namespace_is_rejected_too(healthy_cluster):
    assert k8s.list_workloads(namespace="default")["ok"] is False
    assert k8s.get_pod_logs("any-pod", namespace="kube-system")["ok"] is False


def test_restart_needs_confirmation(healthy_cluster):
    asked = k8s.restart_deployment()
    assert asked["needs_confirmation"] is True and healthy_cluster.patches == []

    done = k8s.restart_deployment(confirm=True)
    assert done["restarted"] is True and done["rollout_finished"] is True
    assert [p["namespace"] for p in healthy_cluster.patches] == ["mlops"]


def test_restart_reports_when_new_pods_do_not_become_ready(broken_cluster, monkeypatch):
    """A restart that does not fix anything is reported honestly, not as a success."""
    monkeypatch.setattr(k8s, "ROLLOUT_WAIT", 0)
    done = k8s.restart_deployment(confirm=True)
    assert done["restarted"] is True and done["rollout_finished"] is False
    assert done["ready_replicas"] == 0 and "not all ready" in done["note"]


def test_healthy_cluster_is_reported_healthy(healthy_cluster):
    assert k8s.find_unhealthy_pods()["unhealthy_pods"] == []
    assert k8s.get_services()["services"][0]["can_serve_traffic"] is True
    version = k8s.get_deployed_version()
    assert version["image_tag"] == "1.0.0" and version["ready_replicas"] == 2
    assert version["live_version"]["model"]["model_version"] == "4"


def test_crash_loop_is_diagnosed_from_status_events_and_logs(broken_cluster):
    """Injected failure: a bad MODEL_PATH makes every pod crash on startup."""
    unhealthy = k8s.find_unhealthy_pods()
    assert unhealthy["healthy_pods"] == 0
    pod = unhealthy["unhealthy_pods"][0]
    assert pod["problems"][0]["waiting_reason"] == "CrashLoopBackOff"
    assert pod["problems"][0]["last_exit"]["exit_code"] == 3
    assert unhealthy["restart_events"][0]["reason"] == "BackOff"

    assert k8s.get_services()["services"][0]["can_serve_traffic"] is False
    logs = k8s.get_pod_logs(pod["name"])
    assert any("missing-model.joblib" in line for line in logs["error_lines"])
    version = k8s.get_deployed_version()
    assert version["live_version"] is None  # no ready pod, so nothing is made up
    assert "missing-model" in version["configmap"]["MODEL_PATH"]


def test_missing_pod_gives_a_clean_error(healthy_cluster):
    result = k8s.get_pod_logs("no-such-pod")
    assert result["ok"] is False and "Not found" in result["error"]


def test_no_cluster_configured(monkeypatch):
    """Service failure: no kubeconfig on the machine."""
    from kubernetes.config.config_exception import ConfigException

    def no_config(**kwargs):
        raise ConfigException("no kubeconfig")

    k8s._apis.clear()
    monkeypatch.setattr(k8s.config, "load_kube_config", no_config)
    result = k8s.list_workloads()
    assert result["ok"] is False and "No Kubernetes cluster" in result["error"]
