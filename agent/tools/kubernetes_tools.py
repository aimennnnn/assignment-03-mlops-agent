"""Task 4 - Kubernetes operations tools.

Each function below is one "tool" the AI agent can call.
They use the official Kubernetes Python client (the same access kubectl has).
Everything is limited to ONE approved namespace, and the only tool that changes
the cluster (restart_deployment) needs confirm=True.

Try it by hand:   python -m agent.tools.kubernetes_tools
"""

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException

load_dotenv()

# Safety: the ONLY namespace these tools may read or change.
APPROVED_NAMESPACE = os.environ.get("K8S_NAMESPACE", "mlops")
APP = os.environ.get("K8S_APP_NAME", "cancer-api")  # Deployment, Service and label name
TIMEOUT = 10  # seconds per request
# Event reasons that mean a pod was restarted or is failing its health checks.
RESTART_REASONS = {"BackOff", "Killing", "Unhealthy", "Failed", "FailedScheduling", "OOMKilling"}

_apis = {}


def _api(kind):
    """Connect to the cluster from the kubeconfig file (done once, then reused)."""
    if not _apis:
        config.load_kube_config(context=os.environ.get("K8S_CONTEXT") or None)
        _apis.update(core=client.CoreV1Api(), apps=client.AppsV1Api(),
                     discovery=client.DiscoveryV1Api())
    return _apis[kind]


def _safe(tool):
    """Turn any failure into {"ok": False, "error": ...} so the agent can report it."""

    def wrapper(*args, **kwargs):
        try:
            return {"ok": True, "tool": tool.__name__, **tool(*args, **kwargs)}
        except (PermissionError, ValueError) as e:
            error = str(e)
        except ConfigException:
            error = "No Kubernetes cluster is configured on this machine (kubeconfig not found)"
        except ApiException as e:
            if e.status == 404:
                error = "Not found in Kubernetes: " + _reason(e)
            elif e.status in (401, 403):
                error = "Kubernetes refused the request (not authorized): " + _reason(e)
            else:
                error = f"Kubernetes answered {e.status}: " + _reason(e)
        except Exception as e:  # connection refused, timeout, ...
            error = f"Kubernetes cluster is not reachable ({type(e).__name__})"
        return {"ok": False, "tool": tool.__name__, "error": error}

    wrapper.__name__ = tool.__name__
    wrapper.__doc__ = tool.__doc__
    return wrapper


def _reason(e):
    try:
        return str(json.loads(e.body).get("message", e.reason))[:300]
    except (TypeError, ValueError):
        return str(e.reason)[:300]


def _ns(namespace):
    """Refuse any namespace except the approved one."""
    namespace = namespace or APPROVED_NAMESPACE
    if namespace != APPROVED_NAMESPACE:
        raise PermissionError(
            f"Namespace '{namespace}' is not approved. These tools only work in "
            f"'{APPROVED_NAMESPACE}'."
        )
    return namespace


def _time(value):
    return value.isoformat(timespec="seconds") if value else None


def _container_problem(status):
    """Why a container is not running normally, taken from its recorded state."""
    waiting = status.state.waiting if status.state else None
    last = status.last_state.terminated if status.last_state else None
    problem = {}
    if waiting and waiting.reason not in (None, "ContainerCreating", "PodInitializing"):
        problem["waiting_reason"] = waiting.reason  # e.g. CrashLoopBackOff, ImagePullBackOff
        problem["waiting_message"] = (waiting.message or "")[:200]
    if last:
        problem["last_exit"] = {"reason": last.reason, "exit_code": last.exit_code,
                                "finished": _time(last.finished_at)}
    return problem


def _pod_summary(pod):
    statuses = pod.status.container_statuses or []
    ready = bool(statuses) and all(s.ready for s in statuses)
    problems = [p for p in (_container_problem(s) for s in statuses) if p]
    return {
        "name": pod.metadata.name,
        "phase": pod.status.phase,
        "ready": ready,
        "restarts": sum(s.restart_count for s in statuses),
        "image": pod.spec.containers[0].image,
        "started": _time(pod.status.start_time),
        "being_deleted": pod.metadata.deletion_timestamp is not None,
        "problems": problems,
    }


def _events(namespace, object_name=None, warnings_only=False, limit=8):
    """Recent Kubernetes events (newest last), optionally for one pod."""
    selector = f"involvedObject.name={object_name}" if object_name else None
    items = _api("core").list_namespaced_event(
        namespace, field_selector=selector, _request_timeout=TIMEOUT
    ).items
    out = []
    for e in items:
        if warnings_only and e.type != "Warning":
            continue
        when = e.last_timestamp or e.event_time or e.first_timestamp
        out.append({
            "time": _time(when),
            "type": e.type,
            "reason": e.reason,
            "object": e.involved_object.name,
            "message": (e.message or "")[:250],
            "count": e.count or 1,
        })
    out.sort(key=lambda x: x["time"] or "")
    return out[-limit:]


@_safe
def list_workloads(namespace=None):
    """List the application's Deployments and Pods with their current state."""
    ns = _ns(namespace)
    deployments = _api("apps").list_namespaced_deployment(ns, _request_timeout=TIMEOUT).items
    pods = _api("core").list_namespaced_pod(ns, _request_timeout=TIMEOUT).items
    return {
        "namespace": ns,
        "deployments": [
            {
                "name": d.metadata.name,
                "desired_replicas": d.spec.replicas,
                "ready_replicas": d.status.ready_replicas or 0,
                "available_replicas": d.status.available_replicas or 0,
                "unavailable_replicas": d.status.unavailable_replicas or 0,
                "image": d.spec.template.spec.containers[0].image,
            }
            for d in deployments
        ],
        "pods": [_pod_summary(p) for p in pods],
    }


@_safe
def get_pod_status(pod_name, namespace=None):
    """Get one Pod's detailed status: conditions, container state, probes and its events."""
    ns = _ns(namespace)
    pod = _api("core").read_namespaced_pod(pod_name, ns, _request_timeout=TIMEOUT)
    c = pod.spec.containers[0]

    def probe(p):
        if not p or not p.http_get:
            return None
        return {"path": p.http_get.path, "port": p.http_get.port,
                "every_seconds": p.period_seconds, "fail_after": p.failure_threshold}

    return {
        **_pod_summary(pod),
        "namespace": ns,
        "node": pod.spec.node_name,
        "conditions": {x.type: x.status for x in (pod.status.conditions or [])},
        "readiness_probe": probe(c.readiness_probe),
        "liveness_probe": probe(c.liveness_probe),
        "resources": {
            "requests": (c.resources.requests or {}) if c.resources else {},
            "limits": (c.resources.limits or {}) if c.resources else {},
        },
        "events": _events(ns, pod_name),
    }


@_safe
def get_pod_logs(pod_name, namespace=None, tail_lines=30, previous=False):
    """Get the last log lines of a Pod. previous=True reads the run before the last restart."""
    ns = _ns(namespace)
    text = _api("core").read_namespaced_pod_log(
        pod_name, ns, tail_lines=min(int(tail_lines), 200), previous=bool(previous),
        _request_timeout=TIMEOUT,
    )
    lines = [line[:300] for line in (text or "").splitlines()]
    return {
        "namespace": ns,
        "pod": pod_name,
        "previous_run": bool(previous),
        "line_count": len(lines),
        "error_lines": [x for x in lines if "ERROR" in x or "FAILED" in x or "Traceback" in x][-5:],
        "log_tail": lines,
    }


@_safe
def get_services(namespace=None):
    """Inspect the application's Services and how many healthy Pods each one can reach."""
    ns = _ns(namespace)
    out = []
    for s in _api("core").list_namespaced_service(ns, _request_timeout=TIMEOUT).items:
        slices = _api("discovery").list_namespaced_endpoint_slice(
            ns, label_selector=f"kubernetes.io/service-name={s.metadata.name}",
            _request_timeout=TIMEOUT,
        ).items
        endpoints = [e for sl in slices for e in (sl.endpoints or [])]
        ready = sum(1 for e in endpoints if e.conditions and e.conditions.ready)
        out.append({
            "name": s.metadata.name,
            "type": s.spec.type,
            "cluster_ip": s.spec.cluster_ip,
            "ports": [{"port": p.port, "target_port": p.target_port, "protocol": p.protocol}
                      for p in (s.spec.ports or [])],
            "selector": s.spec.selector or {},
            "ready_endpoints": ready,
            "not_ready_endpoints": len(endpoints) - ready,
            "can_serve_traffic": ready > 0,
        })
    return {"namespace": ns, "services": out}


@_safe
def find_unhealthy_pods(namespace=None):
    """Find Pods that are not ready or have restarted, with reasons and restart events."""
    ns = _ns(namespace)
    pods = [_pod_summary(p) for p in
            _api("core").list_namespaced_pod(ns, _request_timeout=TIMEOUT).items]
    unhealthy = [p for p in pods if not p["ready"] or p["restarts"] > 0]
    warnings = _events(ns, warnings_only=True, limit=15)
    for p in unhealthy:
        p["warning_events"] = [e for e in warnings if e["object"] == p["name"]][-5:]
    return {
        "namespace": ns,
        "total_pods": len(pods),
        "healthy_pods": len(pods) - len(unhealthy),
        "unhealthy_pods": unhealthy,
        "restart_events": [e for e in warnings if e["reason"] in RESTART_REASONS][-8:],
    }


@_safe
def get_deployed_version(deployment=APP, namespace=None):
    """Report which application version and model are deployed right now."""
    ns = _ns(namespace)
    d = _api("apps").read_namespaced_deployment(deployment, ns, _request_timeout=TIMEOUT)
    image = d.spec.template.spec.containers[0].image
    result = {
        "namespace": ns,
        "deployment": deployment,
        "image": image,
        "image_tag": image.rsplit(":", 1)[1] if ":" in image else "latest",
        "version_label": (d.metadata.labels or {}).get("app.kubernetes.io/version"),
        "desired_replicas": d.spec.replicas,
        "ready_replicas": d.status.ready_replicas or 0,
        "configmap": {},
        "live_version": None,
    }
    for ref in d.spec.template.spec.containers[0].env_from or []:
        if ref.config_map_ref:
            cm = _api("core").read_namespaced_config_map(
                ref.config_map_ref.name, ns, _request_timeout=TIMEOUT
            )
            result["configmap"] = dict(cm.data or {})
    # Ask the running app itself, through the Kubernetes API (no port-forward needed).
    if result["ready_replicas"] > 0:
        try:
            reply = _api("core").connect_get_namespaced_service_proxy_with_path(
                f"{deployment}:80", ns, "version", _preload_content=False,
                _request_timeout=TIMEOUT,
            )
            result["live_version"] = json.loads(reply.data)
        except Exception as e:  # the app did not answer; keep the recorded facts
            result["live_version_error"] = f"app did not answer /version ({type(e).__name__})"
    else:
        result["live_version_error"] = "no ready pods, so the running app could not be asked"
    return result


@_safe
def restart_deployment(deployment=APP, namespace=None, confirm=False):
    """Restart a Deployment's pods one by one. Changes the cluster, so it needs confirm=True."""
    ns = _ns(namespace)
    d = _api("apps").read_namespaced_deployment(deployment, ns, _request_timeout=TIMEOUT)
    if not confirm:
        return {
            "restarted": False,
            "needs_confirmation": True,
            "message": f"This would restart the {d.spec.replicas} pod(s) of deployment "
            f"'{deployment}' in namespace '{ns}'. Ask the user to confirm, "
            "then call again with confirm=True.",
        }
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    patch = {"spec": {"template": {"metadata": {"annotations": {
        "kubectl.kubernetes.io/restartedAt": now}}}}}
    _api("apps").patch_namespaced_deployment(deployment, ns, patch, _request_timeout=TIMEOUT)
    return {"restarted": True, "deployment": deployment, "namespace": ns, "requested_at": now}


# The list the agent will pick from in Step 5.
KUBERNETES_TOOLS = [
    list_workloads,
    get_pod_status,
    get_pod_logs,
    get_services,
    find_unhealthy_pods,
    get_deployed_version,
    restart_deployment,
]


def _show(result):
    print(json.dumps(result, indent=2, default=str), "\n")


if __name__ == "__main__":
    workloads = list_workloads()
    _show(workloads)
    _show(get_services())
    _show(get_deployed_version())
    unhealthy = find_unhealthy_pods()
    _show(unhealthy)
    if workloads["ok"] and workloads["pods"]:
        # Look closely at a broken pod if there is one, otherwise at the first pod.
        broken = unhealthy["unhealthy_pods"] if unhealthy["ok"] else []
        name = (broken or workloads["pods"])[0]["name"]
        _show(get_pod_status(name))
        _show(get_pod_logs(name, tail_lines=10))
    _show(restart_deployment())  # no confirm -> asks first, changes nothing
    _show(restart_deployment(namespace="kube-system", confirm=True))  # not approved -> refused
    _show(get_pod_logs("no-such-pod"))  # does not exist -> clean error, no guess
