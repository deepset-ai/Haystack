# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import sys
from typing import Any, Dict, Optional

from haystack import logging
from haystack.version import __version__

logger = logging.getLogger(__name__)

# This value cannot change during the lifetime of the process
_IS_DOCKER_CACHE = None


def _in_podman() -> bool:
    """
    Check if the code is running in a Podman container.

    Podman run would create the file /run/.containernv, see:
    https://github.com/containers/podman/blob/main/docs/source/markdown/podman-run.1.md.in#L31
    """
    return os.path.exists("/run/.containerenv")


def _has_dockerenv() -> bool:
    """
    Check if the code is running in a Docker container.

    This might not work anymore at some point (even if it's been a while now), see:
    https://github.com/moby/moby/issues/18355#issuecomment-220484748
    """
    return os.path.exists("/.dockerenv")


def _has_docker_cgroup_v1() -> bool:
    """
    This only works with cgroups v1.
    """
    path = "/proc/self/cgroup"  # 'self' should be always symlinked to the actual PID
    return os.path.isfile(path) and any("docker" in line for line in open(path))


def _has_docker_cgroup_v2() -> bool:
    """
    Check if the code is running in a Docker container using the cgroups v2 version.

    inspired from: https://github.com/jenkinsci/docker-workflow-plugin/blob/master/src/main/java/org/jenkinsci/plugins/docker/workflow/client/DockerClient.java
    """
    path = "/proc/self/mountinfo"  # 'self' should be always symlinked to the actual PID
    return os.path.isfile(path) and any("/docker/containers/" in line for line in open(path))


def _is_containerized() -> Optional[bool]:
    """
    This code is based on the popular 'is-docker' package for node.js
    """
    global _IS_DOCKER_CACHE  # pylint: disable=global-statement

    if _IS_DOCKER_CACHE is None:
        _IS_DOCKER_CACHE = _in_podman() or _has_dockerenv() or _has_docker_cgroup_v1() or _has_docker_cgroup_v2()

    return _IS_DOCKER_CACHE


def collect_system_specs() -> Dict[str, Any]:
    """
    Collects meta-data about the setup that is used with Haystack.

    Data collected includes: operating system, python version, Haystack version, transformers version,
    pytorch version, number of GPUs, execution environment.

    These values are highly unlikely to change during the runtime of the pipeline,
    so they're collected only once.
    """
    specs = {
        "libraries.haystack": __version__,
        "os.containerized": _is_containerized(),
        "os.version": platform.release(),
        "os.family": platform.system(),
        "os.machine": platform.machine(),
        "python.version": platform.python_version(),
        "hardware.cpus": os.cpu_count(),
        "libraries.pytest": sys.modules["pytest"].__version__ if "pytest" in sys.modules.keys() else False,
        "libraries.ipython": sys.modules["ipython"].__version__ if "ipython" in sys.modules.keys() else False,
        "libraries.colab": sys.modules["google.colab"].__version__ if "google.colab" in sys.modules.keys() else False,
    }

    return specs
