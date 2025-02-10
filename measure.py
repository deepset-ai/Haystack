# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
import statistics
import subprocess


def measure_import_times(n_runs=30, module="haystack"):
    """
    Measure the time it takes to import a module.
    """
    user_times = []
    sys_times = []

    print(f"Running {n_runs} measurements...")

    for i in range(n_runs):
        # Run the import command and capture output
        result = subprocess.run(["time", "python", "-c", f"import {module}"], capture_output=True, text=True)

        # Check both stdout and stderr
        time_output = result.stderr

        # Extract times using regex - matches patterns like "3.21user 0.17system"
        time_pattern = r"([\d.]+)user\s+([\d.]+)system"
        match = re.search(time_pattern, time_output)

        if match:
            user_time = float(match.group(1))
            sys_time = float(match.group(2))

            user_times.append(user_time)
            sys_times.append(sys_time)

        # print(user_times)

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} runs...")

    # Calculate statistics
    avg_user = statistics.mean(user_times)
    avg_sys = statistics.mean(sys_times)
    avg_total = avg_user + avg_sys

    # Calculate standard deviations
    std_user = statistics.stdev(user_times)
    std_sys = statistics.stdev(sys_times)

    print("\nResults:")
    print(f"Average user time: {avg_user:.3f}s ± {std_user:.3f}s")
    print(f"Average sys time:  {avg_sys:.3f}s ± {std_sys:.3f}s")
    print(f"Average total (user + sys): {avg_total:.3f}s")


if __name__ == "__main__":
    measure_import_times()
