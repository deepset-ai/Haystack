# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path


class ImportTracker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        """
        If the module name contains "torch", print the full name and the stack trace.
        """
        if "torch" in fullname:
            print(f"\nAttempting to import: {fullname}")
            import traceback

            for frame in traceback.extract_stack()[:-1]:  # Exclude this frame
                if "haystack" in frame.filename:
                    print(f"  In Haystack file: {frame.filename}:{frame.lineno}")
                    print(f"    {frame.line}")


# Install the import tracker
sys.meta_path.insert(0, ImportTracker())

# Record modules before import
print("Recording initial modules...")
modules_before = set(sys.modules.keys())

# Import haystack
print("Importing haystack...")
import haystack

# Find new modules after import
print("Analyzing new modules...")
modules_after = set(sys.modules.keys())
new_modules = modules_after - modules_before

# Filter for haystack modules that imported torch
haystack_importers = {}

for name in new_modules:
    if name.startswith("haystack"):
        module = sys.modules[name]
        # Check if this module uses torch
        module_dict = getattr(module, "__dict__", {})
        for value in module_dict.values():
            if isinstance(value, types.ModuleType) and "torch" in value.__name__:
                if name not in haystack_importers:
                    haystack_importers[name] = set()
                haystack_importers[name].add(value.__name__)

if haystack_importers:
    print("\nFound haystack modules that imported torch:")
    for module_name, torch_modules in sorted(haystack_importers.items()):
        print(f"\n{module_name}:")
        for torch_module in sorted(torch_modules):
            print(f"  - {torch_module}")
else:
    print("\nNo haystack modules imported torch")
