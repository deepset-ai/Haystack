# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys

from lazy_imports import LazyImporter

_import_structure = {
    ".utils.device": ["ComponentDevice", "Device", "DeviceMap", "DeviceType"],
    ".utils.auth": ["Secret", "deserialize_secrets_inplace"],
    ".utils.callable_serialization": ["deserialize_callable", "serialize_callable"],
    ".utils.docstore_deserialization": ["deserialize_document_store_in_init_params_inplace"],
    ".utils.expit": ["expit"],
    ".utils.filters": ["document_matches_filter", "raise_on_invalid_filter_syntax"],
    ".utils.type_serialization": ["deserialize_type", "serialize_type"],
    ".utils.jinja2_extensions": ["Jinja2TimeExtension"],
    ".utils.jupyter": ["is_in_jupyter"],
    ".utils.requests_utils": ["request_with_retry"],
}

sys.modules[__name__] = LazyImporter(__name__, globals()["__file__"], _import_structure)
