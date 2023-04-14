from typing import Any, List, Tuple, Dict, Optional, Union
from collections import defaultdict
import logging

from haystack.nodes.base import BaseComponent
from haystack.schema import Document

logger = logging.getLogger(__name__)


class RouteDocuments(BaseComponent):
    """
    A node to split a list of `Document`s by `content_type` or by the values of a metadata field and route them to
    different nodes.
    """

    # By default (split_by == "content_type"), the node has two outgoing edges.
    outgoing_edges = 2

    def __init__(
        self,
        split_by: str = "content_type",
        metadata_values: Optional[Union[List[str], List[List[str]]]] = None,
        return_remaining: bool = False,
    ):
        """
        :param split_by: Field to split the documents by, either `"content_type"` or a metadata field name.
            If this parameter is set to `"content_type"`, the list of `Document`s will be split into a list containing
            only `Document`s of type `"text"` (will be routed to `"output_1"`) and a list containing only `Document`s of
            type `"table"` (will be routed to `"output_2"`).
            If this parameter is set to a metadata field name, you need to specify the parameter `metadata_values` as
            well.
        :param metadata_values: If the parameter `split_by` is set to a metadata field name, you need to provide a list
            of values to group the `Document`s to. `Document`s whose metadata field is equal to the first value of the
            provided list will be routed to `"output_1"`, `Document`s whose metadata field is equal to the second
            value of the provided list will be routed to `"output_2"`, etc.
        :param return_remaining: Whether to return all remaining documents that don't match the `split_by` or
            `metadata_values` into an additional output route. This additional output route will at the end of the previous
             output routes.
        """

        if split_by != "content_type" and metadata_values is None:
            raise ValueError(
                "If split_by is set to the name of a metadata field, you must provide metadata_values "
                "to group the documents to."
            )

        super().__init__()

        self.split_by = split_by
        self.metadata_values = metadata_values
        self.return_remaining = return_remaining

        if self.split_by != "content_type" and not isinstance(self.metadata_values, list):
            raise ValueError("Provide metadata_values if you want to split a list of Documents by a metadata field.")

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        split_by = component_params.get("split_by", "content_type")
        metadata_values = component_params.get("metadata_values", None)
        return_remaining = component_params.get("return_remaining", False)

        # If we split list of Documents by a metadata field, number of outgoing edges might change
        if split_by != "content_type" and metadata_values is not None:
            num_edges = len(metadata_values)
        else:
            num_edges = 2

        if return_remaining:
            num_edges += 1
        return num_edges

    def _split_by_content_type(self, documents: List[Document]) -> Dict[str, List[Document]]:
        mapping = {"text": "output_1", "table": "output_2"}
        split_documents: Dict[str, List[Document]] = {"output_1": [], "output_2": [], "output_3": []}
        for doc in documents:
            output_route = mapping.get(doc.content_type, "output_3")
            split_documents[output_route].append(doc)

        if not self.return_remaining and len(split_documents["output_3"]) > 0:
            # Used to avoid unnecessarily calculating other_content_types depending on logging level
            if logger.isEnabledFor(logging.WARNING):
                other_content_types = {x.content_type for x in split_documents["output_3"]}
                logger.warning(
                    "%s document(s) were skipped because they have content type(s) %s. Only the content "
                    "types 'text' and 'table' are routed.",
                    len(split_documents["output_3"]),
                    other_content_types,
                )
            del split_documents["output_3"]

        return split_documents

    def _split_by_metadata_values(self, documents: List[Document]) -> Dict[str, List[Document]]:
        split_documents = {f"output_{i + 1}": [] for i in range(len(self.metadata_values))}
        if self.return_remaining:
            split_documents[f"output_{len(self.metadata_values)}"] = []

        # TODO Simplify for loop and if statements.
        #      Support return_remaining
        for doc in documents:
            current_metadata_value = doc.meta.get(self.split_by, None)
            # Disregard current document if it does not contain the provided metadata field
            if current_metadata_value is not None:
                try:
                    # TODO Calculate index properly for list of lists
                    index = self.metadata_values.index(current_metadata_value)
                except ValueError:
                    # Disregard current document if current_metadata_value is not in the provided metadata_values
                    logger.warning(
                        "Document with id %s was skipped because the meta data value '%s' is not included in `metadata_values`.",
                        doc.id,
                        current_metadata_value,
                    )
                    continue
                split_documents[f"output_{index + 1}"].append(doc)
            else:
                logger.warning(
                    "Document with id %s was skipped because it does not have the metadata field '%s'.",
                    doc.id,
                    self.split_by,
                )
        return split_documents

    def run(self, documents: List[Document]) -> Tuple[Dict, str]:  # type: ignore
        if self.split_by == "content_type":
            split_documents = self._split_by_content_type(documents)
        else:
            split_documents = self._split_by_metadata_values(documents)
        return split_documents, "split"

    def run_batch(self, documents: Union[List[Document], List[List[Document]]]) -> Tuple[Dict, str]:  # type: ignore
        if isinstance(documents[0], Document):
            return self.run(documents)  # type: ignore
        else:
            split_documents = defaultdict(list)
            for doc_list in documents:
                results, _ = self.run(documents=doc_list)  # type: ignore
                for key in results:
                    split_documents[key].append(results[key])
            return split_documents, "split"
