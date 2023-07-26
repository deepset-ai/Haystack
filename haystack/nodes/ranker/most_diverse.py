import logging
from pathlib import Path
from typing import List, Optional, Union

from haystack.nodes import BaseRanker
from haystack.schema import Document
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from sentence_transformers import SentenceTransformer
    from haystack.modeling.utils import initialize_device_settings  # pylint: disable=ungrouped-imports


class MostDiverseRanker(BaseRanker):
    """
    Implements a document ranking algorithm that orders documents in such a way as to maximize the overall diversity
    of the documents.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "all-MiniLM-L6-v2",
        use_gpu: Optional[bool] = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
    ):
        """
        Initialize a MostDiverseRanker.

        :param model_name_or_path: Path to a pretrained sentence-transformers model.
        :param use_gpu: Whether to use GPU (if available). If no GPUs are available, it falls back on a CPU.
        :param devices: List of torch devices (for example, cuda:0, cpu, mps) to limit inference to specific devices.
        """
        torch_and_transformers_import.check()
        super().__init__()
        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=True)
        self.model = SentenceTransformer(model_name_or_path, device=str(self.devices[0]))

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Rank the documents based on their diversity and return the top_k documents.

        :param query: The query.
        :param documents: A list of Document objects that should be ranked.
        :param top_k: The maximum number of documents to return.

        :return: A list of top_k documents ranked based on diversity.
        """
        if query is None or len(query) == 0:
            raise ValueError("Query is empty")
        if documents is None or len(documents) == 0:
            raise ValueError("No documents to choose from")

        diversity_sorted = self.most_diverse_order(query=query, documents=documents)
        return diversity_sorted[:top_k]

    def most_diverse_order(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Orders the given list of documents to maximize diversity. The algorithm first calculates embeddings for
        each document and the query. It starts by selecting the document that is semantically closest to the query.
        Then, for each remaining document, it selects the one that, on average, is least similar to the already
        selected documents. This process continues until all documents are selected, resulting in a list where
        each subsequent document contributes the most to the overall diversity of the selected set.

        :param query: The search query.
        :param documents: The list of Document objects to be ranked.

        :return: A list of documents ordered to maximize diversity.
        """

        # Calculate embeddings
        doc_embeddings: torch.Tensor = self.model.encode([d.content for d in documents], convert_to_tensor=True)
        query_embedding: torch.Tensor = self.model.encode([query], convert_to_tensor=True)

        n = len(documents)
        selected: List[int] = []

        # Compute the similarity matrix between documents
        doc_sim_matrix: torch.Tensor = doc_embeddings @ doc_embeddings.T

        # Compute the similarity vector between the query and documents
        query_doc_sim: torch.Tensor = query_embedding @ doc_embeddings.T

        # Start with the document with the highest similarity to the query
        selected.append(int(torch.argmax(query_doc_sim).item()))

        while len(selected) < n:
            # Indices of unselected documents
            unselected: List[int] = [i for i in range(n) if i not in selected]

            # Compute unselected/selected similarity matrix
            # First iteration through the loop it's shape is (len(unselected), 1)
            # and then it's (len(unselected), 2) and so on until (1, len(selected)) in the last iteration
            unselected_selected_sim_matrix = doc_sim_matrix[unselected, :][:, selected]

            # Compute the average similarity score of each unselected document to the selected group of documents
            avg_sim_unselected: torch.Tensor = torch.mean(unselected_selected_sim_matrix, dim=-1)

            # Select the document with the lowest average similarity score
            index_unselected = int(torch.argmin(avg_sim_unselected).item())
            selected.append(unselected[index_unselected])

        ranked_docs: List[Document] = [documents[i] for i in selected]

        return ranked_docs

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        """
        Rank the documents based on their diversity and return the top_k documents.

        :param queries: The queries.
        :param documents: A list (or a list of lists) of Document objects that should be ranked.
        :param top_k: The maximum number of documents to return.
        :param batch_size: The number of documents to process in one batch.

        :return: A list (or a list of lists) of top_k documents ranked based on diversity.
        """
        if queries is None or len(queries) == 0:
            raise ValueError("No queries to choose from")
        if documents is None or len(documents) == 0:
            raise ValueError("No documents to choose from")
        if len(documents) > 0 and isinstance(documents[0], Document):
            # Docs case 1: single list of Documents -> rerank single list of Documents based on single query
            if len(queries) != 1:
                raise ValueError("Number of queries must be 1 if a single list of Documents is provided.")
            return self.predict(query=queries[0], documents=documents, top_k=top_k)  # type: ignore
        else:
            # Docs case 2: list of lists of Documents -> rerank each list of Documents based on corresponding query
            # If queries contains a single query, apply it to each list of Documents
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise ValueError("Number of queries must be equal to number of provided Document lists.")

            results = []
            for query, cur_docs in zip(queries, documents):
                results.append(self.predict(query=query, documents=cur_docs, top_k=top_k))  # type: ignore
            return results
