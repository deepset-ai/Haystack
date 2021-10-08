from typing import List, Union, Dict, Optional, Tuple

import json
from haystack import BaseComponent, Document, MultiLabel
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers import pipeline


class EntityExtractor(BaseComponent):
    """
    This node is used to extract entities out of documents.
    The most common use case for this would be as a named entity extractor.
    The default model used is dslim/bert-base-NER.
    This node can be placed in a querying pipeline to perform entity extraction on retrieved documents only,
    or it can be placed in an indexing pipeline so that all documents in the document store have extracted entities.
    The entities extracted by this Node will populate Document.entities
    """

    outgoing_edges = 1

    def __init__(self,
                 model_name_or_path="dslim/bert-base-NER"):
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        token_classifier = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.model = pipeline("ner", model=token_classifier, tokenizer=tokenizer)#, aggregation_strategy="simple")

    def run(self, 
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        params: Optional[dict] = None
        ) -> Tuple[Dict, str]:
        """
        This is the method called when this node is used in a pipeline
        """
        for doc in documents:
            # In a querying pipeline, doc is a haystack.schema.Document object
            try:
                doc.meta["entities"] = self.extract(doc.text)
            # In an indexing pipeline, doc is a dictionary
            except AttributeError:
                doc["meta"]["entities"] = self.extract(doc["text"])
        output = {"documents": documents}
        return output, "output_1"

    def extract(self, text):
        """
        This function can be called to perform entity extraction when using the node in isolation.
        """
        entities = self.model(text)
        return entities


def print_ner_and_qa(output): 
    """
    [
        { 
            answer: { ... }
            entities: [ { ... }, {} ]
        }
    ]
    """
    compact_output = []
    for answer in output["answers"]:

        entities = []
        for entity in answer["meta"]["entities"]:
            if entity["start"] >= answer["offset_start_in_doc"] and entity["end"] <= answer["offset_end_in_doc"]:
                entities.append(entity["word"])  

        compact_output.append({
            "answer": answer["answer"],
            "entities": entities
        })
        
    print(json.dumps(compact_output, indent=4, default=str))
