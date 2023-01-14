from typing import List, Optional, Union

import logging

import torch
from tqdm.auto import tqdm
from transformers import pipeline

from haystack.schema import Document
from haystack.nodes.image_to_text.base import BaseImageToText
from haystack.modeling.utils import initialize_device_settings
from haystack.utils.torch_utils import ListDataset

logger = logging.getLogger(__name__)


class TransformersImageToText(BaseImageToText):
    """
    Transformer based model to generate captions for images using the HuggingFace's transformers framework

    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?pipeline_tag=image-to-text>`__

    **Example**

     ```python
        image_file_paths = ["/path/to/images/apple.jpg",
                            "/path/to/images/cat.jpg", ]

        # Generate captions
        documents = image_to_text.generate_captions(image_file_paths=image_file_paths)

        # Show results (List of Documents, containing caption and image file_path)
        print(documents)

        [
            {
                "content": "a red apple is sitting on a pile of hay",
                ...
                "meta": {
                            "image_file_path": "/path/to/images/apple.jpg",
                            ...
                        },
                ...
            },
            ...
        ]
    ```
    """

    def __init__(
        self,
        model_name_or_path: str = "nlpconnect/vit-gpt2-image-captioning",
        model_version: Optional[str] = None,
        generate_kwargs: Optional[dict] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        Load an Image To Text model from Transformers.
        See the up-to-date list of available models at
        https://huggingface.co/models?pipeline_tag=image-to-text

        :param model_name_or_path: Directory of a saved model or the name of a public model.
                                   See https://huggingface.co/models?pipeline_tag=image-to-text for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param generate_kwargs: Dictionary containing arguments for the generate method of the Hugging Face model.
                                See https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of documents to process at a time.
        :param progress_bar: Whether to show a progress bar.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
            )

        self.model = pipeline(
            task="image-to-text",
            model=model_name_or_path,
            revision=model_version,
            device=self.devices[0],
            use_auth_token=use_auth_token,
        )
        self.generate_kwargs = generate_kwargs
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    def generate_captions(
        self, image_file_paths: List[str], generate_kwargs: Optional[dict] = None, batch_size: Optional[int] = None
    ) -> List[Document]:
        """
        Generate captions for provided image files

        :param image_file_paths: Paths of the images
        :param generate_kwargs: Dictionary containing arguments for the generate method of the Hugging Face model.
                                See https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
        :param batch_size: Number of images to process at a time.
        :return: List of Documents. Document.content is the caption. Document.meta["image_file_path"] contains the image file path.
        """
        generate_kwargs = generate_kwargs or self.generate_kwargs
        batch_size = batch_size or self.batch_size

        if len(image_file_paths) == 0:
            raise AttributeError("ImageToText needs at least one filepath to produce a caption.")

        images_dataset = ListDataset(image_file_paths)

        captions: List[str] = []

        for captions_batch in tqdm(
            self.model(images_dataset, generate_kwargs=generate_kwargs, batch_size=batch_size),
            disable=not self.progress_bar,
            total=len(images_dataset),
            desc="Generating captions",
        ):
            captions.append("".join([el["generated_text"] for el in captions_batch]).strip())

        result: List[Document] = []
        for caption, image_file_path in zip(captions, image_file_paths):
            document = Document(content=caption, content_type="text", meta={"image_file_path": image_file_path})
            result.append(document)

        return result
