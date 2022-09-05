variable "HAYSTACK_VERSION" {
  default = "main"
}

variable "GITHUB_REF" {
  default = ""
}

variable "IMAGE_NAME" {
  default = "deepset/haystack"
}

variable "IMAGE_TAG" {
  default = "local"
}

variable "HAYSTACK_EXTRAS" {
  default = ""
}

group "base" {
  targets = ["base", "base-gpu"]
}

group "api" {
  targets = ["cpu", "gpu"]
}

group "all" {
  targets = ["base", "base-gpu", "cpu", "gpu"]
}

target "docker-metadata-action" {}

target "base" {
  dockerfile = "Dockerfile.base"
  tags = ["${IMAGE_NAME}:base-${IMAGE_TAG}"]
  args = {
    build_image = "python:3.9-slim"
    base_immage = "python:3.9-slim"
    haystack_version = "${HAYSTACK_VERSION}"
    haystack_extras = notequal("",HAYSTACK_EXTRAS) ? "${HAYSTACK_EXTRAS}" : "docstores,crawler,preprocessing,ocr,onnx,beir"
    torch_scatter = "https://data.pyg.org/whl/torch-1.12.0+cpu.html"
  }
}

target "base-gpu" {
  dockerfile = "Dockerfile.base"
  tags = ["${IMAGE_NAME}:base-gpu-${IMAGE_TAG}"]
  args = {
    build_image = "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime"
    base_immage = "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime"
    haystack_version = "${HAYSTACK_VERSION}"
    haystack_extras = notequal("",HAYSTACK_EXTRAS) ? "${HAYSTACK_EXTRAS}" : "docstores-gpu,crawler,preprocessing,ocr,onnx-gpu,beir"
    torch_scatter = "https://data.pyg.org/whl/torch-1.12.1%2Bcu113.html"
  }
}

target "cpu" {
  dockerfile = "Dockerfile.api"
  tags = ["${IMAGE_NAME}:cpu-${IMAGE_TAG}"]
  args = {
    base_image_tag = "base-${IMAGE_TAG}"
  }
}

target "gpu" {
  dockerfile = "Dockerfile.api"
  tags = ["${IMAGE_NAME}:gpu-${IMAGE_TAG}"]
  args = {
    base_image_tag = "base-gpu-${IMAGE_TAG}"
  }
  platforms = [
    "linux/amd64"
  ]
}