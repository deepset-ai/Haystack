from haystack.nodes import FARMReader
import json
import requests
from pathlib import Path

from typing import Union
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

download_links = {
    "squad2": {
        "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "test": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    },
    "squad": {
        "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "test": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    }
}

# loading json config file
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def download_file(url: str, path: Path):
    request = requests.get(url, allow_redirects=True)
    with path.open("wb") as f:
        f.write(request.content)

def download_dataset(dataset: Union[dict, str], download_folder: Path):
    train_file = "train.json"
    test_file = "test.json"
    # checking if dataset is already downloaded
    if download_folder.exists():
        assert download_folder.is_dir()
        if (download_folder/train_file).is_file() and (download_folder/test_file).is_file():
            return train_file, test_file
    if type(dataset) is str: # check if dataset needs to be looked up
        dataset = download_links[dataset]
    train = dataset["train"]
    test = dataset["test"]
    download_folder.mkdir(parents=True, exist_ok=True)
    download_file(train, download_folder/train_file)
    download_file(test, download_folder/test_file)
    return train_file, test_file

def eval(model: FARMReader, download_folder: Path, test_file: str):
    return model.eval_on_file(data_dir=download_folder, test_filename=test_file)

def train_student(model_name: str, download_folder: Path, train_file: str, test_file: str, epochs: int, batch_size: int) -> dict:
    # loading student model
    model = FARMReader(model_name_or_path=model_name)
    # training student model
    model.train(data_dir=download_folder, train_filename=train_file, n_epochs=epochs, batch_size=batch_size, caching=True, max_seq_len=512, learning_rate=3e-5)
    return eval(model, download_folder, test_file)

def train_student_with_distillation(student_name: str, teacher_name: str, download_folder: Path, train_file: str, test_file: str,
epochs: int, student_batch_size: int, teacher_batch_size: int, distillation_loss: str, distillation_loss_weight: float,
temperature: float) -> dict:
    # loading student and teacher models
    student = FARMReader(model_name_or_path=student_name)
    teacher = FARMReader(model_name_or_path=teacher_name)
    # distilling
    student.distil_from(teacher, data_dir=download_folder, train_filename=train_file, n_epochs=epochs, caching=True,
    student_batch_size=student_batch_size, teacher_batch_size=teacher_batch_size, distillation_loss=distillation_loss,
    distillation_loss_weight=distillation_loss_weight, temperature=temperature, max_seq_len=512, learning_rate=3e-5)
    return eval(student, download_folder, test_file)

def main():
    # loading config
    parent = Path(__file__).parent.resolve()
    config = load_config(parent/"distillation_config.json")
    download_folder = parent/config["download_folder"]
    student = config["student_model"]
    teacher = config["teacher_model"]

    temperatures = config["temperature"]
    distillation_loss_weights = config["distillation_loss_weight"]

    if not isinstance(temperatures, list):
        temperatures = [temperatures]

    if not isinstance(distillation_loss_weights, list):
        distillation_loss_weights = [distillation_loss_weights]
    
    # loading dataset
    logger.info("Downloading dataset")
    train_file, test_file = download_dataset(config["dataset"], download_folder)

    results_student_with_distillation = []
    for temperature in temperatures:
        for distillation_loss_weight in distillation_loss_weights:
            # distillation training
            logger.info(f"Training student with distillation (temperature: {temperature} distillation loss weight: {distillation_loss_weight}")
            results_student_with_distillation.append((temperature, distillation_loss_weight, train_student_with_distillation(student["model_name_or_path"], teacher["model_name_or_path"], download_folder,
            train_file, test_file, config["epochs"], student["batch_size"], teacher["batch_size"], config["distillation_loss"], distillation_loss_weight,
            temperature)))

    # baseline
    logger.info("Training student without distillation as a baseline")
    results_student = train_student(student["model_name_or_path"], download_folder, train_file, test_file, config["epochs"], student["batch_size"])

    # evaluating teacher as upper bound for performance
    logger.info("Evaluating teacher")
    results_teacher = eval(FARMReader(model_name_or_path=teacher["model_name_or_path"]), download_folder, test_file)

    # printing evaluation results
    logger.info("Evaluation results:")
    descriptions = ["Results of teacher", "Results of student without distillation (baseline)"] \
         + [f"Results of student with distillation (temperature: {temperature} distillation loss weight: {distillation_loss_weight}" for temperature, distillation_loss_weight, _ in results_student_with_distillation]
    for evaluation, description in zip([results_teacher, results_student] + [res for _, _, res in results_student_with_distillation], descriptions):
        logger.info(description)
        logger.info(f"EM: {evaluation['EM']}")
        logger.info(f"F1: {evaluation['f1']}")
        logger.info(f"Top n accuracy: {evaluation['top_n_accuracy']}")



if __name__ == "__main__":
    main()