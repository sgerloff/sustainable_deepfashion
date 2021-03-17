import joblib, json, os, subprocess, zipfile

from src.utility import get_project_dir, remove_project_dir


def instance_from_string(module_path):
    path_list = module_path.split(".")
    class_name = path_list[-1]
    directory_of_module = ".".join(path_list[:-1])
    module = __import__(directory_of_module, fromlist=[class_name])
    return getattr(module, class_name)


def load_instruction(path):
    instruction_file = os.path.join(get_project_dir(), "instructions", path)
    with open(instruction_file, "r") as file:
        instruction_ = json.load(file)
    return instruction_


def load_dataframe(path):
    return joblib.load(os.path.join(get_project_dir(), path))


def evaluate_values_of_dict(dict_in):
    return {key: eval(str(value)) for key, value in dict_in.items()}


class InstructionParser:
    def __init__(self, instruction_path):
        self.identifier = self.set_identifier(instruction_path)
        self.instruction = load_instruction(instruction_path)
        self.model_factory = self.load_model_factory()
        self.set_basemodel_freeze_ratio()

        self.model_save_path = None
        self.tensorboard_log_dir = None

        self.metadata_path = None
        self.metadata = None

    def set_identifier(self, path):
        basename = os.path.basename(path)
        identifier = os.path.splitext(basename)[0]
        if os.path.isfile(self.get_metadata_path(identifier)):
            i = 0
            while os.path.isfile(self.get_metadata_path(identifier + "-" + str(i))):
                i += 1
            return identifier + "-" + str(i)
        else:
            return identifier

    def load_model_factory(self):
        self.instruction["model"]["kwargs"] = evaluate_values_of_dict(self.instruction["model"]["kwargs"])
        return instance_from_string(
            self.instruction["model"]["factory"])(**self.instruction["model"]["kwargs"])

    def set_basemodel_freeze_ratio(self):
        ratio = None
        if "basemodel_freeze_ratio" in self.instruction["model"].keys():
            ratio = self.instruction["model"]["basemodel_freeze_ratio"]

        if ratio is not None:
            self.model_factory.set_basemodel_freeze_ratio(ratio)

    def load_dataset(self, dataset_instruction):
        dataframe = load_dataframe(dataset_instruction["dataframe"])
        dataset_factory = instance_from_string(dataset_instruction["factory"])(
            dataframe,
            preprocessor=self.model_factory.preprocessor(),
            input_shape=self.model_factory.input_shape
        )
        return dataset_factory.get_dataset(**dataset_instruction["kwargs"])

    def get_train_dataset(self):
        return self.load_dataset(self.instruction["train_data"])

    def get_validation_dataset(self):
        return self.load_dataset(self.instruction["validation_data"])

    def get_model(self):
        return self.model_factory.get_model()

    def get_loss(self):
        if self.instruction["loss"] != "None":
            return instance_from_string(self.instruction["loss"]["loss"])(**self.instruction["loss"]["kwargs"])
        else:
            return None

    def get_optimizer(self):
        if self.instruction["optimizer"] != "None":
            return instance_from_string(self.instruction["optimizer"]["optimizer"])(
                **self.instruction["optimizer"]["kwargs"])
        else:
            return None

    def get_metric(self):
        if self.instruction["metric"] != "None":
            return instance_from_string(self.instruction["metric"]["metric"])(
                **self.instruction["metric"]["kwargs"]).score
        else:
            return None

    def get_fit_kwargs(self):
        return self.instruction["model"]["fit"]["kwargs"]

    def get_callbacks(self):
        callbacks = []
        for key in self.instruction["callbacks"].keys():
            self.instruction["callbacks"][key] = self.replace_default_filename_in_kwargs(
                self.instruction["callbacks"][key])
            callback = instance_from_string(key)(**self.instruction["callbacks"][key])

            if key == "src.models.callbacks.Checkpoint":
                self.model_save_path = callback.filepath
            if key == "src.models.callbacks.Tensorboard":
                self.tensorboard_log_dir = callback.log_dir

            callbacks.append(callback)
        return callbacks

    def replace_default_filename_in_kwargs(self, dict_in):
        for key, value in dict_in.items():
            if value == "__default_filename__":
                dict_in[key] = self.identifier

        return dict_in

    def get_cleanup_cmd(self):
        cmd = self.instruction["cleanup_cmd"]
        cmd = cmd.replace("__default_filename__", self.identifier)
        return cmd

    def write_metadata(self):
        if self.model_save_path is not None:
            self.metadata_path = self.get_metadata_path(self.identifier)
            self.metadata = {
                "saved_model": remove_project_dir(self.model_save_path),
                "tensorboard_log_dir": remove_project_dir(self.tensorboard_log_dir),
                "git_commit": subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8"),
                "instruction": self.instruction
            }
            with open(self.metadata_path, "w") as file:
                json.dump(self.metadata, file, indent=4)

    @staticmethod
    def get_metadata_path(identifier):
        return os.path.join(get_project_dir(), "models", identifier + ".meta")

    def zip_results(self):
        with zipfile.ZipFile(self.identifier + ".zip", "w") as zipf:
            zipf.write(remove_project_dir(self.metadata_path))
            zipf.write(self.metadata["saved_model"])
            zipf.write(self.metadata["tensorboard_log_dir"])


if __name__ == "__main__":
    ip = InstructionParser("efficientnetb0_frozen.json")
    callbacks = ip.get_callbacks()
    print(callbacks[0])
