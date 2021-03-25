import joblib, json, os, subprocess, zipfile, shutil

from src.utility import get_project_dir, remove_project_dir, savely_unfreeze_layers_of_model


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


def load_metadata(path):
    metadata_file = os.path.join(get_project_dir(), "models", path)
    with open(metadata_file, "r") as file:
        metadata_ = json.load(file)
    return metadata_


def load_model_from_metadata(path, compile=True):
    metadata = load_metadata(path)
    ip = InstructionParser(metadata["instruction"], is_dict=True)
    model = ip.get_model()
    model.load_weights(os.path.join(get_project_dir(), metadata["saved_model"]))
    compile_kwargs = get_compile_kwargs_from_instruction_parser(ip)
    if compile:
        model.compile(**compile_kwargs)
    return model


def get_compile_kwargs_from_instruction_parser(instruction_parser):
    compile_kwargs = {}
    if instruction_parser.get_loss() is not None:
        compile_kwargs["loss"] = instruction_parser.get_loss()
    if instruction_parser.get_optimizer() is not None:
        compile_kwargs["optimizer"] = instruction_parser.get_optimizer()
    if instruction_parser.get_metric() is not None:
        compile_kwargs["metrics"] = [instruction_parser.get_metric()]
    return compile_kwargs


def load_dataframe(path):
    return joblib.load(os.path.join(get_project_dir(), path))


def evaluate_list_and_tuples_in_dict(dict_in):
    for key, value in dict_in.items():
        if isinstance(value, str):
            if string_is_list(value) or string_is_tuple(value):
                dict_in[key] = eval(value)
    return dict_in


def string_is_list(input):
    return input.startswith("[") and input.endswith("]")


def string_is_tuple(input):
    return input.startswith("(") and input.endswith(")")


class InstructionParser:
    def __init__(self, instruction, is_dict=False):
        if is_dict:
            self.identifier = "instruction_from_metadata_THIS_SHOULD_NOT_HAPPEN"
            self.instruction = instruction
        else:
            self.identifier = self.set_identifier(instruction)
            self.instruction = load_instruction(instruction)

            self.copy_instruction()

            self.model_save_path = None
            self.tensorboard_log_dir = None
            self.best_top_1_model_path = None

            self.metadata_path = None
            self.metadata = None

            self.to_load_weights = None

        if self.instruction["model"]["load"] == "None":
            self.model_factory = self.load_model_factory()
        else:
            self.model_factory = self.load_model_factory_from_metafile()

    def copy_instruction(self):
        if "copy_instruction" in self.instruction.keys():
            parent_instruction = load_instruction(self.instruction["copy_instruction"])
            self.instruction.pop("copy_instruction")

            for key in self.instruction.keys():
                parent_instruction = self.overwrite_values(parent_instruction, self.instruction, key)

            self.instruction = parent_instruction

    def overwrite_values(self, target_dict, source_dict, key):
        if key in target_dict.keys():
            if isinstance(source_dict[key], dict) and isinstance(target_dict[key], dict):
                for k in source_dict[key].keys():
                    target_dict[key] = self.overwrite_values(target_dict[key], source_dict[key], k)
            else:
                target_dict[key] = source_dict[key]
        else:
            target_dict[key] = source_dict[key]
        return target_dict

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
        self.instruction["model"]["kwargs"] = evaluate_list_and_tuples_in_dict(self.instruction["model"]["kwargs"])
        self.model_factory = instance_from_string(
            self.instruction["model"]["factory"]
        )(**self.instruction["model"]["kwargs"])

        self.set_model_factory_basemodel_freeze_ratio()
        return self.model_factory

    def set_model_factory_basemodel_freeze_ratio(self):
        ratio = None
        if "basemodel_freeze_ratio" in self.instruction["model"].keys():
            ratio = self.instruction["model"]["basemodel_freeze_ratio"]

        if ratio is not None:
            self.model_factory.set_basemodel_freeze_ratio(ratio)

    def load_model_factory_from_metafile(self):
        metadata = load_metadata(self.instruction["model"]["load"])
        instruction_parser_from_meta = InstructionParser(metadata["instruction"], is_dict=True)

        self.to_load_weights = os.path.join(get_project_dir(), metadata["saved_model"])
        return instruction_parser_from_meta.model_factory

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
        if self.instruction["validation_data"] == "None":
            return None
        else:
            return self.load_dataset(self.instruction["validation_data"])

    def get_model(self):
        model = self.model_factory.get_model()
        if self.instruction["model"]["load"] != "None":
            model.load_weights(self.to_load_weights)
            model = self.apply_basemodel_freeze_ratio(model)
        return model

    def apply_basemodel_freeze_ratio(self, model):
        ratio = None
        if "basemodel_freeze_ratio" in self.instruction["model"].keys():
            ratio = self.instruction["model"]["basemodel_freeze_ratio"]

        if ratio is not None:
            model.layers[1] = savely_unfreeze_layers_of_model(model.layers[1], ratio)
        return model

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
            self.instruction["callbacks"][key] = evaluate_list_and_tuples_in_dict(self.instruction["callbacks"][key])

            self.instruction["callbacks"][key] = self.replace_default_filename_in_kwargs(
                self.instruction["callbacks"][key])

            kwargs = self.instruction["callbacks"][key]
            if key == "src.models.callbacks.TopKValidation":
                kwargs["preprocessor"] = self.model_factory.preprocessor()

            callback = instance_from_string(key)(**kwargs)

            if key == "src.models.callbacks.TopKValidation":
                self.best_top_1_model_path = callback.best_model_filepath
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

    def write_metadata(self, logs=None):
        if self.model_save_path is not None:
            self.metadata_path = self.get_metadata_path(self.identifier)
            self.metadata = {
                "saved_model": remove_project_dir(self.model_save_path),
                "tensorboard_log_dir": remove_project_dir(self.tensorboard_log_dir),
                "best_top_1_model": remove_project_dir(self.best_top_1_model_path),
                "git_commit": subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8"),
                "instruction": self.instruction,
                "logs": logs
            }
            with open(self.metadata_path, "w") as file:
                json.dump(self.metadata, file, indent=4)

    @staticmethod
    def get_metadata_path(identifier):
        return os.path.join(get_project_dir(), "models", identifier + ".meta")

    def zip_results(self):
        shutil.make_archive(self.identifier, "zip", base_dir=self.metadata["tensorboard_log_dir"],
                            root_dir=get_project_dir())
        with zipfile.ZipFile(self.identifier + ".zip", "a") as zipf:
            zipf.write(remove_project_dir(self.metadata_path))
            zipf.write(self.metadata["saved_model"])
            if self.best_top_1_model_path is not None:
                zipf.write(remove_project_dir(self.best_top_1_model_path))


if __name__ == "__main__":
    ip = InstructionParser("mobilenet_v2_unfreeze_ratio_0.5.json")
    model = ip.get_model()
    ip.get_callbacks()
