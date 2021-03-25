import argparse, json, os, re

from src.utility import get_project_dir


class TripletModelOutputParser:
    def __init__(self, output_file, k_list=[1,5,10]):
        self.end_of_epochs = self.get_end_of_epochs(output_file)

        self.loss_pattern = "(?<=\sloss:\s)\d*\.\d*"
        self.top_k_patterns = self.get_top_k_patterns(k_list)

    def get_end_of_epochs(self, output_file):
        with open(output_file, "r") as file:
            content = file.readlines()

        end_of_epochs = []
        for line in content:
            if "validation: " in line:
                end_of_epochs.append(line)

        return end_of_epochs

    def get_top_k_patterns(self, k_list):
        top_k_patterns = {}
        for i in k_list:
            top_k_patterns[f"top_{i}"] = f"(?<=\stop_{i}\s=\s)\d*\.\d*"
        return top_k_patterns


    def get_logs(self):
        loss = []
        top_k_validation = {}
        for i, epoch in enumerate(self.end_of_epochs):
            loss.append(eval(re.search(self.loss_pattern, epoch).group(0)))
            top_k_validation[i + 1] = {
                key: eval(re.search(pattern, epoch).group(0)) for key, pattern in self.top_k_patterns.items()
            }
        logs = {
            "history": {
                "loss": loss
            },
            "TopKValidation": top_k_validation
        }
        return logs


class VAEModelOutputParser(TripletModelOutputParser):
    def __init__(self, output_file):
        super().__init__(output_file, k_list=[1,5,10])
        self.reconstruction_loss_pattern = "(?<=\sreconstruction_loss:\s)\d*\.\d*"
        self.kl_loss_pattern = "(?<=\skl_loss:\s)\d*\.\d*"

    def get_end_of_epochs(self, output_file):
        with open(output_file, "r") as file:
            content = file.readlines()

        end_of_epochs = []
        for i in range(len(content)):
            if "validation: " in content[i] and "loss" in content[i-1]:
                end_of_epochs.append(content[i-1]+content[i])

        return end_of_epochs

    def get_logs(self):
        loss = []
        reconstruction_loss = []
        kl_loss = []
        top_k_validation = {}
        for i, epoch in enumerate(self.end_of_epochs):
            loss.append(eval(re.search(self.loss_pattern, epoch).group(0)))
            reconstruction_loss.append(eval(re.search(self.reconstruction_loss_pattern, epoch).group(0)))
            kl_loss.append(eval(re.search(self.kl_loss_pattern, epoch).group(0)))
            top_k_validation[i + 1] = {
                key: eval(re.search(pattern, epoch).group(0)) for key, pattern in self.top_k_patterns.items()
            }
        logs = {
            "history": {
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss
            },
            "TopKValidation": top_k_validation
        }
        return logs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses the output files to retrieve some logging data that is added to the metafile')
    parser.add_argument('--output', default="reports/VAE_convd.out", type=str,
                        help='File containing the standard output of the training run')
    parser.add_argument('--meta', default="../models/VAE_conv2d.meta", type=str,
                        help='metadata file to add the retrieved logs too')

    args = parser.parse_args()

    output_file = os.path.join(get_project_dir(), args.output)
    if os.path.basename(args.output).startswith("VAE"):
        parser = VAEModelOutputParser(output_file)
    else:
        parser = TripletModelOutputParser(output_file)
    logs = parser.get_logs()

    with open(args.meta, "r") as file:
        metadata = json.load(file)

    metadata["logs"] = logs

    with open(args.meta, "w") as file:
        json.dump(metadata, file, indent=4)