from types import SimpleNamespace

import torch
import torchaudio as ta

import shared

logger = shared.fetch_main_logger()


def load_speechcommands_dataset(data_path):
    logger.info(f"Loading SpeechCommands dataset from {data_path}")

    train_dataset = MySpeechCommands(root=data_path, download=False, subset="training")
    test_dataset = MySpeechCommands(root=data_path, download=False, subset="testing")
    return SimpleNamespace(train=train_dataset, val=test_dataset), train_dataset.num_classes


class MySpeechCommands(ta.datasets.SPEECHCOMMANDS):
    def __init__(self, root, download, subset="training"):

        super().__init__(root=root, download=True, subset=subset)
        self.sample_rate = 16000
        self.new_sample_rate = 8000
        self.labels = [
            "backward",
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "follow",
            "forward",
            "four",
            "go",
            "happy",
            "house",
            "learn",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "visual",
            "wow",
            "yes",
            "zero",
        ]
        self.desired_length = 8000  # 1 sec duration
        self.num_classes = len(self.labels)

    def __getitem__(self, i):
        waveform, _, word, _, _ = super().__getitem__(i)
        x = self.preprocess(waveform)
        y = self.word_to_index(word)
        return x, y

    def word_to_index(self, word):
        # Return the position of the word in the labels
        return torch.tensor(self.labels.index(word))

    def preprocess(self, waveform):
        # Resample the waveform to the desired sample rate
        waveform = ta.transforms.Resample(orig_freq=self.sample_rate, new_freq=self.new_sample_rate)(waveform)
        # Pad the waveform to the desired length
        if waveform.size(1) < self.desired_length:
            amount_to_pad = self.desired_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, amount_to_pad))
        return waveform
