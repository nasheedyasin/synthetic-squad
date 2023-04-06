import json
import torch
import string
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, pipeline
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Literal, Optional, Sequence, Tuple


TASKS = ('same', 'hvm', 'aa')


class SynDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame
    ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # The data sample looks like:
        # Predict: (index, title, generation)
        # Other stages: (index, title, generation, alg)
        return (index, *self.samples.iloc[index])



class SynBatcher:
    def __init__(
        self,
        tnkzr_path: str,
        has_targets: bool = True,
        concat_title_and_generation: bool = False
    ):
        """
        Args:
            tnkzr_path (str): Path to load the `Transformers` tokenizer
            to be used.
            has_targets (bool): Does the dataset have target information.
            Defaults to True.
            concat_title_and_generation (bool): Whether to concatenate the title
            to the generation. Could be useful for non-autoregressive models.
        """
        self.has_targets = has_targets
        self.tokenizer = AutoTokenizer.from_pretrained(tnkzr_path)

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'

        self.concat_title_and_generation = concat_title_and_generation

    def __call__(self, batch: Sequence):
        """Use this function as the `collate_fn` mentioned earlier.
        """
        ids = torch.tensor([int(sample[0]) for sample in batch], dtype=torch.int32)

        if self.concat_title_and_generation:
            text_tokens = self.tokenizer(
                [f"{sample[1]} {sample[2]}" for sample in batch],
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
        else:
            # The tokenization is done with the title as sentence 1 and
            # generation as sentence 2
            text_tokens = self.tokenizer(
                [sample[1] for sample in batch], # The title
                [sample[2] for sample in batch], # The generation
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )

        if not self.has_targets:
            return ids, text_tokens

        targets = torch.tensor(
            [sample[3]for sample in batch],
            dtype=torch.long
        )

        return ids, text_tokens, targets


class SynDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batcher: SynBatcher,
        task: Literal['same', 'hvm', 'aa'],
        label2id: Dict[str, int],
        batch_size: int = 16,
        srcs_to_keep: List[str] = ['aa_paper'],
    ):
        """
        Args:
            data_path: Path to data CSV files.
            data_path (str): Path to the pickled annotations file.
            batcher (SynBatcher): Custom data batching logic.
            task (Literal[`same`, `hvm`, `aa`]): Task under consideration.
                `aa`: Authorship Attribution, `same`: Same Method or Not, `hvm`: Human vs. Machine
            label2id (Dict[str, int]): A dictionary mapping the generation algs
            to an integer Id.
            srcs_to_keep (List[str]): Data sources to keep.
        """
        super().__init__()
        
        if task not in TASKS:
            raise NotImplementedError(f"{task} has not been implemented")

        self.task = task
        self.batcher = batcher
        self.data_path = data_path
        self.label2id = label2id
        self.batch_size = batch_size
        self.srcs_to_keep = srcs_to_keep

    def setup(
        self,
        stage: Optional[str] = None,
        train_fraction: float = 0.88
    ) -> None:
        """Read in the data csv file and perform splitting here.
        Args:
            train_fraction (float): Fraction to use as training data.
        """
        df_base = pd.read_csv(self.data_path)

        # Only keep the source in consideration
        df_base = df_base[df_base['src'].isin(self.srcs_to_keep)]

        # Drop null values
        df_base.dropna(inplace=True)

        # Binary labels for `hvm` task
        if self.task == 'hvm':
            df_base['alg'] = df_base['alg'].apply(
                lambda x: int(x != 'human')
            )
        # Convert alg to ids
        else:
            if "alg" in df_base.columns:
                df_base['alg'] = df_base['alg'].apply(
                    lambda x: self.label2id[x]
                )

        if stage == "fit" or stage is None:
            # Split the training dataset into train and validation.
            df_tr, df_val = train_test_split(
                df_base[['title', 'generation', 'alg']],
                stratify=df_base[['alg']],
                train_size=train_fraction,
                random_state=44
            )

            # Train Dataset
            self.train_dataset = SynDataset(df_tr)
            # Validation Dataset
            self.val_dataset = SynDataset(df_val)

        if stage == "test" or stage is None:
            self.test_dataset = SynDataset(df_base[['title', 'generation', 'alg']])

        if stage == "predict" or stage is None:
            self.pred_dataset = SynDataset(df_base[['title', 'generation']])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher,
            shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )
