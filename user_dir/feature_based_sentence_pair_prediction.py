import logging
import os

import numpy as np
import torch

from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

from .data.feature_dropout_dataset import FeatureDropoutDataset
from .data.truncate_dataset import TruncateDataset
from .data.data_utils import load_distillation_target, load_indexed_dataset
from .feature_based_sentence_prediction import FeatureBasedSentencePredictionTask

logger = logging.getLogger(__name__)

@register_task("feature_based_sentence_pair_prediction")
class FeatureBasedSentencePairPredictionTask(SentencePredictionTask):

    def add_args(parser):
        SentencePredictionTask.add_args(parser)
        parser.add_argument("--distillation-target", action="store_true", default=False)
        parser.add_argument("--use-focal", action="store_true", default=False)
        parser.add_argument("--exclude-last-shard", action="store_true", default=False)
        parser.add_argument("--exclude-last-instances", type=int, default=0)
        parser.add_argument("--init-embed", type=str, default=None)

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        # data_dict.pad_to_multiple_(args.model_parallel_size * 8)
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        if not args.distillation_target:
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, "label", "dict.txt"),
                source=False,
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        # the same as sentence prediction task
        def get_path(key, split):
            return os.path.join(self.args.data, key, split)

        # the same as sentence prediction task
        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            try:
                dataset = load_indexed_dataset(
                    split_path,
                    dictionary,
                    self.args.dataset_impl,
                    combine=combine,
                    exclude_last_shard=self.args.exclude_last_shard,
                )
            except Exception as e:
                if "StorageException: [404] Path not found" in str(e):
                    logger.warning(f"dataset {e} not found")
                    dataset = None
                else:
                    raise e
            return dataset    

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )

        # the make_dataset will automatically add </s> at the end; we don't need this so we strip it.
        input0 = StripTokenDataset(input0, id_to_strip=self.source_dictionary.eos())
        src_tokens = input0

        # dropout features (for regularization) + truncate if there are too many features
        src_tokens = FeatureDropoutDataset(
            src_tokens, 
            self.args.feature_dropout if split == "train" else 0.0, # don't use feature dropout for validation
            self.max_positions(), 
            self.args.seed
        )

        input1 = make_dataset("input1", self.source_dictionary)
        assert input1 is not None, "could not find dataset: {}".format(
            get_path("input1", split)
        )

        # the make_dataset will automatically add </s> at the end; we don't need this so we strip it.
        input1 = StripTokenDataset(input1, id_to_strip=self.source_dictionary.eos())
        src_tokens1 = input1

        # dropout features (for regularization) + truncate if there are too many features
        src_tokens1 = FeatureDropoutDataset(
            src_tokens1, 
            self.args.feature_dropout if split == "train" else 0.0, # don't use feature dropout for validation
            self.max_positions(), 
            self.args.seed
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "input0_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "input0_lengths": NumelDataset(src_tokens, reduce=False),
                "input1_tokens": RightPadDataset(src_tokens1, pad_idx=self.source_dictionary.pad()),
                "input1_lengths": NumelDataset(src_tokens1, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        # gold labels
        label_dataset = make_dataset("label", self.label_dictionary)

        if label_dataset is not None:
            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(
                        label_dataset,
                        id_to_strip=self.label_dictionary.eos(),
                    ),
                    offset=-self.label_dictionary.nspecial,
                )
            )

        # distill targets
        if self.args.distillation_target:
            # load distillation target from a pre-computed npy file
            label_path = "{0}".format(get_path("distill", split))
            if os.path.exists(label_path + ".npy"):
                distillation_target = load_distillation_target(label_path, exclude_last_shard=self.args.exclude_last_shard,)
                dataset.update(distill_target=distillation_target)

            label_path = "{0}_freq.npy".format(get_path("distill", split))
            if os.path.exists(label_path) and self.args.use_focal:
                logger.info("Using focal loss, loading weights from {}".format(label_path))
                freq = np.load(label_path)
                freq = freq / len(label_dataset)
                freq = freq.astype(np.float32)
                dataset.update(freq=RawLabelDataset(freq))
            else:
                freq = np.zeros(len(label_dataset))
                dataset.update(freq=RawLabelDataset(freq))

        # turn the dictionary into a real dataset instance?
        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if split == "train" and self.args.exclude_last_instances > 0:
            actual_size = len(nested_dataset) - self.args.exclude_last_instances
            nested_dataset = TruncateDataset(nested_dataset, actual_size)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(nested_dataset))

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset

        # for i in range(5):
        #     print(dataset[i])

        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_classes,
        )

        # load pre-computed embedding table
        if getattr(args, "init_embed", None) is not None:
            logger.info("Loading embeddings from {}".format(args.init_embed))

            embed_table = np.load(args.init_embed)
            embed_tokens = model.encoder.embed_tokens.weight.detach().numpy()

            st = self.dictionary.nspecial # the first few tokens are special
            ed = st + embed_table.shape[0]

            embed_tokens[st: ed, :] = embed_table
            model.encoder.embed_tokens.from_pretrained(torch.tensor(embed_tokens))

        return model