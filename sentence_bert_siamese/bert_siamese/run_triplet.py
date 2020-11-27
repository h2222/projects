# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import model_triplet as train_model
import my_modeling2 as test_model
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Required parameters
flags.DEFINE_string(
    "data_dir",
    None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.",
)

flags.DEFINE_string("train_data", "train.csv", "Train data")
flags.DEFINE_string("dev_data", "dev.csv", "Valid data")
flags.DEFINE_string("predict_data", "test.csv", "Predict data")
flags.DEFINE_string("predict_output", "test_results.tsv", "Predict data")

flags.DEFINE_string(
    "bert_config_file",
    None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.",
)

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "vocab_file", None, "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_string(
    "output_dir",
    None,
    "The output directory where the model checkpoints will be written.",
)
# Other parameters

flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Initial checkpoint (usually from a pre-trained BERT model).",
)

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.",
)

flags.DEFINE_integer(
    "max_seq_length",
    64,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_bool("raw_data", True, "Whether to read raw data and convert to tfrecord.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False, "Whether to run the model in inference mode on the test set."
)

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 3.0, "Total number of training epochs to perform."
)

flags.DEFINE_float(
    "warmup_proportion",
    0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.",
)

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000, "How often to save the model checkpoint."
)

flags.DEFINE_integer(
    "iterations_per_loop", 1000, "How many steps to make in each estimator call."
)

flags.DEFINE_string("gpu", "2", "Which gpu to use.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name",
    None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.",
)

tf.flags.DEFINE_string(
    "tpu_zone",
    None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.",
)

tf.flags.DEFINE_string(
    "gcp_project",
    None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.",
)

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores",
    8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.",
)


class InputExample_train(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_p=None, text_n=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_p = text_p
        self.text_n = text_n


class InputExample_test(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures_train(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_a_ids,
        input_p_ids,
        input_n_ids,
        input_a_mask,
        input_p_mask,
        input_n_mask,
        segment_a_ids,
        segment_p_ids,
        segment_n_ids,
        is_real_example=True,
    ):
        self.input_a_ids = input_a_ids
        self.input_p_ids = input_p_ids
        self.input_n_ids = input_n_ids
        self.input_a_mask = input_a_mask
        self.input_p_mask = input_p_mask
        self.input_n_mask = input_n_mask
        self.segment_a_ids = segment_a_ids
        self.segment_p_ids = segment_p_ids
        self.segment_n_ids = segment_n_ids
        self.is_real_example = is_real_example


class InputFeatures_test(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_a_ids,
        input_b_ids,
        input_a_mask,
        input_b_mask,
        segment_a_ids,
        segment_b_ids,
        label_id,
        is_real_example=True,
    ):
        self.input_a_ids = input_a_ids
        self.input_b_ids = input_b_ids
        self.input_a_mask = input_a_mask
        self.input_b_mask = input_b_mask
        self.segment_a_ids = segment_a_ids
        self.segment_b_ids = segment_b_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MyProcessor(DataProcessor):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        file_path = os.path.join(data_dir, FLAGS.train_data)
        with open(file_path, "r") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader[1:]):
            guid = "train-%d" % index
            split_line = line.strip().split(",")
            text_a = tokenization.convert_to_unicode(split_line[2])
            text_p = tokenization.convert_to_unicode(split_line[3])
            text_n = tokenization.convert_to_unicode(split_line[4])
            # print(label,type(label))
            examples.append(
                InputExample_train(guid=guid, text_a=text_a, text_p=text_p, text_n=text_n)
            )
            # print(examples)

        return examples

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        file_path = os.path.join(data_dir, FLAGS.dev_data)
        with open(file_path, "r") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader[1:]):
            guid = "dev-%d" % index
            split_line = line.strip().split(",")
            text_a = tokenization.convert_to_unicode(split_line[2])
            text_b = tokenization.convert_to_unicode(split_line[3])
            label = split_line[8]
            examples.append(
                InputExample_test(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        file_path = os.path.join(data_dir, FLAGS.predict_data)
        with open(file_path, "r") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader[1:]):
            guid = "test-%d" % index
            split_line = line.strip().split(",")
            text_a = tokenization.convert_to_unicode(split_line[2])
            text_b = tokenization.convert_to_unicode(split_line[3])
            label = split_line[8]
            examples.append(
                InputExample_test(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1"]


def convert_text_to_tokens(text, max_seq_length, tokenizer):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[: (max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    def padding_zero(seq, max_seq_length):
        return seq + [0] * (max_seq_length - len(seq))

    input_ids = padding_zero(input_ids, max_seq_length)
    input_mask = padding_zero(input_mask, max_seq_length)
    segment_ids = padding_zero(segment_ids, max_seq_length)
    return tokens, input_ids, input_mask, segment_ids


def convert_single_example_train(ex_index, example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures_train(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False,
        )

    tokens_a, input_a_ids, input_a_mask, segment_a_ids = convert_text_to_tokens(example.text_a, max_seq_length, tokenizer)
    tokens_p, input_p_ids, input_p_mask, segment_p_ids = convert_text_to_tokens(example.text_p, max_seq_length, tokenizer)
    tokens_n, input_n_ids, input_n_mask, segment_n_ids = convert_text_to_tokens(example.text_n, max_seq_length, tokenizer)

    assert len(input_a_ids) == len(input_p_ids) == len(input_n_ids) == max_seq_length
    assert len(input_a_mask) == len(input_p_mask) == len(input_n_mask) == max_seq_length
    assert len(segment_a_ids) == len(segment_p_ids) == len(segment_n_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info(
            "tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens_a])
        )
        tf.logging.info(
            "tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens_p])
        )
        tf.logging.info(
            "tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens_n])
        )
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_a_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_a_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_a_ids]))

    feature = InputFeatures_train(
        input_a_ids=input_a_ids,
        input_p_ids=input_p_ids,
        input_n_ids=input_n_ids,
        input_a_mask=input_a_mask,
        input_p_mask=input_p_mask,
        input_n_mask=input_n_mask,
        segment_a_ids=segment_a_ids,
        segment_p_ids=segment_p_ids,
        segment_n_ids=segment_n_ids,
        is_real_example=True,
    )
    return feature


def convert_single_example_test(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures_test(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False,
        )
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a, input_a_ids, input_a_mask, segment_a_ids = convert_text_to_tokens(example.text_a, max_seq_length, tokenizer)
    tokens_b, input_b_ids, input_b_mask, segment_b_ids = convert_text_to_tokens(example.text_b, max_seq_length, tokenizer)

    assert len(input_a_ids) == len(input_b_ids) == max_seq_length
    assert len(input_a_mask) == len(input_b_mask) == max_seq_length
    assert len(segment_a_ids) == len(segment_b_ids) == max_seq_length
    label_id = label_map[example.label]

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info(
            "tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens_a])
        )
        tf.logging.info(
            "tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens_b])
        )
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_a_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_a_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_a_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures_test(
        input_a_ids=input_a_ids,
        input_b_ids=input_b_ids,
        input_a_mask=input_a_mask,
        input_b_mask=input_b_mask,
        segment_a_ids=segment_a_ids,
        segment_b_ids=segment_b_ids,
        label_id=label_id,
        is_real_example=True,
    )
    return feature


def file_based_convert_examples_to_features_train(
    examples, label_list, max_seq_length, tokenizer, output_file
):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example_train(
            ex_index, example, max_seq_length, tokenizer
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_a_ids"] = create_int_feature(feature.input_a_ids)
        features["input_p_ids"] = create_int_feature(feature.input_p_ids)
        features["input_n_ids"] = create_int_feature(feature.input_n_ids)
        features["input_a_mask"] = create_int_feature(feature.input_a_mask)
        features["input_p_mask"] = create_int_feature(feature.input_p_mask)
        features["input_n_mask"] = create_int_feature(feature.input_n_mask)
        features["segment_a_ids"] = create_int_feature(feature.segment_a_ids)
        features["segment_p_ids"] = create_int_feature(feature.segment_p_ids)
        features["segment_n_ids"] = create_int_feature(feature.segment_n_ids)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_convert_examples_to_features_test(
    examples, label_list, max_seq_length, tokenizer, output_file
):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example_test(
            ex_index, example, label_list, max_seq_length, tokenizer
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_a_ids"] = create_int_feature(feature.input_a_ids)
        features["input_b_ids"] = create_int_feature(feature.input_b_ids)
        features["input_a_mask"] = create_int_feature(feature.input_a_mask)
        features["input_b_mask"] = create_int_feature(feature.input_b_mask)
        features["segment_a_ids"] = create_int_feature(feature.segment_a_ids)
        features["segment_b_ids"] = create_int_feature(feature.segment_b_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    if is_training:
        name_to_features = {
            "input_a_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_p_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_n_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_a_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "input_p_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "input_n_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_a_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_p_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_n_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }
    else:
        name_to_features = {
            "input_a_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_b_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_a_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "input_b_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_a_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_b_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        )

        return d

    return input_fn


def create_train_model(
    bert_config,
    is_training,
    input_a_ids,
    input_p_ids,
    input_n_ids,
    input_a_mask,
    input_p_mask,
    input_n_mask,
    segment_a_ids,
    segment_p_ids,
    segment_n_ids,
    use_one_hot_embeddings,
    margin
):
 
    model = train_model.BertModel(
        config=bert_config,
        is_training=is_training,
        input_a_ids=input_a_ids,
        input_p_ids=input_p_ids,
        input_n_ids=input_n_ids,
        input_a_mask=input_a_mask,
        input_p_mask=input_p_mask,
        input_n_mask=input_n_mask,
        token_a_type_ids=segment_a_ids,
        token_p_type_ids=segment_p_ids,
        token_n_type_ids=segment_n_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )
    output_layer_a = model.get_pooled_output_a()
    output_layer_p = model.get_pooled_output_p()
    output_layer_n = model.get_pooled_output_n()
    # sequence_output_a = model.get_sequence_output_a()
    # sequence_output_b = model.get_sequence_output_b()

    # mul_mask = lambda x, m: (x * tf.expand_dims(m, axis=-1))[:, 1:, :]
    # masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
    #     tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10 - 1
    # )
    # input_a_mask = tf.cast(input_a_mask, tf.float32)
    # input_b_mask = tf.cast(input_b_mask, tf.float32)
    # output_layer_a = masked_reduce_mean(sequence_output_a, input_a_mask)
    # output_layer_b = masked_reduce_mean(sequence_output_b, input_b_mask)

    with tf.variable_scope("loss"):
        # output_layer_a = tf.layers.dense(
        #             output_layer_a,
        #             bert_config.hidden_size,
        #             activation=tf.nn.relu,
        #             kernel_initializer=train_model.create_initializer(bert_config.initializer_range),
        #         )
        # output_layer_b = tf.layers.dense(
        #             output_layer_b,
        #             bert_config.hidden_size,
        #             activation=tf.nn.relu,
        #             kernel_initializer=train_model.create_initializer(bert_config.initializer_range),
        #         )
        if is_training:
            # I.e., 0.1 dropout
            output_layer_a = tf.nn.dropout(output_layer_a, keep_prob=0.9)
            output_layer_p = tf.nn.dropout(output_layer_p, keep_prob=0.9)
            output_layer_n = tf.nn.dropout(output_layer_n, keep_prob=0.9)

        loss = triplet_loss(output_layer_a, output_layer_p, output_layer_n, margin=margin)
        return loss


def create_test_model(
    bert_config,
    is_training,
    input_a_ids,
    input_b_ids,
    input_a_mask,
    input_b_mask,
    segment_a_ids,
    segment_b_ids,
    use_one_hot_embeddings
):
    model = test_model.BertModel(
        config=bert_config,
        is_training=is_training,
        input_a_ids=input_a_ids,
        input_b_ids=input_b_ids,
        input_a_mask=input_a_mask,
        input_b_mask=input_b_mask,
        token_a_type_ids=segment_a_ids,
        token_b_type_ids=segment_b_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )
    output_layer_a = model.get_pooled_output_a()
    output_layer_b = model.get_pooled_output_b()
    return cal_sim(output_layer_a, output_layer_b)


def cal_sim(tensor_a, tensor_b):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tensor_a * tensor_a, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tensor_b * tensor_b, 1))
    pooled_mul_12 = tf.reduce_sum(tensor_a * tensor_b, 1)
    sim = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
    sim = 1 - sim
    return sim


def triplet_loss(tensor_a, tensor_p, tensor_n, margin):
    pooled_len_a = tf.sqrt(tf.reduce_sum(tensor_a * tensor_a, 1))
    pooled_len_p = tf.sqrt(tf.reduce_sum(tensor_p * tensor_p, 1))
    pooled_len_n = tf.sqrt(tf.reduce_sum(tensor_n * tensor_n, 1))
    pooled_mul_ap = tf.reduce_sum(tensor_a * tensor_p, 1)
    pooled_mul_an = tf.reduce_sum(tensor_a * tensor_n, 1)
    distance_ap = 1 - tf.div(pooled_mul_ap, pooled_len_a * pooled_len_p + 1e-8)
    distance_an = 1 - tf.div(pooled_mul_an, pooled_len_a * pooled_len_n + 1e-8)
    loss = tf.reduce_mean(tf.maximum(distance_ap - distance_an + margin, 0.0))
    return loss


def model_fn_builder(
    bert_config,
    num_labels,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings,
    margin
):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if is_training:
            total_loss = create_train_model(
                bert_config,
                is_training,
                features["input_a_ids"],
                features["input_p_ids"],
                features["input_n_ids"],
                features["input_a_mask"],
                features["input_p_mask"],
                features["input_n_mask"],
                features["segment_a_ids"],
                features["segment_p_ids"],
                features["segment_n_ids"],
                use_one_hot_embeddings,
                margin
            )
            sim = 0
        else:
            total_loss = 0
            sim = create_test_model(
                bert_config,
                is_training,
                features["input_a_ids"],
                features["input_b_ids"],
                features["input_a_mask"],
                features["input_b_mask"],
                features["segment_a_ids"],
                features["segment_b_ids"],
                use_one_hot_embeddings
            )

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = train_model.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info(
                    "    name = %s, shape = %s%s", var.name, var.shape, init_string
                )
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu
            )

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, scaffold_fn=scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"sim": sim},
                loss=total_loss,
                # predictions={"sim": sim},
                scaffold_fn=scaffold_fn,
            )

        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                # predictions={"sim": sim, "loss": total_loss, "label_ids": label_ids},
                predictions={"sim": sim},
                scaffold_fn=scaffold_fn,
            )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    processors = {"mine": MyProcessor}

    tokenization.validate_case_matches_checkpoint(
        FLAGS.do_lower_case, FLAGS.init_checkpoint
    )

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True."
        )

    bert_config = train_model.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d"
            % (FLAGS.max_seq_length, bert_config.max_position_embeddings)
        )

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case
    )

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host,
        ),
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs
        )
        num_train_steps_per_epoch = int(len(train_examples) / FLAGS.train_batch_size)
        run_config = run_config.replace(save_checkpoints_steps=num_train_steps_per_epoch)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        margin=0.8
    )
    print(num_train_steps)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
    )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if FLAGS.raw_data:
            file_based_convert_examples_to_features_train(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("    Num examples = %d", len(train_examples))
        tf.logging.info("    Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("    Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if FLAGS.raw_data:
            file_based_convert_examples_to_features_test(
                    eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info(
            "    Num examples = %d (%d actual, %d padding)",
            len(eval_examples),
            num_actual_eval_examples,
            len(eval_examples) - num_actual_eval_examples,
        )
        tf.logging.info("    Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder
        )

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("    %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        if FLAGS.raw_data:
            file_based_convert_examples_to_features_test(
                predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file
            )
        tf.logging.info("***** Running prediction*****")
        tf.logging.info(
            "    Num examples = %d (%d actual, %d padding)",
            len(predict_examples),
            num_actual_predict_examples,
            len(predict_examples) - num_actual_predict_examples,
        )
        tf.logging.info("    Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder
        )
        result = estimator.predict(input_fn=predict_input_fn)
        # print("result", result)
        output_predict_file = os.path.join(FLAGS.output_dir, FLAGS.predict_output)
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            # print('result', result)
            for (i, prediction) in enumerate(result):
                probabilities = prediction["sim"]
                if i <= 10:
                    print("sim", probabilities)
                # print("label", prediction["label_ids"])
                # print("loss", prediction["loss"])
                if i >= num_actual_predict_examples:
                    break
                if (i % 10000 == 0) and (i > 0):
                    print("Predicting %d of %d" % (i, num_actual_predict_examples))
                # output_line = "\t".join(
                #        str(class_probability)
                #        for class_probability in probabilities) + "\n"
                output_line = str(probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
