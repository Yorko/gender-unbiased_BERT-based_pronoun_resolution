# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a modification of run_classifier.py for BERT from Google AI
# The modification was done by Ken Krige April 2019 to parse GAP data for
# kaggle competition https://www.kaggle.com/c/gendered-pronoun-resolution

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import csv
import re
import os
import BERT.modeling as modeling
import BERT.optimization as optimization
import BERT.tokenization as tokenization
import tensorflow as tf
tf.reset_default_graph()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", 'data',
    "The input data dir. Should contain the .tsv files (or other data files) ")

flags.DEFINE_string("bert_config_file", 'BERT/uncased_L-12_H-768_A-12/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model.")

flags.DEFINE_string("task_name", "GAP", "The name of the task to train.")

flags.DEFINE_string("vocab_file", 'BERT/uncased_L-12_H-768_A-12/vocab.txt',
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("output_dir", "out/6400",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("init_checkpoint", 'BERT/uncased_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True, "True for uncased BERT models")

flags.DEFINE_integer("max_seq_length", 64, "Sequence length after WordPiece tokenization.")

flags.DEFINE_bool("convert_data", False, "Convert to feminine to match pseudo label training.")

flags.DEFINE_bool("pre_train", False, "Run pre-training to create TFRecords.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Predict mode on the test set.")

flags.DEFINE_integer("train_batch_size", 256, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("smoothing", 0.01, "Probability smoothing factor.")

flags.DEFINE_float("num_train_epochs", 3, "Total number of training epochs.")

flags.DEFINE_integer("epoch_size", 2454, "How often to checkpoint.")

flags.DEFINE_float("warmup_proportion", 0.1, "Learning rate linear warmup proportion.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000, "Batch steps per estimator call.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string("tpu_name", "node-2", "The Cloud TPU to use for training.")

tf.flags.DEFINE_string("tpu_zone", "europe-west4-a", "[Optional] GCE zone")

tf.flags.DEFINE_string("gcp_project", None,"[Optional] Project name")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8, "Number of TPU cores to use.")


class InputExample(object):
 
  def __init__(self, guid, text_a, char_offsets, label=None):
    self.guid = guid
    self.text_a = text_a
    self.char_offsets = char_offsets
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputFeatures(object):

  def __init__(self, input_ids, input_mask,
               P_mask, A_mask, B_mask,
               segment_ids, label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.P_mask = P_mask
    self.A_mask = A_mask
    self.B_mask = B_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):

  def get_train_examples(self, data_dir):
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    raise NotImplementedError()

  def get_labels(self):
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar) #\t
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class GAProcessor(DataProcessor):
  """Processor for the GAP data set."""

  def get_train_examples(self, path):
    tf.logging.info(path)
    return self._create_examples(
        self._read_tsv(path), "train")

  def get_dev_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    return ["A", "B", "Neither"]

  def _create_examples(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = tokenization.convert_to_unicode(line[0]) 
      text_a = tokenization.convert_to_unicode(line[1])
      a_coref = tokenization.convert_to_unicode(line[6][0])
      b_coref = tokenization.convert_to_unicode(line[9][0])
      P_offset = int(tokenization.convert_to_unicode(line[3]))
      A_offset = int(tokenization.convert_to_unicode(line[5]))
      B_offset = int(tokenization.convert_to_unicode(line[8]))
      char_offsets = [P_offset, A_offset, B_offset]
      #Check alignment of offsets
      assert text_a[P_offset:P_offset + 2] == tokenization.convert_to_unicode(line[2])[:2] 
      label = tokenization.convert_to_unicode("Neither")
      if a_coref == tokenization.convert_to_unicode("T"):
        label = tokenization.convert_to_unicode("A")
      elif b_coref == tokenization.convert_to_unicode("T"):
        label = tokenization.convert_to_unicode("B")
      examples.append(
          InputExample(guid=guid, text_a=text_a, char_offsets=char_offsets, label=label))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):

  P_mask = [0] * max_seq_length
  A_mask = [0] * max_seq_length
  B_mask = [0] * max_seq_length

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        P_mask=P_mask,
        A_mask=A_mask,
        B_mask=B_mask,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  P_offset = example.char_offsets[0]
  A_offset = example.char_offsets[1]
  B_offset = example.char_offsets[2]
  char_off = sorted([
      [P_offset, 0],
      [A_offset, 1],
      [B_offset, 2]
  ], key=lambda x: x[0])
  text_segments = [example.text_a[:char_off[0][0]], 
    example.text_a[char_off[0][0]:char_off[1][0]], 
    example.text_a[char_off[1][0]:char_off[2][0]], 
    example.text_a[char_off[2][0]:]]

  if FLAGS.convert_data:
    for i, segment in enumerate(text_segments):
      seg = re.sub(r"(?<![a-zA-Z])he(?![a-zA-Z])", "she", segment.lower())
      seg = re.sub(r"(?<![a-zA-Z])his(?![a-zA-Z])", "her", seg)
      seg = re.sub(r"(?<![a-zA-Z])hers(?![a-zA-Z])", "her", seg)
      seg = re.sub(r"(?<![a-zA-Z])him(?![a-zA-Z])", "her", seg)
      seg = re.sub(r"(?<![a-zA-Z])himself(?![a-zA-Z])", "herself", seg)
      text_segments[i] = seg
  
  token_segments = []
  tokens_in_segment = []
  for segment in text_segments:
    token_segment = tokenizer.tokenize(segment)
    token_segments.append(token_segment)
    tokens_in_segment.append(len(token_segment))

  while np.sum(tokens_in_segment) > (max_seq_length - 2):
    index = np.argmax([tokens_in_segment[0] * 2, 
                       tokens_in_segment[1], 
                       tokens_in_segment[2], 
                       tokens_in_segment[3] * 2])
    if index == 0:
      token_segments[index] = token_segments[index][1:]
    elif index == 3:
      token_segments[index] = token_segments[index][:-1]
    else: #middle segments
      middle = tokens_in_segment[index] // 2
      token_segments[index] = token_segments[index][:middle] + token_segments[index][middle + 1:]
    tokens_in_segment[index] -= 1

  tokens = []
  segment_ids = [0] * max_seq_length
  tokens.append("[CLS]")
  for segment in token_segments:
    temp = ''
    for token in segment:
      tokens.append(token)
  tokens.append("[SEP]")

  offset = 1 #to account for "[CLS]"
  for i, row in enumerate(char_off):
    offset += tokens_in_segment[i]
    row[0] = offset

  token_off = sorted(char_off, key=lambda x: x[1])
  P_mask[token_off[0][0]] = 1
  A_mask[token_off[1][0]] = 1
  B_mask[token_off[2][0]] = 1

  assert len(tokens) < max_seq_length + 1

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 3:
    tf.logging.info("tokens: %s" % " ".join(
        [(str(P_mask[i]) + str(A_mask[i]) + str(B_mask[i]) + 
          tokenization.printable_text(tokens[i])) for i in range(len(tokens))]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      P_mask=P_mask,
      A_mask=A_mask,
      B_mask=B_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
  writer = tf.python_io.TFRecordWriter(output_file)
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
    feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["P_mask"] = create_int_feature(feature.P_mask)
    features["A_mask"] = create_int_feature(feature.A_mask)
    features["B_mask"] = create_int_feature(feature.B_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "P_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "A_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "B_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
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
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = None
    if is_training:
      gap_file = os.path.join(input_file, "*.tsv.tf_record")
      gap = tf.data.Dataset.list_files(gap_file)\
        .flat_map(tf.data.TFRecordDataset)\
        .take(64)
      if len(tf.gfile.ListDirectory(input_file)) > 3:
        pseudo_file = os.path.join(input_file, "*-tsv.tf_record")
        pseudo = tf.data.Dataset.list_files(pseudo_file)\
          .flat_map(tf.data.TFRecordDataset)\
          .shuffle(buffer_size=1000)\
          .take(128)
        d = gap.concatenate(pseudo).shuffle(buffer_size=128)
      else:
        d = tf.data.Dataset.list_files(gap_file)\
          .flat_map(tf.data.TFRecordDataset)
      d = d.repeat()
    else:
      d = tf.data.TFRecordDataset(input_file)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(bert_config, is_training, input_ids,
                input_mask, P_mask, A_mask, B_mask,
                segment_ids, labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
 
  all_out = model.get_sequence_output()
  hidden_size = all_out.shape[-1].value

  '''The next 3 lines of code work on GPU for boolean masking
  However, boolean_mask() is not implemented for TPU because it
  results in dynamic tensor sizes.
  P = tf.boolean_mask(all_out, P_mask)
  A = tf.boolean_mask(all_out, A_mask)
  B = tf.boolean_mask(all_out, B_mask)'''

  #The next 15 lines of code are a TPU workaround
  #Messy but works. There may be a better way.
  _P_mask = tf.cast(P_mask, tf.float32)
  _A_mask = tf.cast(A_mask, tf.float32)
  _B_mask = tf.cast(B_mask, tf.float32)

  _P_mask_ = tf.broadcast_to(_P_mask, shape=(tf.shape(all_out)[2],tf.shape(all_out)[0],tf.shape(all_out)[1]))
  P_mask_ = tf.transpose(_P_mask_, perm=[1,2,0])
  _A_mask_ = tf.broadcast_to(_A_mask, shape=(tf.shape(all_out)[2],tf.shape(all_out)[0],tf.shape(all_out)[1]))
  A_mask_ = tf.transpose(_A_mask_, perm=[1,2,0])
  _B_mask_ = tf.broadcast_to(_B_mask, shape=(tf.shape(all_out)[2],tf.shape(all_out)[0],tf.shape(all_out)[1]))
  B_mask_ = tf.transpose(_B_mask_, perm=[1,2,0])

  P_ = tf.multiply(all_out, P_mask_)
  P = tf.reduce_sum(P_, axis=1)
  A_ = tf.multiply(all_out, A_mask_)
  A = tf.reduce_sum(A_, axis=1)
  B_ = tf.multiply(all_out, B_mask_)
  B = tf.reduce_sum(B_, axis=1)
  #End of TPU workaround

  PA = tf.multiply(P, A)
  PB = tf.multiply(P, B)
  PP = tf.multiply(P, P)
  AB = tf.multiply(A, B)
  N = tf.subtract(PP, AB)

  AB_weights = tf.get_variable(
      "AB_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  N_weights = tf.get_variable(
      "N_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  if is_training:
     # I'm not sure if dropout on weights, rather than data
     # is accepted practice, but it seemed to work
     AB_weights = tf.nn.dropout(AB_weights, keep_prob=0.9)
     N_weights = tf.nn.dropout(N_weights, keep_prob=0.9)

  A_out = tf.matmul(PA, AB_weights, transpose_b=True)
  B_out = tf.matmul(PB, AB_weights, transpose_b=True)
  N_out = tf.matmul(N, N_weights, transpose_b=True)

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    logits = tf.concat([A_out, B_out, N_out], axis=1)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):

  def model_fn(features, labels, mode, params):
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    P_mask = features["P_mask"]
    A_mask = features["A_mask"]
    B_mask = features["B_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, P_mask, A_mask, B_mask,
        segment_ids, label_ids, num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=100)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }
      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)

    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  record_dir = os.path.join(FLAGS.data_dir, "trainrecords" + str(FLAGS.max_seq_length))
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "gap": GAProcessor,
  }

  tf.estimator.RunConfig(model_dir=FLAGS.output_dir)
  
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.pre_train:
    raise ValueError("One of `pre_train`, `do_train`, `do_eval` or `do_predict' must be True.")

  if FLAGS.do_train and  FLAGS.pre_train:
    raise ValueError("Cannot `pre_train` and `do_train` in a single pass. First do `pre_train`")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    num_train_steps = int(
        FLAGS.epoch_size / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.pre_train:
    tsv_dir = os.path.join(FLAGS.data_dir, "trainQ")
    tf.gfile.MakeDirs(record_dir)
    files = tf.gfile.ListDirectory(tsv_dir)
    for in_file in files:
      in_path = os.path.join(tsv_dir, in_file)
      train_examples = processor.get_train_examples(in_path)
      train_file = in_file + '.tf_record'
      train_path = os.path.join(record_dir, train_file)
      file_based_convert_examples_to_features(
          train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_path)

  if FLAGS.do_train:
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
         input_file=record_dir,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    if not tf.gfile.Exists(eval_file):
      file_based_convert_examples_to_features(
          eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    if not tf.gfile.Exists(predict_file):
      file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    # My own algorithm for keeping probs away from extremes 0 and 1
    # Standard clipping might be better, but this is what I used for kaggle comp.
    def smooth(prob):
      return (1.0 - FLAGS.smoothing) * prob  + FLAGS.smoothing / 3.0

    guids = []
    # This is a hack. If order gets shuffled ids will not match predictions.
    # So predict must NOT be parrarelised!
    # Tried to do this with feature_forwarding, but so far a fail.
    for example in predict_examples:
      guids.append(example.guid)

    output_predict_file = os.path.join(FLAGS.output_dir, "kenkrige_results.csv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        writer.write("ID,A,B,NEITHER\n")
        for i, prediction in enumerate(result):
            guid = guids[i]
            out = prediction['probabilities'].tolist()
            output_line = guid + ',' + ",".join(str(smooth(result)) for result in out) + "\n"
            writer.write(output_line)

if __name__ == "__main__":
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
