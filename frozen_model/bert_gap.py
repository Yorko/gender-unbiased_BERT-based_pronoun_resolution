import pandas as pd
from tqdm import tqdm
import collections
import json
import random
import re
import fnmatch
import string
import bert
from bert import modeling, tokenization
from bert_utils import InputFeatures, model_fn_builder, input_fn_builder
import tensorflow as tf
tf.set_random_seed(17)


class GAPNormalizer(object):

    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.init_vocabulary()
        self.letters = set(string.ascii_letters + '*')

    def init_vocabulary(self):
        self.vocabulary = []
        with open(self.vocab_file) as f:
            for line in f:
                term = line.strip()
                if len(term) < 3 or '[' in term or '#' in term or any(char.isdigit() for char in term):
                    continue
                self.vocabulary.append(term)

    def normalize(self, text):
        for found_star in re.finditer('\*', text):
            start_index = found_star.start()
            while start_index >= 0 and text[start_index] in self.letters:
                start_index -= 1
            start_index += 1
            end_index = found_star.start()
            while end_index < len(text) and text[end_index] in self.letters:
                end_index += 1
            if end_index - start_index < 4:
                continue
            star_term = text[start_index:end_index]
            if star_term.count('*') >= len(star_term) - 1:
                continue
            replacements = fnmatch.filter(self.vocabulary, star_term.replace('*', '?'))
            if len(replacements) > 0:
                text = text[:start_index] + replacements[0] + text[end_index:]
        return text


class BertGAP(object):

    def __init__(self,
            bert_model='models/cased_L-24_H-1024_A-16',
            emb_size=1024,
            seq_len=64,
            n_layers=10,
            start_layer = 1,
            do_lower_case=False,
            normalize_text=False):
        self.bert_model = bert_model
        self.do_lower_case = do_lower_case
        self.emb_size = emb_size
        self.normalize_text = normalize_text
        self.vocab_file = '{}/vocab.txt'.format(self.bert_model)
        self.init_checkpoint = '{}/bert_model.ckpt'.format(self.bert_model)
        self.seq_len = seq_len
        self.layer_indexes = [-(l+start_layer) for l in range(n_layers)]
        self.predict_batch_size = 4

        self.bert_config = modeling.BertConfig.from_json_file('{}/bert_config.json'.format(self.bert_model))

        self.init_tokenizer()
        self.normalizer = GAPNormalizer(self.vocab_file)
        self.rng = random.Random(1122)

    def tokenize_single_row(self, row, middle_shift=0, random_rate=0.0, mask_rate=0.0, use_mask_padding=False):
        """Converts a single DataFrame row into a single `InputFeatures`."""

        def align_tokens(tokens, target_token_index):
            seq_len = self.seq_len - 2
            if len(tokens) > seq_len:
                start_index = max(0, int(target_token_index - seq_len / 2 + middle_shift))
                start_index = min(start_index, len(tokens) - seq_len)
                while tokens[start_index].startswith('#') and start_index + seq_len > target_token_index + 1:
                    start_index -= 1
                start_index = max(0, start_index)
                tokens = tokens[start_index : start_index + seq_len]
                target_token_index -= start_index
            tokens = ['[CLS]', ] + tokens + ['[SEP]', ]
            target_token_index += 1
            return tokens, target_token_index

        def preprocess_text(text):
            text = text.replace('*', '')
            return text

        break_points = [
                (row["Pronoun-offset"], row["Pronoun"], False),
                (row["A-offset"], row["A"], True),
                (row["B-offset"], row["B"], True),
            ]

        segment_tokens = []
        if self.normalize_text:
            row_text = self.normalizer.normalize(row["Text"].lower()) \
                if self.do_lower_case else self.normalizer.normalize(row["Text"])
        else:
            row_text = row["Text"]

        for offset, text, use_mask in break_points:
            # Tokens before the target
            tokens = []
            tmp_tokens = self.tokenizer.tokenize(preprocess_text(row_text[:offset]))
            if len(tmp_tokens) > 0:
                tokens.extend(tmp_tokens)

            # Tokenize the target
            target_token_index = len(tokens)
            if use_mask:
                tokens.append('[MASK]')
            else:
                tmp_tokens = self.tokenizer.tokenize(row_text[offset:offset+len(text)])
                tokens.extend(tmp_tokens)

            # Tokens after the target
            tmp_tokens = self.tokenizer.tokenize(preprocess_text(row_text[offset+len(text):]))
            if len(tmp_tokens) > 0:
                tokens.extend(tmp_tokens)

            if use_mask_padding:
                padding_len = int(self.seq_len / 2)
                tokens = ['[MASK]'] * padding_len + tokens + ['[MASK]'] * padding_len
                target_token_index += padding_len

            tokens, target_token_index = align_tokens(tokens, target_token_index)

            if random_rate > 0.0:
                for token_index in range(1, len(tokens) - 1):
                    if token_index == target_token_index or tokens[token_index].startswith('#') \
                            or self.rng.random() < random_rate:
                        continue
                    random_token = self.normalizer.vocabulary[self.rng.randint(0, len(self.normalizer.vocabulary) - 1)]
                    tokens[token_index] = random_token

            if mask_rate > 0.0:
                for token_index in range(1, len(tokens) - 1):
                    if token_index == target_token_index or tokens[token_index].startswith('#') or \
                            self.rng.random() < mask_rate:
                        continue
                    tokens[token_index] = '[MASK]'

            segment_tokens.append({'target_token_index': target_token_index, 'tokens': tokens})

        return segment_tokens

    def tokens_to_features(self, unique_id, tokens):
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        input_type_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.seq_len:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        return InputFeatures(
            unique_id=unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids)

    def tokenize(self, aug_id=0):
        self.feats = {'features': [], 'segments': [], 'ids': [], 'df_ids': [], 'target_token_ids': []}
        unique_id = 0
        for _, row in tqdm(self.df.iterrows()):
            segment_tokens = self.tokenize_single_row(row)
            for j, segment in enumerate(segment_tokens):
                if segment['target_token_index'] > 0:
                    features = self.tokens_to_features(unique_id, segment['tokens'])
                    unique_id += 1
                    self.feats['features'].append(features)
                    self.feats['segments'].append(j)
                    self.feats['target_token_ids'].append(segment['target_token_index'] )
                    self.feats['df_ids'].append(row.ID)
                    self.feats['ids'].append('{}_{}'.format(row.ID, aug_id))

    def process_embeddings(self, input_file, sep, output_file, aug_id=0):
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            master=None,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=8,
                per_host_input_for_training=is_per_host))

        self.load_dataset(input_file, sep)
        self.tokenize(aug_id=aug_id)

        tf.logging.info('Process {} embeddings, there are {} samples'.format(input_file, len(self.feats['features'])))
        features = self.feats['features']

        model_fn = model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=self.init_checkpoint,
            layer_indexes=self.layer_indexes,
            use_tpu=False,
            use_one_hot_embeddings=False)

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=self.predict_batch_size)

        input_fn = input_fn_builder(
            features=features, seq_length=self.seq_len)

        with open(output_file, 'w') as wf:
            for result in tqdm(estimator.predict(input_fn, yield_single_examples=True), total=len(features)):
                unique_id = int(result['unique_id'])
                idx = self.feats['ids'][unique_id]
                df_idx = self.feats['df_ids'][unique_id]
                segment = self.feats['segments'][unique_id]
                target_token_index = self.feats['target_token_ids'][unique_id]
                output_json = collections.OrderedDict()
                output_json['linex_index'] = unique_id
                output_json['idx'] = idx
                output_json['df_idx'] = df_idx
                output_json['segment'] = segment
                layers = []
                for (j, layer_index) in enumerate(self.layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    layer = collections.OrderedDict()
                    layer['index'] = layer_index
                    layer['values'] = [
                        round(float(x), 6) for x in layer_output[target_token_index:(target_token_index + 1)].flat
                    ]
                    layers.append(layer)
                output_json['embeddings'] = layers
                wf.write(json.dumps(output_json) + '\n')

    def init_tokenizer(self):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

    def load_dataset(self, input_file, sep):
        print('Loading {}'.format(input_file))
        self.df = pd.read_csv(input_file, sep=sep)
        self.df['Pronoun-len'] = self.df.Pronoun.str.len
        self.df['A-len'] = self.df.A.str.len
        self.df['B-len'] = self.df.B.str.len



