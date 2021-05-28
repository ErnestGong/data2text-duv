# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import io
import codecs
import sys
import re

import torch
import torchtext

import pickle

from onmt.Utils import aeq
from onmt.io.DatasetBase import (ONMTDatasetBase, UNK_WORD,
                                 PAD_WORD, BOS_WORD, EOS_WORD)

PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3
PRETRAIN_PAD_INDEX = 5

class PretrainTextDataset(ONMTDatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """
    def __init__(self, fields, src_examples_iter, tgt_examples_iter, src_data_examples_iter,
                 num_src_feats=0, num_tgt_feats=0, num_src_data_feats=0,
                 src_seq_length=0, tgt_seq_length=0, src_data_seq_length=0,
                 use_filter_pred=True, ent_idx_file=None):
        self.data_type = 'pretrain'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats
        self.n_src_data_feats = num_src_data_feats
        assert self.n_src_feats == self.n_src_data_feats or self.n_src_data_feats == 0

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.

        # pretrain_src = None
        # if pretrain_src_file is not None:
        #     with open(pretrain_src_file) as f:
        #         content = f.readlines()
        #     pretrain_src = [x.strip() for x in content if len(x.strip()) > 0]

        # pretrain_tgt = None
        # if pretrain_tgt_file is not None:
        #     with open(pretrain_tgt_file) as f:
        #         content = f.readlines()
        #     pretrain_tgt = [x.strip() for x in content if len(x.strip()) > 0]

        if tgt_examples_iter is None:
            examples_iter = (self._join_dicts(src, src_data) for src, src_data in
                 zip(src_examples_iter, src_data_examples_iter))
        elif src_data_examples_iter is None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                 zip(src_examples_iter, tgt_examples_iter))
        elif tgt_examples_iter is None and src_data_examples_iter is None:
            examples_iter = src_examples_iter
        else:
            examples_iter = (self._join_dicts(src, tgt, src_data) for src, tgt, src_data in
                 zip(src_examples_iter, tgt_examples_iter, src_data_examples_iter))

        ent_idx = None
        if ent_idx_file is not None:
            with open(ent_idx_file, "rb") as f:
                ent_idx = pickle.load(f)
            examples_iter = self._get_pretrain_ent_datas(examples_iter, ent_idx)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]

        # assert False, (keys)
        # assert False, (fields)

        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        tgt_sets = set()

        for ex_values in example_values:
            # object of "torchtext.data.Example"
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            src_size += len(example.src_pretrain)
            for each_one in example.tgt_pretrain:
                tgt_sets.add(each_one)
            out_examples.append(example)

        self.tgt_opt_num = len(tgt_sets)
        print("average src size", src_size / len(out_examples),
              len(out_examples))

        def filter_pred(example):

            return True
            # return (not "src1" in example.__dict__) or (0 < len(example.src1) <= src_seq_length \
            #    and 0 < len(example.tgt1) <= tgt_seq_length \
            #        and (pointers_file is None or 1 < example.ptrs.size(0)))

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(PretrainTextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def _get_pretrain_ent_datas(self, examples_iter, ent_idx):
        loop_index = -1
        for example in examples_iter:
            loop_index += 1

            if ent_idx is not None:
                curr_idx = ent_idx[loop_index]
                example['src_pretrain_data_idx'] = curr_idx
            else:
                example['src_pretrain_data_idx'] = None

            yield example  

    @staticmethod
    def make_text_examples_nfeats_tpl(path, truncate, side):
        """
        Args:
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ["src_pretrain", "tgt_pretrain"]

        if path is None:
            return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            PretrainTextDataset.read_text_file(path, truncate, side)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    @staticmethod
    def read_text_file(path, truncate, side):
        """
        Args:
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for i, line in enumerate(corpus_file):
                line = line.strip().split()
                if truncate:
                    assert False, "doesn't support yet"
                    line = line[:truncate]

                words, feats, n_feats = \
                    PretrainTextDataset.extract_text_features(line)

                example_dict = {side: words, "indices": i}
                if side == "tgt_pretrain":
                    example_dict = {side: [int(word) for word in words], "indices": i}
                if feats:
                    prefix = side + "_feat_"
                    example_dict.update((prefix + str(j), f)
                                        for j, f in enumerate(feats))
                yield example_dict, n_feats

    @staticmethod
    def get_fields(n_src_features, n_tgt_features, n_src_pretrain_data_features):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        # fields for pretrain
        fields["src_pretrain"] = torchtext.data.Field(
            pad_token = PAD_WORD,
            include_lengths = False)

        for j in range(n_src_features):
            fields["src_pretrain_feat_" + str(j)] = \
                torchtext.data.Field(
                    pad_token = PAD_WORD,
                    include_lengths = False)

        fields["tgt_pretrain"] = torchtext.data.Field(
            use_vocab=False,
            pad_token=PRETRAIN_PAD_INDEX,
            include_lengths = False)

        for j in range(n_tgt_features):
            fields["tgt_pretrain_feat_" + str(j)] = \
                torchtext.data.Field(
                    use_vocab=False,
                    pad_token=PRETRAIN_PAD_INDEX,
                    include_lengths = False)


        if n_src_pretrain_data_features is not None:
            fields['src_pretrain_data'] = torchtext.data.Field(
                pad_token = PAD_WORD,
                include_lengths = False)
            for j in range(n_src_pretrain_data_features):
                fields['src_pretrain_data_feat_' + str(j)] = torchtext.data.Field(
                    pad_token = PAD_WORD,
                    include_lengths = False)
            fields['src_pretrain_data_idx'] = torchtext.data.RawField(
                )
        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            f_line = cf.readline().strip().split()
            _, _, num_feats = PretrainTextDataset.extract_text_features(f_line)

        return num_feats



class ShardedPretrainCorpusIterator(object):
    """
    This is the iterator for text corpus, used for sharding large text
    corpus into small shards, to avoid hogging memory.

    Inside this iterator, it automatically divides the corpus file into
    shards of size `shard_size`. Then, for each shard, it processes
    into (example_dict, n_features) tuples when iterates.
    """
    def __init__(self, corpus_path, line_truncate, side, shard_size,
                 assoc_iter=None):
        """
        Args:
            corpus_path: the corpus file path.
            line_truncate: the maximum length of a line to read.
                            0 for unlimited.
            side: "src" or "tgt".
            shard_size: the shard size, 0 means not sharding the file.
            assoc_iter: if not None, it is the associate iterator that
                        this iterator should align its step with.
        """
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)

        self.line_truncate = line_truncate
        self.side = side
        self.shard_size = shard_size
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.line_index = -1
        self.eof = False

    def __iter__(self):
        """
        Iterator of (example_dict, nfeats).
        On each call, it iterates over as many (example_dict, nfeats) tuples
        until this shard's size equals to or approximates `self.shard_size`.
        """
        iteration_index = -1
        if self.assoc_iter is not None:
            # We have associate iterator, just yields tuples
            # util we run parallel with it.
            while self.line_index < self.assoc_iter.line_index:
                line = self.corpus.readline()
                if line == '':
                    raise AssertionError(
                        "Two corpuses must have same number of lines!")

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
        else:
            # Yield tuples util this shard's size reaches the threshold.
            self.corpus.seek(self.last_pos)
            while True:
                if self.shard_size != 0 and self.line_index % 64 == 0:
                    # This part of check is time consuming on Py2 (but
                    # it is quite fast on Py3, weird!). So we don't bother
                    # to check for very line. Instead we chekc every 64
                    # lines. Thus we are not dividing exactly per
                    # `shard_size`, but it is not too much difference.
                    cur_pos = self.corpus.tell()
                    if cur_pos >= self.last_pos + self.shard_size:
                        self.last_pos = cur_pos
                        raise StopIteration

                line = self.corpus.readline()
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    raise StopIteration

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

    def hit_end(self):
        return self.eof

    @property
    def num_feats(self):
        # We peek the first line and seek back to
        # the beginning of the file.
        saved_pos = self.corpus.tell()

        line = self.corpus.readline().split()
        if self.line_truncate:
            assert False, "doesn't use this"
            line = line[:self.line_truncate]
        _, _, self.n_feats = PretrainTextDataset.extract_text_features(line)

        self.corpus.seek(saved_pos)

        return self.n_feats



    def _example_dict_iter(self, line, index):
        line = line.split()
        if self.line_truncate:
            assert False, "doesn't use this"
            line = line[:self.line_truncate]
        words, feats, n_feats = PretrainTextDataset.extract_text_features(line)
        example_dict = {self.side: words, "indices": index}
        if self.side == "tgt_pretrain":
            example_dict = {self.side: [int(word) for word in words], "indices": index}
        if feats:
            # All examples must have same number of features.
            aeq(self.n_feats, n_feats)

            prefix = self.side + "_feat_"
            example_dict.update((prefix + str(j), f)
                                for j, f in enumerate(feats))

        return example_dict
