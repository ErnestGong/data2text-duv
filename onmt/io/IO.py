# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, OrderedDict
from itertools import count

import torch
import torchtext.data
import torchtext.vocab

from onmt.io.DatasetBase import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.io.TextDataset import TextDataset
from onmt.io.PretrainTextDataset import PretrainTextDataset
from onmt.io.ImageDataset import ImageDataset
from onmt.io.AudioDataset import AudioDataset


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def get_fields(data_type, n_src_features, n_tgt_features, n_src_hist_nfeats1, pretrain_pad_index, mlb, nopretrain=False, n_src_pretrain_data_features=None, n_src_ent_features=None, n_src_ent_data_features=None, marg=False, s2_multi_attn_multi_copy=False):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """

    if data_type == "pretrain":
        return PretrainTextDataset.get_fields(n_src_features, n_tgt_features, n_src_pretrain_data_features)
    elif data_type == 'text':
        return TextDataset.get_fields(n_src_features, n_tgt_features, n_src_hist_nfeats1, pretrain_pad_index, mlb, nopretrain, n_src_ent_features, n_src_ent_data_features, marg, s2_multi_attn_multi_copy)
    elif data_type == 'img':
        return ImageDataset.get_fields(n_src_features, n_tgt_features, n_src_hist_nfeats1)
    elif data_type == 'audio':
        return AudioDataset.get_fields(n_src_features, n_tgt_features, n_src_hist_nfeats1)


def load_fields_from_vocab(vocab, data_type, opt, nopretrain=False, ent_use_data=False):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    if data_type == "pretrain":
        n_src_features = len(collect_features(vocab, 'src_pretrain'))
        n_src_hist_features = None
        n_tgt_features = len(collect_features(vocab, 'tgt_pretrain'))
        pretrain_pad_index = None
        if ent_use_data or (hasattr(opt, "pre_ent_use_data") and opt.pre_ent_use_data):
        # if opt.pretrain_emb_ent and opt.pre_ent_use_data:
            src_pretrain_data_nfeats = len(collect_features(vocab, 'src_pretrain_data'))
        else:
            src_pretrain_data_nfeats = None

        src_ent_features = None
        src_ent_data_features = None
    else:
        n_src_features = len(collect_features(vocab, 'src1'))
        if 'src1_hist' in vocab:
            n_src_hist_features = len(collect_features(vocab, 'src1_hist'))
        else:
            n_src_hist_features = None
        n_tgt_features = len(collect_features(vocab, 'tgt1'))
        pretrain_pad_index = vocab["src1_pretrain"]["pretrain_pad_index"]
        src_pretrain_data_nfeats = None
        if 'src1_ent_pretrain' in vocab:
            src_ent_features = len(collect_features(vocab, 'src1_ent_pretrain'))
        else:
            src_ent_features = None

        if ent_use_data or (hasattr(opt, "pre_ent_use_data") and opt.pre_ent_use_data):
            src_ent_data_features = len(collect_features(vocab, 'src1_ent_pretrain_data'))
        else:
            src_ent_data_features = None

    fields = get_fields(data_type, n_src_features, n_tgt_features, n_src_hist_features, pretrain_pad_index, True if hasattr(opt, "mlb") and opt.mlb else False, nopretrain, src_pretrain_data_nfeats, src_ent_features, src_ent_data_features, True if hasattr(opt, "marginal") and opt.marginal else False, True if hasattr(opt, "s2_multi_attn_multi_copy") and opt.s2_multi_attn_multi_copy else False)

    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        if k in ("src1_pretrain", "src1_ent_pretrain"):
            assert isinstance(v, dict) and (len(v) == 3 or len(v) == 2)
            v["pretrain"].stoi = defaultdict(lambda: 0, v["pretrain"].stoi)
            v["not_pretrain"].stoi = defaultdict(lambda: 0, v["not_pretrain"].stoi)
        else:
            v.stoi = defaultdict(lambda: 0, v.stoi)
        if k in fields:
            fields[k].vocab = v

    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            if k in ("src1_pretrain", "src1_ent_pretrain"):
                assert isinstance(f.vocab, dict) and (len(f.vocab) == 3 or len(f.vocab) == 2)
                f.vocab["pretrain"].stoi = dict(f.vocab["pretrain"].stoi)
                f.vocab["not_pretrain"].stoi = dict(f.vocab["not_pretrain"].stoi)
            else:
                f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab


def merge_vocabs(vocabs, field, **kwargs):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))

    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return field.vocab_cls(merged,
                           specials=specials,
                           **kwargs)


def get_num_features(data_type, corpus_file, side):
    """
    Args:
        data_type (str): type of the source input.
            Options are [text|img|audio].
        corpus_file (str): file path to get the features.
        side (str): for source or for target.

    Returns:
        number of features on `side`.
    """
    assert side in ['src1', 'src2', 'tgt1', 'tgt2', 'src1_hist', 'src_pretrain', 'tgt_pretrain', 'src_pretrain_data', 'src1_ent_pretrain']

    if corpus_file is None:
        return None
    elif data_type == "pretrain":
        return PretrainTextDataset.get_num_features(corpus_file, side)
    elif data_type == 'text':
        return TextDataset.get_num_features(corpus_file, side)
    elif data_type == 'img':
        return ImageDataset.get_num_features(corpus_file, side)
    elif data_type == 'audio':
        return AudioDataset.get_num_features(corpus_file, side)

def make_pretrain_ent_features(batch, side, data_type='text'):
    """
    Args:
        batch (Variable): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src1_ent_pretrain', 'src1_ent_pretrain_data']

    ent_pre, select_idx = batch.__dict__[side]
    assert len(ent_pre.size()) == 3
    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k][0] for k in keys]
    levels = [ent_pre] + features

    cat_res = torch.cat([level.unsqueeze(3) for level in levels], 3)
    if side == 'src1_ent_pretrain':
        assert select_idx is not None
        cat_res = (cat_res, select_idx)
    return cat_res

def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Variable): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src1', 'src2', 'tgt1', 'tgt2', 'src1_char', 'src2_char', 'src1_hist', 'src_pretrain', "tgt_pretrain", "src_pretrain_data"]
    char_flag = (side.endswith("_char"))
    if isinstance(batch.__dict__[side], tuple) and not char_flag:
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    if not char_flag:
        feat_start = side + "_feat_"
        keys = sorted([k for k in batch.__dict__ if feat_start in k])
        features = [batch.__dict__[k] for k in keys]
        levels = [data] + features
    else:
        levels = [data]

    if (data_type == 'text' or data_type == 'pretrain') and not char_flag:
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


def collect_features(fields, side="src1"):
    """
    Collect features from Field object.
    """
    assert side in ['src1', 'src2', 'tgt1', 'tgt2', 'src1_hist', 'src_pretrain', 'tgt_pretrain', 'src_pretrain_data', 'src1_ent_pretrain', 'src1_ent_pretrain_data']
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def collect_feature_vocabs(fields, side):
    """
    Collect feature Vocab objects from Field object.
    """
    assert side in ['src1', 'src2', 'tgt1', 'tgt2', 'src1_hist', "src_pretrain", "tgt_pretrain", "src_pretrain_data"]
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs


def build_dataset(fields, data_type, src_path, src_hist_path, src_pretrain_path, tgt_path, src_path2, tgt_path2, src_dir=None,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True, pretrain_usage_ent_path=None, model_opt=None):

    # Build src/tgt examples iterator from corpus files, also extract
    # number of features.

    if data_type == "pretrain":
        src_examples_iter, num_src_feats = PretrainTextDataset.make_text_examples_nfeats_tpl(
                src_path, src_seq_length_trunc, "src_pretrain")
        tgt_examples_iter, num_tgt_feats = PretrainTextDataset.make_text_examples_nfeats_tpl(
                tgt_path, tgt_seq_length_trunc, "tgt_pretrain")
    else:
        src_examples_iter, num_src_feats = \
            _make_examples_nfeats_tpl(data_type, src_path, src_dir,
                                      src_seq_length_trunc, sample_rate,
                                      window_size, window_stride,
                                      window, normalize_audio, "src1")

        src_hist_examples_iter, num_src_hist_feats = \
            _make_examples_nfeats_tpl(data_type, src_hist_path, src_dir,
                                      src_seq_length_trunc, sample_rate,
                                      window_size, window_stride,
                                      window, normalize_audio, "src1_hist")

        tgt_examples_iter, num_tgt_feats = \
            TextDataset.make_text_examples_nfeats_tpl(
                tgt_path, tgt_seq_length_trunc, "tgt1")

        src_examples_iter2, num_src_feats2 = \
            _make_examples_nfeats_tpl(data_type, src_path2, src_dir,
                                      src_seq_length_trunc, sample_rate,
                                      window_size, window_stride,
                                      window, normalize_audio, "src2")

        tgt_examples_iter2, num_tgt_feats2 = \
            TextDataset.make_text_examples_nfeats_tpl(
                tgt_path2, tgt_seq_length_trunc, "tgt2")

    if data_type == "pretrain":
        dataset = PretrainTextDataset(
                fields, src_examples_iter, tgt_examples_iter, None,
                num_src_feats, num_tgt_feats,
                src_seq_length=src_seq_length,
                tgt_seq_length=tgt_seq_length
                )
    elif data_type == 'text':
        dataset = TextDataset(fields, src_examples_iter, src_hist_examples_iter, tgt_examples_iter, src_examples_iter2, tgt_examples_iter2,
                              num_src_feats, num_src_hist_feats, num_tgt_feats,
                              src_seq_length=src_seq_length,
                              tgt_seq_length=tgt_seq_length,
                              dynamic_dict=dynamic_dict,
                              use_filter_pred=use_filter_pred,
                              pretrain_usage_file=src_pretrain_path,
                              pretrain_usage_ent_file=pretrain_usage_ent_path, s2_multi_attn_multi_copy=model_opt.s2_multi_attn_multi_copy if hasattr(model_opt, "s2_multi_attn_multi_copy") else False)
    elif data_type == 'img':
        dataset = ImageDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, num_tgt_feats,
                               tgt_seq_length=tgt_seq_length,
                               use_filter_pred=use_filter_pred)

    elif data_type == 'audio':
        dataset = AudioDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, num_tgt_feats,
                               tgt_seq_length=tgt_seq_length,
                               sample_rate=sample_rate,
                               window_size=window_size,
                               window_stride=window_stride,
                               window=window,
                               normalize_audio=normalize_audio,
                               use_filter_pred=use_filter_pred)

    return dataset


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)


def build_pretrain_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """

    counter = {}
    for k in fields:
        counter[k] = Counter()

    for path in train_dataset_files:
        dataset = torch.load(path)
        print(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                val = getattr(ex, k, None)
                if val is not None and k in ('indices', 'src_map', 'alignment', "src_map_multi_attn"):
                    val = [val]
                counter[k].update(val)

    for tgt in ("tgt_pretrain",):
        _build_field_vocab(fields[tgt], counter[tgt],
                           max_size=tgt_vocab_size)
        print(" * %s vocab size: %d." % (tgt, len(fields[tgt].vocab)))

        # All datasets have same num of n_tgt_features,
        # getting the last one is OK.

        for j in range(dataset.n_tgt_feats):
            key = tgt+"_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

    if data_type == 'pretrain':

        if 'src_pretrain_data' in fields:
            src_tuples = ("src_pretrain", 'src_pretrain_data', )
        else:
            src_tuples = ("src_pretrain", )

        for src in src_tuples:
            _build_field_vocab(fields[src], counter[src],
                               max_size=src_vocab_size,
                               min_freq=src_words_min_frequency)
            print(" * %s vocab size: %d." % (src, len(fields[src].vocab)))

            # All datasets have same num of n_src_features,
            # getting the last one is OK.
            iter_range = dataset.n_src_feats
            for j in range(iter_range):
                key = src+"_feat_" + str(j)
                _build_field_vocab(fields[key], counter[key])
                print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

        # Merge the input and output vocabularies.
        if share_vocab:
            assert False, "Not supported yet for pretrain datas"
        else:
            pass
    else:
        assert False

    return fields


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size, tgt_words_min_frequency, hist, ent_pretrain, s2_multi_attn):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """

    counter = {}
    for k in fields:
        counter[k] = Counter()

    for path in train_dataset_files:
        dataset = torch.load(path)
        print(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                # print(k)
                val = getattr(ex, k, None)
                if val is not None and k in ('src1_pretrain',):
                    assert isinstance(val, tuple) and len(val) == 5
                    val = val[2]
                elif val is not None and k in ('indices', 'src_map', 'alignment', "src_map_multi_attn"):
                    val = [val]
                elif k in ("src1_ent_pretrain_data_idx", "src1_ent_pretrain", "src1_ent_pretrain_data") or k.startswith("src1_ent_pretrain_feat_") or k.startswith("src1_ent_pretrain_data_feat_"):
                    # print("yes")
                    continue

                counter[k].update(val)


    if ent_pretrain:
        pre_key = ("src1_pretrain", 'src1_ent_pretrain_not_pretrain')
    else:
        pre_key = ("src1_pretrain", )
    for pre in pre_key:
        _build_field_vocab(fields[pre], counter[pre],
                               max_size=src_vocab_size,
                               min_freq=src_words_min_frequency)
        print(" * %s vocab size: %d." % (pre, len(fields[pre].vocab)))
    for tgt in ("tgt1", "tgt2"):
        if tgt == "tgt2":
            _build_field_vocab(fields[tgt], counter[tgt],
                           max_size=tgt_vocab_size,
                           min_freq=tgt_words_min_frequency)
        else:
            _build_field_vocab(fields[tgt], counter[tgt],
                           max_size=tgt_vocab_size)
        print(" * %s vocab size: %d." % (tgt, len(fields[tgt].vocab)))

        # All datasets have same num of n_tgt_features,
        # getting the last one is OK.

        for j in range(dataset.n_tgt_feats):
            key = tgt+"_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

    if data_type == 'text':
        if hist:
            src_keys = ("src1", "src2", "src1_hist")
        else:
            src_keys = ("src1", "src2")
        for src in src_keys:
            _build_field_vocab(fields[src], counter[src],
                               max_size=src_vocab_size,
                               min_freq=src_words_min_frequency)
            print(" * %s vocab size: %d." % (src, len(fields[src].vocab)))

            # All datasets have same num of n_src_features,
            # getting the last one is OK.
            iter_range = dataset.n_src_hist_feats if src == "src1_hist" else dataset.n_src_feats
            for j in range(iter_range):
                key = src+"_feat_" + str(j)
                _build_field_vocab(fields[key], counter[key])
                print(" * %s vocab size: %d." % (key, len(fields[key].vocab)))

        # Merge the input and output vocabularies.
        if share_vocab:
            assert False, "Not supported yet for pretrain datas"
            # `tgt_vocab_size` is ignored when sharing vocabularies
            print(" * merging src and tgt vocab...")
            merged_vocab = merge_vocabs(
                [fields["src1"].vocab, fields["src1_hist"].vocab, fields["src2"].vocab, fields["tgt2"].vocab],
                fields["src1"])
            fields["src1"].vocab = merged_vocab
            fields["src1_hist"].vocab = merged_vocab
            fields["src2"].vocab = merged_vocab
            fields["tgt2"].vocab = merged_vocab
        else:
            if hist:
            # `tgt_vocab_size` is ignored when sharing vocabularies
                print(" * merging src1 and hist vocab...")
                merged_vocab = merge_vocabs(
                    [fields["src1"].vocab, fields["src1_hist"].vocab],
                    fields["src1"])
                fields["src1"].vocab = merged_vocab
                fields["src1_hist"].vocab = merged_vocab

        if s2_multi_attn:
            merged_vocab = merge_vocabs(
                    [fields["src1"].vocab, fields["src2"].vocab],
                    fields["src2"])
            fields["src1"].vocab = merged_vocab
            fields["src2"].vocab = merged_vocab

            for tmp_i in count():
                s1_key = "src1"+"_feat_" + str(tmp_i)
                s2_key = "src2"+"_feat_" + str(tmp_i)
                if s1_key in fields:
                    assert s2_key in fields
                    merged_vocab = merge_vocabs(
                            [fields[s1_key].vocab, fields[s2_key].vocab],
                            fields[s2_key])
                    fields[s1_key].vocab = merged_vocab
                    fields[s2_key].vocab = merged_vocab
                else:
                    break


    else:
        assert False
    return fields


def _make_examples_nfeats_tpl(data_type, src_path, src_dir,
                              src_seq_length_trunc, sample_rate,
                              window_size, window_stride,
                              window, normalize_audio, src="src1"):
    """
    Process the corpus into (example_dict iterator, num_feats) tuple
    on source side for different 'data_type'.
    """

    if data_type == 'text':
        src_examples_iter, num_src_feats = \
            TextDataset.make_text_examples_nfeats_tpl(
                src_path, src_seq_length_trunc, src)

    elif data_type == 'img':
        src_examples_iter, num_src_feats = \
            ImageDataset.make_image_examples_nfeats_tpl(
                src_path, src_dir)

    elif data_type == 'audio':
        src_examples_iter, num_src_feats = \
            AudioDataset.make_audio_examples_nfeats_tpl(
                src_path, src_dir, sample_rate,
                window_size, window_stride, window,
                normalize_audio)

    return src_examples_iter, num_src_feats


class OrderedIterator(torchtext.data.Iterator):
    # needs to pay attention that during validate and test, data within batch shouldn't be sorted
    def sort_batch_key(self, ex):
        """ Sort using length of source sentences. """

        if "src_pretrain" in ex.__dict__:
            return len(ex.src_pretrain)
        elif "src1" in ex.__dict__:
            return len(ex.src1)
        else:
            return 1
        # return len(ex.src1) if "src1" in ex.__dict__ else 1
    def create_batches(self):
        if self.train:
            def pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_batch_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(b)
                # self.batches.append(sorted(b, key=self.sort_batch_key))
