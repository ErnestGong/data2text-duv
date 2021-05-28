#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys

import torch

import onmt.io
import opts
import random
import numpy as np
import codecs

DELIM="￨"

def makedir(path):
    path = os.path.split(path)[0]
    if not os.path.exists(path):
        os.makedirs(path)

def check_existing_pt_files(opt):
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup exisiting pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(opt.seed)

    makedir(opt.save_data)

    check_existing_pt_files(opt)

    return opt

def build_save_pretrain_dataset_in_shards(src_corpus, tgt_corpus, fields,
                                      corpus_type, opt, src_data_corpus, src_data_idx):
    '''
    Divide the big corpus into shards, and build dataset separately.
    This is currently only for data_type=='text'.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.

    To tackle this, we only read in part of the corpus file of size
    `max_shard_size`(actually it is multiples of 64 bytes that equals
    or is slightly larger than this size), and process it into dataset,
    then write it to disk along the way. By doing this, we only focus on
    part of the corpus at any moment, thus effectively reducing memory use.
    According to test, this method can reduce memory footprint by ~50%.

    Note! As we process along the shards, previous shards might still
    stay in memory, but since we are done with them, and no more
    reference to them, if there is memory tight situation, the OS could
    easily reclaim these memory.

    If `max_shard_size` is 0 or is larger than the corpus size, it is
    effectively preprocessed into one dataset, i.e. no sharding.

    NOTE! `max_shard_size` is measuring the input corpus size, not the
    output pt file size. So a shard pt file consists of examples of size
    2 * `max_shard_size`(source + target).
    '''

    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning. The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to use less memory." % src_corpus)

    if opt.max_shard_size != 0:
        print(' * divide corpus into shards and build dataset separately'
              '(shard_size = %d bytes).' % opt.max_shard_size)

    ret_list = []
    src_iter = onmt.io.ShardedPretrainCorpusIterator(
                src_corpus, opt.src_seq_length_trunc,
                "src_pretrain", opt.max_shard_size)

    tgt_iter = onmt.io.ShardedPretrainCorpusIterator(
                tgt_corpus, opt.tgt_seq_length_trunc,
                "tgt_pretrain", opt.max_shard_size,
                assoc_iter=src_iter)
    if opt.ent_use_data:
        src_data_iter = onmt.io.ShardedPretrainCorpusIterator(
                    src_data_corpus, opt.src_data_seq_length_trunc,
                    "src_pretrain_data", opt.max_shard_size,
                    assoc_iter=src_iter)
    else:
        src_data_iter = None

    index = 0
    while not src_iter.hit_end():
        index += 1
        dataset = onmt.io.PretrainTextDataset(
                fields, src_iter, tgt_iter, src_data_iter,
                src_iter.num_feats, tgt_iter.num_feats, src_data_iter.num_feats if src_data_iter else 0,
                src_seq_length=opt.pretrain_src_seq_length,
                tgt_seq_length=opt.pretrain_tgt_seq_length,
                src_data_seq_length=opt.pretrain_src_data_seq_length,
                ent_idx_file=src_data_idx
                )

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:s}.{:d}.pt".format(
                opt.save_data, "pretrain", corpus_type, index)
        print(" * saving %s data shard to %s." % (corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

    return ret_list

def build_save_text_dataset_in_shards(src_corpus, src_hist_corpus, tgt_corpus, src_corpus2, tgt_corpus2, fields,
                                      corpus_type, opt, pointers, ent_freq, pretrain_usage, pretrain_usage_ent):
    '''
    Divide the big corpus into shards, and build dataset separately.
    This is currently only for data_type=='text'.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.

    To tackle this, we only read in part of the corpus file of size
    `max_shard_size`(actually it is multiples of 64 bytes that equals
    or is slightly larger than this size), and process it into dataset,
    then write it to disk along the way. By doing this, we only focus on
    part of the corpus at any moment, thus effectively reducing memory use.
    According to test, this method can reduce memory footprint by ~50%.

    Note! As we process along the shards, previous shards might still
    stay in memory, but since we are done with them, and no more
    reference to them, if there is memory tight situation, the OS could
    easily reclaim these memory.

    If `max_shard_size` is 0 or is larger than the corpus size, it is
    effectively preprocessed into one dataset, i.e. no sharding.

    NOTE! `max_shard_size` is measuring the input corpus size, not the
    output pt file size. So a shard pt file consists of examples of size
    2 * `max_shard_size`(source + target).
    '''

    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning. The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to use less memory." % src_corpus)

    if opt.max_shard_size != 0:
        print(' * divide corpus into shards and build dataset separately'
              '(shard_size = %d bytes).' % opt.max_shard_size)

    ret_list = []
    src_iter = onmt.io.ShardedTextCorpusIterator(
                src_corpus, opt.src_seq_length_trunc,
                "src1", opt.max_shard_size)
    src_hist_iter = onmt.io.ShardedTextCorpusIterator(
                src_hist_corpus, opt.src_seq_length_trunc,
                "src1_hist", opt.max_shard_size, assoc_iter=src_iter) if src_hist_corpus is not None else None

    tgt_iter = onmt.io.ShardedTextCorpusIterator(
                tgt_corpus, opt.tgt_seq_length_trunc,
                "tgt1", opt.max_shard_size,
                assoc_iter=src_iter)
    src_iter2 = onmt.io.ShardedTextCorpusIterator(
                src_corpus2, opt.src_seq_length_trunc,
                "src2", opt.max_shard_size)
    tgt_iter2 = onmt.io.ShardedTextCorpusIterator(
                tgt_corpus2, opt.tgt_seq_length_trunc,
                "tgt2", opt.max_shard_size,
                assoc_iter=src_iter2)

    index = 0
    while not src_iter.hit_end():
        index += 1
        dataset = onmt.io.TextDataset(
                fields, src_iter, src_hist_iter, tgt_iter, src_iter2, tgt_iter2,
                src_iter.num_feats, src_hist_iter.num_feats if src_hist_iter is not None else None, tgt_iter.num_feats, src_iter2.num_feats, tgt_iter2.num_feats,
                src_seq_length=opt.src_seq_length,
                tgt_seq_length=opt.tgt_seq_length,
                dynamic_dict=opt.dynamic_dict, pointers_file=pointers, ent_freq_file=ent_freq, pretrain_usage_file=pretrain_usage, pretrain_usage_ent_file=pretrain_usage_ent, s2_multi_attn_multi_copy=opt.s2_multi_attn_multi_copy)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.pt".format(
                opt.save_data, corpus_type, index)
        print(" * saving %s data shard to %s." % (corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

    return ret_list

def build_save_pretrain_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_pretrain_src
        tgt_corpus = opt.train_pretrain_tgt
        if opt.ent_use_data:
            src_data_corpus = opt.train_pretrain_data
            src_data_idx = opt.train_pretrain_data_idx
        else:
            src_data_corpus = None
            src_data_idx = None
    else:
        src_corpus = opt.valid_pretrain_src
        tgt_corpus = opt.valid_pretrain_tgt
        if opt.ent_use_data:
            src_data_corpus = opt.valid_pretrain_data
            src_data_idx = opt.valid_pretrain_data_idx
        else:
            src_data_corpus = None
            src_data_idx = None

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    # if opt.data_type == 'pretrain':
    return build_save_pretrain_dataset_in_shards(
            src_corpus, tgt_corpus, fields,
            corpus_type, opt, src_data_corpus, src_data_idx)
    assert False


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src1
        src_hist_corpus = opt.train_src1_hist
        tgt_corpus = opt.train_tgt1
        src_corpus2 = opt.train_src2
        tgt_corpus2 = opt.train_tgt2
        pointers = opt.train_ptr
        ent_freq = opt.train_ent_freq
        pretrain_usage = opt.train_src1_pretrain
        pretrain_usage_ent = opt.train_src1_pretrain_ent
        # pretrain_src = opt.train_pretrain_src
        # pretrain_tgt = opt.train_pretrain_tgt
    else:
        src_corpus = opt.valid_src1
        src_hist_corpus = opt.valid_src1_hist
        tgt_corpus = opt.valid_tgt1
        src_corpus2 = opt.valid_src2
        tgt_corpus2 = opt.valid_tgt2
        pointers = None
        ent_freq = None
        pretrain_usage = opt.valid_src1_pretrain
        pretrain_usage_ent = opt.valid_src1_pretrain_ent
        # pretrain_src = opt.valid_pretrain_src
        # pretrain_tgt = opt.valid_pretrain_tgt

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    # if opt.data_type == 'text':
    return build_save_text_dataset_in_shards(
            src_corpus, src_hist_corpus, tgt_corpus, src_corpus2, tgt_corpus2, fields,
            corpus_type, opt, pointers=pointers, ent_freq=ent_freq, pretrain_usage=pretrain_usage, pretrain_usage_ent=pretrain_usage_ent)

    assert False
    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
    dataset = onmt.io.build_dataset(
                fields, opt.data_type, src_corpus, tgt_corpus,
                src_dir=opt.src_dir,
                src_seq_length=opt.src_seq_length,
                tgt_seq_length=opt.tgt_seq_length,
                src_seq_length_trunc=opt.src_seq_length_trunc,
                tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
                dynamic_dict=opt.dynamic_dict,
                sample_rate=opt.sample_rate,
                window_size=opt.window_size,
                window_stride=opt.window_stride,
                window=opt.window)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    print(" * saving %s dataset to %s." % (corpus_type, pt_file))
    torch.save(dataset, pt_file)

    return [pt_file]


def build_save_vocab(train_dataset, fields, opt, pretrain_src_vocab, ent_pretrain_src_field):
    fields = onmt.io.build_vocab(train_dataset, fields, "text",
                                 opt.share_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab_size,
                                 opt.tgt_words_min_frequency,
                                 opt.train_src1_hist is not None,
                                 ent_pretrain_src_field is not None,
                                 opt.s2_multi_attn)

    # Can't save fields, so remove/reconstruct at training time.
    fields["src1_pretrain"].vocab = {"not_pretrain": fields["src1_pretrain"].vocab,
                                     "pretrain": pretrain_src_vocab,
                                     "pretrain_pad_index": fields["src1_pretrain"].pad_index}

    if ent_pretrain_src_field:
        fields["src1_ent_pretrain"].vocab = {
            "not_pretrain": fields['src1_ent_pretrain_not_pretrain'].vocab,
            "pretrain": ent_pretrain_src_field["src_pretrain"].vocab
        }

        count = 0
        while True:
            if "src_pretrain_feat_" + str(count) in ent_pretrain_src_field:
                fields["src1_ent_pretrain_feat_" + str(count)].vocab = ent_pretrain_src_field["src_pretrain_feat_" + str(count)].vocab
            else:
                break
            count += 1

        if "src_pretrain_data" in ent_pretrain_src_field:
            fields["src1_ent_pretrain_data"].vocab = ent_pretrain_src_field["src_pretrain_data"].vocab
            count = 0
            while True:
                if "src_pretrain_data_feat_" + str(count) in ent_pretrain_src_field:
                    fields["src1_ent_pretrain_data_feat_" + str(count)].vocab = ent_pretrain_src_field["src_pretrain_data_feat_" + str(count)].vocab
                else:
                    break
                count += 1

    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)

def build_save_pretrain_vocab(train_dataset, fields, opt):
    fields = onmt.io.build_pretrain_vocab(train_dataset, fields, "pretrain",
                                 opt.share_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab_size)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.pretrain.vocab.pt'
    torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)

def main():
    opt = parse_args()

    print("Extracting features...")
    # feature num except first element - value
    if opt.mode == "textgen":
        src_nfeats1 = onmt.io.get_num_features('text', opt.train_src1, 'src1')
        if opt.train_src1_hist:
            src_hist_nfeats1 = onmt.io.get_num_features('text', opt.train_src1_hist, 'src1_hist')
        else:
            src_hist_nfeats1 = None

        tgt_nfeats1 = onmt.io.get_num_features('text', opt.train_tgt1, 'tgt1')
        src_nfeats2 = onmt.io.get_num_features('text', opt.train_src2, 'src2')
        tgt_nfeats2 = onmt.io.get_num_features('text', opt.train_tgt2, 'tgt2')

        if opt.train_src1_pretrain_ent:
            src_ent_nfeats, src_ent_data_nfeats = onmt.io.get_num_features('text', opt.train_src1_pretrain_ent, 'src1_ent_pretrain')
        else:
            src_ent_nfeats, src_ent_data_nfeats = None, None

        print(" * number of source features- stage 1: %d." % src_nfeats1)
        if src_hist_nfeats1:
            print(" * number of source history features- stage 1: %d." % src_hist_nfeats1)
        print(" * number of target features- stage 1: %d." % tgt_nfeats1)
        print(" * number of source features- stage 2: %d." % src_nfeats2)
        if src_ent_nfeats:
            print(" * number of source ent usage name features: %s." % str(src_ent_nfeats))
        if src_ent_data_nfeats:
            print(" * number of source ent usage data features: %s." % str(src_ent_data_nfeats))

    elif opt.mode == "pretrain":
        src_pretrain_nfeats = onmt.io.get_num_features("pretrain", opt.train_pretrain_src, 'src_pretrain')
        tgt_pretrain_nfeats = onmt.io.get_num_features("pretrain", opt.train_pretrain_tgt, 'tgt_pretrain')
        if opt.ent_use_data:
            src_pretrain_data_nfeats = onmt.io.get_num_features("pretrain", opt.train_pretrain_data, 'src_pretrain_data')
        else:
            src_pretrain_data_nfeats = None


        print(" * number of source features- pretrain: %d." % src_pretrain_nfeats)
        print(" * number of target features- pretrain: %d." % tgt_pretrain_nfeats)
        if opt.ent_use_data:
            print(" * number of ent data features- pretrain: %d." % src_pretrain_data_nfeats)
    else:
        assert False, "invalid mode"

    print("Building `Fields` object...")
    # getting the pretrain padding index from src text
    with codecs.open(opt.train_src1, "r", "utf-8") as f:
        test_line = f.readline().strip().split(" ")
        test_line = [each_one.strip().split(DELIM)[0] for each_one in test_line]
    pretrain_pad_index = test_line.index("<blank>")

    if opt.mode == "textgen":
        fields = onmt.io.get_fields("text", src_nfeats1, tgt_nfeats1, src_hist_nfeats1, pretrain_pad_index, True if opt.mlb else False, n_src_ent_features=src_ent_nfeats, n_src_ent_data_features=src_ent_data_nfeats, marg=True if opt.marginal else False, s2_multi_attn_multi_copy=opt.s2_multi_attn_multi_copy)

        print("Building & saving textgen training data...")
        train_dataset_files = build_save_dataset('train', fields, opt)
        print("Building & saving textgen vocabulary...")
        # TODO: Load val/ent pretrain vocab ~ 
        pretrain_usage_field = onmt.io.load_fields_from_vocab(
            torch.load(opt.pretrain_vocab), "pretrain", opt)
        if opt.pretrain_ent_vocab:
            pretrain_ent_usage_field = onmt.io.load_fields_from_vocab(
                torch.load(opt.pretrain_ent_vocab), "pretrain", opt, ent_use_data=opt.ent_use_data)
        else:
            pretrain_ent_usage_field = None
        # fields = dict([(k, f) for (k, f) in fields.items()
        #            if k in dataset.examples[0].__dict__])
        build_save_vocab(train_dataset_files, fields, opt, pretrain_usage_field["src_pretrain"].vocab, pretrain_ent_usage_field)
        print("Building & saving textgen validation data...")
        build_save_dataset('valid', fields, opt)

    elif opt.mode == "pretrain":
        pretrain_fields = onmt.io.get_fields("pretrain", src_pretrain_nfeats, tgt_pretrain_nfeats, None, None, True if opt.mlb else False, n_src_pretrain_data_features=src_pretrain_data_nfeats)
        print("Building & saving pretrain training data...")
        pretrain_dataset_files = build_save_pretrain_dataset('train', pretrain_fields, opt)
        print("Building & saving pretrain vocabulary...")
        build_save_pretrain_vocab(pretrain_dataset_files, pretrain_fields, opt)
        print("Building & saving pretrain validation data...")
        build_save_pretrain_dataset('valid', pretrain_fields, opt)

    else:
        assert False, "invalid mode"


if __name__ == "__main__":
    main()
