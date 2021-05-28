#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import random

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts
import numpy as np
import json
from onmt.io.PretrainTextDataset import PRETRAIN_PAD_INDEX
from onmt.io.DatasetBase import PAD_WORD

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()

if opt.hier_meta is not None:
    with open(opt.hier_meta, "r") as f:
        opt.hier_meta = json.load(f)

if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    if opt.tgt_word_vec_size == -1:
        opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn2 = (opt.encoder_type2 == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

# more reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if opt.seed > 0:
    np.random.seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid and len(opt.gpuid) > 0:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient

    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(opt.tensorboard_log_dir, comment="Onmt")


def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        if opt.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard(
                "progress", writer, lr, epoch * num_batches + batch)
        report_stats = onmt.Statistics()

    return report_stats


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train, pretrain):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train
        self.pretrain = pretrain

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset


    # def sort_minibatch_key(self, ex):
    #     """ Sort using length of source sentences and length of target sentence """
    #     #Needed for packed sequence
    #     if not self.pretrain:
    #         return len(ex.src1), len(ex.tgt1)
    #     else:
    #         return len(ex.src_pretrain), len(ex.tgt_pretrain)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        if not self.pretrain:
            def sort_minibatch_key(ex):
                """ Sort using length of source sentences and length of target sentence """
                #Needed for packed sequence
                return len(ex.src1), len(ex.tgt1)
        else:
            def sort_minibatch_key(ex):
                """ Sort using length of source sentences and length of target sentence """
                #Needed for packed sequence
                return len(ex.src_pretrain), len(ex.tgt_pretrain)
                
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort_key=sort_minibatch_key,
            sort=False, sort_within_batch=True,
            repeat=False)


def make_dataset_iter(datasets, fields, opt, is_train=True, pretrain=False):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        global max_src_in_batch, max_tgt_in_batch

        if pretrain:
            def batch_size_fn(new, count, sofar):
                global max_src_in_batch, max_tgt_in_batch
                if count == 1:
                    max_src_in_batch = 0
                    max_tgt_in_batch = 0
                max_src_in_batch = max(max_src_in_batch,  len(new.src_pretrain) + 2)
                max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt_pretrain) + 1)
                src_elements = count * max_src_in_batch
                tgt_elements = count * max_tgt_in_batch
                return max(src_elements, tgt_elements)
        else:
            def batch_size_fn(new, count, sofar):
                global max_src_in_batch, max_tgt_in_batch
                if count == 1:
                    max_src_in_batch = 0
                    max_tgt_in_batch = 0
                max_src_in_batch = max(max_src_in_batch,  len(new.src) + 2)
                max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 1)
                src_elements = count * max_src_in_batch
                tgt_elements = count * max_tgt_in_batch
                return max(src_elements, tgt_elements)

    device = torch.device("cuda") if opt.gpuid and len(opt.gpuid) > 0 else torch.device("cpu")

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train, pretrain)


def make_loss_compute(model, tgt_vocab, opt, stage1=True, src_vocab=None):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """

    if opt.pretrain_emb:
        compute = onmt.Loss.PretrainLossCompute(
            PRETRAIN_PAD_INDEX, src_vocab.stoi[PAD_WORD], opt.pre_no_self_loss, opt.pre_loss_type, opt.pre_hinge_thre, opt.pre_only_neighbour_loss, opt.pre_ent_type == "bin")
    elif not stage1:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength, opt.mlb or opt.marginal, opt.s2_multi_attn_multi_copy)
    else:
        compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab, opt.mlb,
            label_smoothing=opt.label_smoothing, decoder_type=opt.decoder_type1)

    device = torch.device("cuda") if use_gpu(opt) else torch.device("cpu")
    compute.to(device)

    return compute


def train_model(model, model2, fields, optim, optim2, data_type, model_opt, opt):
    if model is not None:
        fields_vocab = None if opt.pretrain_emb else fields["tgt1"].vocab
        src_vocab = fields["src_pretrain"].vocab if opt.pretrain_emb else None
        train_loss = make_loss_compute(model, fields_vocab, opt, stage1=True, src_vocab=src_vocab)
        valid_loss = make_loss_compute(model, fields_vocab, opt, stage1=True, src_vocab=src_vocab)
    else:
        train_loss = valid_loss = None

    if model2 is not None:
        fields_vocab = None if opt.pretrain_emb else fields["tgt2"].vocab
        train_loss2 = make_loss_compute(model2, fields_vocab, opt, stage1=False)
        valid_loss2 = make_loss_compute(model2, fields_vocab, opt, stage1=False)
    else:
        train_loss2 = valid_loss2 = None

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    cuda = False
    if opt.gpuid and len(opt.gpuid) > 0:
        cuda = True


    s2_multi_attn_flag = None
    if hasattr(model_opt, "s2_multi_attn"):
        s2_multi_attn_flag = opt.s2_multi_attn or model_opt.s2_multi_attn
    else:
        s2_multi_attn_flag = opt.s2_multi_attn
    trainer = onmt.Trainer(model, model2, train_loss, valid_loss, train_loss2, valid_loss2, optim, optim2, opt.mlb or model_opt.mlb,
                           trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count, cuda, s2_multi_attn_flag)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)

    this_run_epoch = 0
    rein_weight = opt.r_rein_weight
    curr_best_epoch_name = [None, None]
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        if opt.pretrain_emb:
            train_iter = make_dataset_iter(lazily_load_dataset("pretrain.train"),
                                       fields, opt, pretrain=opt.pretrain_emb)
        else:
            train_iter = make_dataset_iter(lazily_load_dataset("train"),
                                       fields, opt, pretrain=opt.pretrain_emb)

        print("reinforce weight {}, mle weight {}".format(rein_weight, 1.0 - rein_weight))
        if opt.s2_multi_attn and opt.s2_multi_attn_shuffle and epoch >= opt.s2_multi_attn_shuffle_start_epoch:
            print("enabling shuffle...")
        train_stats, train_stats2 = trainer.train(train_iter, epoch, model_opt, opt, report_func, rein_weight, opt.pretrain_emb)

        rein_weight = 1.0 - ((1.0 - opt.r_rein_weight) * (opt.r_rein_not_weight_decay ** this_run_epoch))

        if train_stats is not None:
            print('Train perplexity: %g' % train_stats.ppl())
            print('Train accuracy: %g' % train_stats.accuracy())
        if train_stats2 is not None:
            print('Train perplexity2: %g' % train_stats2.ppl())
            print('Train accuracy2: %g' % train_stats2.accuracy())

        # 2. Validate on the validation set.
        if opt.pretrain_emb:
            valid_iter = make_dataset_iter(lazily_load_dataset("pretrain.valid"),
                                       fields, opt,
                                       is_train=False, pretrain=opt.pretrain_emb)
        else:
            valid_iter = make_dataset_iter(lazily_load_dataset("valid"),
                                       fields, opt,
                                       is_train=False, pretrain=opt.pretrain_emb)
        valid_stats, valid_stats2, valid_stats3 = trainer.validate(valid_iter, opt, model_opt, pretrain=opt.pretrain_emb)
        if valid_stats is not None:
            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation accuracy: %g' % valid_stats.accuracy())
        if opt.pre_only_neighbour_loss:
            print('Validation full perplexity: %g' % valid_stats3.ppl())
            print('Validation full accuracy: %g' % valid_stats3.accuracy())            
        if valid_stats2 is not None:
            print('Validation perplexity2: %g' % valid_stats2.ppl())
            print('Validation accuracy2: %g' % valid_stats2.accuracy())

        # 3. Log to remote server.
        if opt.exp_host and train_stats is not None:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)
        if opt.tensorboard and train_stats is not None:
            train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl() if valid_stats is not None else None, valid_stats2.ppl() if valid_stats2 is not None else None, epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats, valid_stats2, opt.save_model, opt.save_best, curr_best_epoch_name)
        this_run_epoch += 1

def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def lazily_load_dataset(corpus_type, prefix=None):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "pretrain.train", "pretrain.valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts_path = opt.data + '.' + corpus_type + '.[0-9]*.pt' if prefix is None else prefix + '.' + corpus_type + '.[0-9]*.pt'
    pts = sorted(glob.glob(pts_path))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!

        pt = opt.data + '.' + corpus_type + '.pt' if prefix is None else prefix + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, data_type, checkpoint1, checkpoint2, opt):
    if checkpoint1 is not None or checkpoint2 is not None:
        # assert checkpoint1['vocab'].stoi == checkpoint2['vocab'].stoi if checkpoint1 is not None and checkpoint2 is not None else True
        print('Loading vocab from checkpoint at {} (or {}).'.format(opt.train_from1, opt.train_from2))
        fields = onmt.io.load_fields_from_vocab(
            checkpoint1['vocab'] if checkpoint1 is not None else checkpoint2['vocab'], data_type, opt)
    else:
        if data_type == "pretrain":
            vocab_file = opt.data + '.pretrain.vocab.pt'
        else:
            vocab_file = opt.data + '.vocab.pt'
        fields = onmt.io.load_fields_from_vocab(
            torch.load(vocab_file), data_type, opt)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == "pretrain":
        if opt.pretrain_emb_ent and opt.pre_ent_use_data:
            print(' * vocabulary size. pretrain src = %d; pretrain tgt = %d; pretrain src data = %d' %
              (len(fields['src_pretrain'].vocab), len(fields['tgt_pretrain'].vocab), len(fields['src_pretrain_data'].vocab)))
        else:
            print(' * vocabulary size. pretrain src = %d; pretrain tgt = %d' %
              (len(fields['src_pretrain'].vocab), len(fields['tgt_pretrain'].vocab)))        
    elif data_type == 'text' or data_type == 'box':
        print(' * vocabulary size. source1 = %d; source1_hist = %d, target1 = %d, source2 = %d; target2 = %d' %
              (len(fields['src1'].vocab), len(fields['src1_hist'].vocab) if 'src1_hist' in fields else -1, len(fields['tgt1'].vocab), len(fields['src2'].vocab), len(fields['tgt2'].vocab)))
    else:
        assert False
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields, pretrain=False):

    if pretrain:
        src_features = onmt.io.collect_features(fields, side='src_pretrain')
        tgt_features = onmt.io.collect_features(fields, side='tgt_pretrain')
    else:
        src_features = onmt.io.collect_features(fields, side='src1')
        tgt_features = onmt.io.collect_features(fields, side='tgt1')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint1, checkpoint2, pretrain=False, tgt_opt_num=None, text_ent_tgt_opt_num=None):
    print('Building model...')

    if pretrain:
        model1 = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                      use_gpu(opt), checkpoint1, stage1=False, pretrain=pretrain, tgt_opt_num=tgt_opt_num)
        model2 = None
    elif opt.basicencdec or model_opt.basicencdec:
        model1 = None
        model2 = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                       use_gpu(opt), checkpoint2, stage1=False, basic_enc_dec=True)
    elif opt.sep_train or model_opt.sep_train:
        assert opt.stage1_train or model_opt.stage1_train or opt.stage2_train or model_opt.stage2_train
        if opt.stage1_train or model_opt.stage1_train:
            model1 = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                      use_gpu(opt), checkpoint1, stage1=True, tgt_opt_num=tgt_opt_num, use_pretrain=opt.use_pretrain or model_opt.use_pretrain, pretrain_word_dict=fields["src1_pretrain"].vocab, text_ent_tgt_opt_num=text_ent_tgt_opt_num, 
                                                          ent_use_pretrain=opt.ent_use_pretrain or model_opt.ent_use_pretrain, val_use_pretrain=opt.val_use_pretrain or model_opt.val_use_pretrain)
            # print(model1.encoder.embeddings.pretrain_model_layer.pretrain_model.transformer.layer_norm.weight)
        else:
            model1 = None

        if opt.stage2_train or model_opt.stage2_train:
            model2 = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                       use_gpu(opt), checkpoint2, stage1=False, tgt_opt_num=tgt_opt_num, use_pretrain=False, pretrain_word_dict=fields["src1_pretrain"].vocab,
                                                            stage2_use_pretrain=opt.use_pretrain or model_opt.use_pretrain)
        else:
            model2 = None

    else:
        # assert checkpoint2 is None
        model1 = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                      use_gpu(opt), checkpoint1, stage1=True, tgt_opt_num=tgt_opt_num, use_pretrain=opt.use_pretrain or model_opt.use_pretrain, pretrain_word_dict=fields["src1_pretrain"].vocab, text_ent_tgt_opt_num=text_ent_tgt_opt_num, 
                                                          ent_use_pretrain=opt.ent_use_pretrain or model_opt.ent_use_pretrain, val_use_pretrain=opt.val_use_pretrain or model_opt.val_use_pretrain)
        model2 = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                       use_gpu(opt), checkpoint2, stage1=False)
    if len(opt.gpuid) > 1:
        assert False, "Not supported yet"
        print('Multi gpu training: ', opt.gpuid)
        if model1 is not None:
            model1 = nn.DataParallel(model1, device_ids=opt.gpuid, dim=1)
        if model2 is not None:
            model2 = nn.DataParallel(model2, device_ids=opt.gpuid, dim=1)
    if model1 is not None:
        print(model1)
    if model2 is not None:
        print(model2)

    return model1, model2


def build_optim(model, checkpoint):
    if checkpoint is not None:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.named_parameters())

    return optim


def main():
    print('Experiment 22-4.4 using attn_dim of 64')
    # Load checkpoint if we resume from a previous training.
    if opt.train_from1 or opt.train_from2:
        print('Loading checkpoint from {} and {}'.format(opt.train_from1, opt.train_from2))
        checkpoint1 = torch.load(opt.train_from1,
                                map_location=lambda storage, loc: storage) if opt.train_from1 is not None else None
        checkpoint2 = torch.load(opt.train_from2,
                                map_location=lambda storage, loc: storage) if opt.train_from2 is not None else None
        model_opt1 = checkpoint1['opt'] if checkpoint1 is not None else None
        model_opt2 = checkpoint2['opt'] if checkpoint2 is not None else None
        # assert model_opt1 == model_opt2 if (model_opt1 is not None and model_opt2 is not None) else True
        # assert checkpoint1['epoch'] == checkpoint2['epoch'] if (checkpoint1 is not None and checkpoint2 is not None) else True
        # I don't like reassigning attributes of opt: it's not clear.
        model_opt = model_opt1 if model_opt1 is not None else model_opt2
        opt.start_epoch = checkpoint1['epoch'] + 1 if checkpoint1 is not None else checkpoint2['epoch'] + 1

    else:
        checkpoint1 = checkpoint2 = None
        model_opt = opt

    tgt_opt_num = None
    text_ent_tgt_opt_num = None
    if opt.pretrain_emb:
        assert not opt.train_from1 and not opt.train_from2
        first_dataset = next(lazily_load_dataset("pretrain.train"))
        tgt_opt_num = first_dataset.tgt_opt_num
        data_type = first_dataset.data_type
        fields = load_fields(first_dataset, data_type, checkpoint1, checkpoint2, opt)
        # Report src/tgt features.
        print("reporting features...")
        collect_report_features(fields, pretrain=True)
    else:
        # Peek the fisrt dataset to determine the data_type.
        # (All datasets have the same data_type).
        first_dataset = next(lazily_load_dataset("train"))
        data_type = first_dataset.data_type

        if opt.use_pretrain or model_opt.use_pretrain:
            test_pretrain = next(lazily_load_dataset("pretrain.train", opt.val_pretrain_data))
            tgt_opt_num = test_pretrain.tgt_opt_num
            if opt.ent_pretrain_data:
                ent_test_pretrain = next(lazily_load_dataset("pretrain.train", opt.ent_pretrain_data))
                text_ent_tgt_opt_num = ent_test_pretrain.tgt_opt_num
            else:
                ent_test_pretrain, text_ent_tgt_opt_num = None, None
            
            

        # Load fields generated from preprocess phase.
        fields = load_fields(first_dataset, data_type, checkpoint1, checkpoint2, opt)
        # print(fields.keys())

        # Report src/tgt features.
        print("reporting features...")
        collect_report_features(fields)

    # Build model.
    model1, model2 = build_model(model_opt, opt, fields, checkpoint1, checkpoint2, opt.pretrain_emb, tgt_opt_num, text_ent_tgt_opt_num)
    if model1 is not None:
        tally_parameters(model1)
        # for each_para in model1.parameters():
        #     if not each_para.is_cuda:
        #         print("model1's {} param is not on cuda".format(each_para))
    if model2 is not None:
        tally_parameters(model2)
        # for each_para in model2.parameters():
        #     if not each_para.is_cuda:
        #         print("model2's {} param is not on cuda".format(each_para))
    check_save_model_path()

    # Build optimizer.
    if model1 is not None:
        optim1 = build_optim(model1, checkpoint1 if (not opt.reinforce and not opt.finetunefromstart) else None)
    else:
        optim1 = None
    if model2 is not None:
        # load_ckpt = checkpoint1 if not (opt.sep_train or model_opt.sep_train) else checkpoint2
        optim2 = build_optim(model2, checkpoint2 if (not opt.reinforce and not opt.finetunefromstart) else None)
    else:
        optim2 = None

    # CHECK HERE
    # Do training.
    train_model(model1, model2, fields, optim1, optim2, data_type, model_opt, opt)

    # If using tensorboard for logging, close the writer after training.
    if opt.tensorboard:
        writer.close()


if __name__ == "__main__":
    main()
