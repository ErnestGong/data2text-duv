from __future__ import division
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules
import random

ENT_NAME_ID = 1
PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, epoch):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/ppl", self.ppl(), epoch)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), epoch)
        writer.add_scalar(prefix + "/tgtper",  self.n_words / t, epoch)
        writer.add_scalar(prefix + "/lr", lr, epoch)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, model2, train_loss, valid_loss, train_loss2, valid_loss2, optim, optim2, mlb, trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, cuda=False, s2_multi_attn=False):
        # Basic attributes.
        self.model = model
        self.model2 = model2
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_loss2 = train_loss2
        self.valid_loss2 = valid_loss2
        self.optim = optim
        self.optim2 = optim2
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.cuda = cuda
        self.tgt1_padding_idx = PAD_INDEX
        self.mlb = mlb
        self.s2_multi_attn = s2_multi_attn

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        if self.model is not None:
            self.model.train()
        if self.model2 is not None:
            self.model2.train()

    def train(self, train_iter, epoch, model_opt, opt, report_func=None, r_rein_weight=None, pretrain=False):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        if self.model is not None:
            total_stats = Statistics()
            report_stats = Statistics()
        else:
            total_stats = report_stats = None

        if self.model2 is not None:
            total_stats2 = Statistics()
            report_stats2 = Statistics()
        else:
            total_stats2 = report_stats2 = None

        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        reinforce_int_results = []
        reinforce_base_results = []

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            if self.train_loss is not None:
                self.train_loss.cur_dataset = cur_dataset
            if self.train_loss2 is not None:
                self.train_loss2.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            # what is batch.tgt?
            if pretrain:
                pass
            elif opt.sep_train or model_opt.sep_train:
                loss_pad = self.train_loss if opt.stage1_train or model_opt.stage1_train else self.train_loss2
                tgt_select = batch.tgt1 if opt.stage1_train or model_opt.stage1_train else batch.tgt2
                if self.norm_method == "tokens":
                    num_tokens = tgt_select[1:].data.view(-1) \
                        .ne(loss_pad.padding_idx).sum()
                    normalization += num_tokens.item()
                else:
                    normalization += batch.batch_size
            else:
                loss_pad = self.train_loss if self.train_loss is not None else self.train_loss2
                if self.norm_method == "tokens":
                    num_tokens = batch.tgt[1:].data.view(-1) \
                        .ne(loss_pad.padding_idx).sum()
                    normalization += num_tokens.item()
                else:
                    normalization += batch.batch_size

            if accum == self.grad_accum_count:
                if pretrain:
                    self._pretrain_gradient_accumulation(true_batchs, total_stats,
                        report_stats, opt)
                elif model_opt.basicencdec:
                    assert self.model is None
                    self._gradient_accumulation_basic_encdec(true_batchs, total_stats2, report_stats2, normalization)
                else:
                    tmp_reinforce_int_results, tmp_reinforce_base_results = self._gradient_accumulation(
                            true_batchs, total_stats,
                            report_stats, total_stats2, report_stats2, normalization, opt.sep_train or model_opt.sep_train, opt.stage1_train or model_opt.stage1_train, opt.stage2_train or model_opt.stage2_train, opt.reinforce or model_opt.reinforce, opt, r_rein_weight, opt.r_topk_sample, opt.use_pretrain or model_opt.use_pretrain, opt.use_pretrain or model_opt.use_pretrain, opt.val_use_pretrain or model_opt.val_use_pretrain, opt.ent_use_pretrain or model_opt.ent_use_pretrain, epoch)
                    reinforce_int_results.append(tmp_reinforce_int_results)
                    reinforce_base_results.append(tmp_reinforce_base_results)

                if report_func is not None:
                    if self.model is not None:
                        report_stats = report_func(
                                epoch, idx, num_batches,
                                total_stats.start_time, self.optim.lr,
                                report_stats)
                    if self.model2 is not None:
                        report_stats2 = report_func(
                                epoch, idx, num_batches,
                                total_stats2.start_time, self.optim2.lr,
                                report_stats2)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            if pretrain:
                self._pretrain_gradient_accumulation(true_batchs, total_stats,
                    report_stats, opt)
            elif self.model is None:
                self._gradient_accumulation_basic_encdec(true_batchs, total_stats2, report_stats2, normalization)
            else:
                tmp_reinforce_int_results, tmp_reinforce_base_results = self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, total_stats2, report_stats2, normalization, opt.sep_train or model_opt.sep_train, opt.stage1_train or model_opt.stage1_train, opt.stage2_train or model_opt.stage2_train, opt.reinforce or model_opt.reinforce, opt, r_rein_weight, opt.r_topk_sample, opt.use_pretrain or model_opt.use_pretrain, opt.use_pretrain or model_opt.use_pretrain, opt.val_use_pretrain or model_opt.val_use_pretrain, opt.ent_use_pretrain or model_opt.ent_use_pretrain, epoch)
                reinforce_int_results.append(tmp_reinforce_int_results)
                reinforce_base_results.append(tmp_reinforce_base_results)
            true_batchs = []

        if opt.debug:
            torch.save(reinforce_int_results, '%s_debug_rein_int_e%d.pt'
                       % (opt.save_model, epoch))
            torch.save(reinforce_base_results, '%s_debug_rein_base_e%d.pt'
                       % (opt.save_model, epoch))
        return total_stats, total_stats2

    def validate(self, valid_iter, opt, model_opt, pretrain=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        if self.model is not None:
            self.model.eval()
            stats = Statistics()
            stats3 = Statistics()
        else:
            stats = None
            stats3 = None

        if self.model2 is not None:
            self.model2.eval()
            stats2 = Statistics()
        else:
            stats2 = None

        device = torch.device("cuda") if self.cuda else torch.device("cpu")

        with torch.no_grad():
            for batch in valid_iter:
                cur_dataset = valid_iter.get_cur_dataset()
                if self.valid_loss is not None:
                    self.valid_loss.cur_dataset = cur_dataset
                if self.valid_loss2 is not None:
                    self.valid_loss2.cur_dataset = cur_dataset

                if pretrain:
                    src = onmt.io.make_features(batch, 'src_pretrain', self.data_type)
                    # tgt = onmt.io.make_features(batch, 'tgt_pretrain', self.data_type)

                    if opt.pretrain_emb_ent and opt.pre_ent_use_data:
                        src = (src, onmt.io.make_features(batch, 'src_pretrain_data', self.data_type), batch.src_pretrain_data_idx)

                    pretrain_rep, outputs = self.model(src)

                    batch_stats = self.valid_loss.monolithic_compute_loss(
                        batch, outputs, None, stage1=False, pretrain=True, print_full_loss=False)

                    batch_stats3 = self.valid_loss.monolithic_compute_loss(
                        batch, outputs, None, stage1=False, pretrain=True, print_full_loss=True)      
                        
                    stats.update(batch_stats)   
                    stats3.update(batch_stats3)

                elif opt.basicencdec or model_opt.basicencdec:
                    src = onmt.io.make_features(batch, 'src1', self.data_type)

                    if self.mlb:
                        _, src_lengths = batch.src1
                    else:
                        src_lengths = torch.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0]).to(device)

                    if 'src1_hist' in batch.__dict__:
                        src_hist = onmt.io.make_features(batch, 'src1_hist', self.data_type)
                    else:
                        src_hist = None
                    
                    tgt = onmt.io.make_features(batch, 'tgt2')
                    # F-prop through the model.
                    outputs, attns, _, _ = self.model2(src, tgt, src_lengths, require_emb=True)

                    # Compute loss.
                    batch_stats = self.valid_loss2.monolithic_compute_loss(
                        batch, outputs, attns, stage1=False)
                    # Update statistics.
                    stats2.update(batch_stats)

                else:
                    if not opt.sep_train or (opt.sep_train and opt.stage1_train) or opt.reinforce:
                        src = onmt.io.make_features(batch, 'src1', self.data_type)
                        if self.mlb:
                            _, src_lengths = batch.src1
                        else:
                            src_lengths = torch.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0]).to(device)

                        if opt.use_pretrain or model_opt.use_pretrain:
                            if opt.val_use_pretrain or model_opt.val_use_pretrain:
                                val_src_pretrain = batch.__dict__["src1_pretrain"]
                            else:
                                val_src_pretrain = None

                            if opt.ent_use_pretrain or model_opt.ent_use_pretrain:
                                tmp_ent_name, tmp_ent_select_idx = onmt.io.make_pretrain_ent_features(batch, 'src1_ent_pretrain')
                                tmp_ent_data = onmt.io.make_pretrain_ent_features(batch, 'src1_ent_pretrain_data')
                                tmp_ent_data_idx = batch.__dict__["src1_ent_pretrain_data_idx"]

                                if tmp_ent_select_idx:
                                    eq_flag = (batch.__dict__["src1_ent_pretrain_not_pretrain"].transpose(0, 1).unsqueeze(2) == tmp_ent_select_idx[1]).all().item()
                                    assert eq_flag == 1
                                    
                                ent_src_pretrain = (tmp_ent_name, tmp_ent_data, tmp_ent_data_idx, tmp_ent_select_idx)
                            else:
                                ent_src_pretrain = None
                                
                            src = (src, val_src_pretrain, ent_src_pretrain)

                        if 'src1_hist' in batch.__dict__:
                            src_hist = onmt.io.make_features(batch, 'src1_hist', self.data_type)
                        else:
                            src_hist = None
                        tgt = batch.tgt1_planning.unsqueeze(2)
                        # F-prop through the model.
                        outputs, attns, _, memory_bank = self.model((src, src_hist), tgt, src_lengths, require_emb=True)
                        # Compute loss.
                        batch_stats = self.valid_loss.monolithic_compute_loss(
                                batch, outputs, attns, stage1=True)
                        # Update statistics.
                        stats.update(batch_stats)

                    if not opt.sep_train or (opt.sep_train and opt.stage2_train):
                        # assert not opt.reinforce
                        src2 = onmt.io.make_features(batch, 'src2', self.data_type)
                        _, src_lengths = batch.src2
                        
                        if opt.sep_train and (opt.use_pretrain or model_opt.use_pretrain):
                            stage2_pretrain_src = onmt.io.make_features(batch, 'src1', self.data_type)
                            src_pretrain = batch.__dict__["src1_pretrain"]
                            stage2_pretrain_src = (stage2_pretrain_src, src_pretrain, None)
                            stage2_tgt = batch.tgt1_planning[0:batch.tgt1.size(0)].unsqueeze(2)[1:-1]
                            pretrain_val_result = self.model2.encoder.stage2_pretrain_model(stage2_pretrain_src)
                            pretrain_val_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                                            zip(torch.transpose(pretrain_val_result, 0, 1), torch.t(torch.squeeze(stage2_tgt, 2)))]
                            pretrain_val_input = torch.transpose(torch.cat(pretrain_val_select), 0, 1)

                        if opt.sep_train:
                            emb = src2
                        else:
                            inp_stage2 = tgt[1:-1]
                            index_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                                            zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(inp_stage2, 2)))]
                            emb = torch.transpose(torch.cat(index_select), 0, 1)
                        
                        if opt.sep_train and (opt.use_pretrain or model_opt.use_pretrain):
                            emb = (emb, pretrain_val_input, None)

                        if self.s2_multi_attn:
                            src1_input = onmt.io.make_features(batch, 'src1', self.data_type)
                            assert opt.sep_train and opt.stage2_train
                            if opt.sep_train and (opt.use_pretrain or model_opt.use_pretrain):
                                src1_input = ((src1_input, pretrain_val_result, None), None)
                            emb = (emb, src1_input)

                            if self.mlb:
                                _, src1_lengths = batch.src1
                            else:
                                src1_lengths = torch.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0]).to(device)
                        else:
                            src1_lengths = None
                            
                        tgt = onmt.io.make_features(batch, 'tgt2')
                        # F-prop through the model.
                        outputs, attns, _, _ = self.model2(emb, tgt, src_lengths, require_emb=opt.sep_train if opt.sep_train else False, src1_length=src1_lengths)


                        # Compute loss.
                        batch_stats = self.valid_loss2.monolithic_compute_loss(
                            batch, outputs, attns, stage1=False)
                        # Update statistics.
                        stats2.update(batch_stats)

        # Set model back to training mode.
        if self.model is not None:
            self.model.train()
        if self.model2 is not None:
            self.model2.train()

        return stats, stats2, stats3

    def epoch_step(self, ppl, ppl2, epoch):
        if self.optim is not None:
            self.optim.update_learning_rate(ppl, epoch)
        if self.optim2 is not None:
            self.optim2.update_learning_rate(ppl2, epoch)


    def drop_checkpoint(self, opt, epoch, fields, valid_stats, valid_stats2, save_path, save_best=False, curr_best_epoch_name=None):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        model1_name = None
        model2_name = None
        if self.model is not None and self.model2 is not None:
            assert not save_best


        if self.model is not None:
            real_model = (self.model.module
                          if isinstance(self.model, nn.DataParallel)
                          else self.model)
            real_generator = (real_model.generator.module
                              if isinstance(real_model.generator, nn.DataParallel)
                              else real_model.generator) if hasattr(real_model, 'generator') else None

            model_state_dict = real_model.state_dict()
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k}
            generator_state_dict = real_generator.state_dict() if real_generator is not None else None
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim,
            }

            model1_name = ['%s_stage1_acc_%.4f_ppl_%.4f_e%d.pt'
                       % (save_path, valid_stats.accuracy(),
                          valid_stats.ppl(), epoch), valid_stats.accuracy()]

            if save_best:
                save_flag = False
            else:
                save_flag = True

            if save_best: 
                if curr_best_epoch_name[0] is None:
                    curr_best_epoch_name[0] = model1_name[0]
                    curr_best_epoch_name[1] = model1_name[1]
                    save_flag = True
                else:
                    if model1_name[1] > curr_best_epoch_name[1]:
                        save_flag = True
                        curr_best_epoch_name[0] = model1_name[0]
                        curr_best_epoch_name[1] = model1_name[1]

            if save_flag:
                torch.save(checkpoint,
                           '%s_stage1_acc_%.4f_ppl_%.4f_e%d.pt'
                           % (save_path, valid_stats.accuracy(),
                              valid_stats.ppl(), epoch))
            

        if self.model2 is not None:
            real_model = (self.model2.module
                          if isinstance(self.model2, nn.DataParallel)
                          else self.model2)
            real_generator = (real_model.generator.module
                              if isinstance(real_model.generator, nn.DataParallel)
                              else real_model.generator)

            model_state_dict = real_model.state_dict()
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k}
            generator_state_dict = real_generator.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'vocab': onmt.io.save_fields_to_vocab(fields),
                'opt': opt,
                'epoch': epoch,
                'optim': self.optim2,
            }

            model2_name = ['%s_stage2_acc_%.4f_ppl_%.4f_e%d.pt'
                       % (save_path, valid_stats2.accuracy(),
                          valid_stats2.ppl(), epoch), valid_stats2.accuracy()]

            if save_best:
                save_flag = False
            else:
                save_flag = True

            if save_best:  
                if curr_best_epoch_name[0] is None:
                    curr_best_epoch_name[0] = model2_name[0]
                    curr_best_epoch_name[1] = model2_name[1]
                    save_flag = True
                else:
                    if model2_name[1] > curr_best_epoch_name[1]:
                        save_flag = True
                        curr_best_epoch_name[0] = model2_name[0]
                        curr_best_epoch_name[1] = model2_name[1]

            if save_flag:
                torch.save(checkpoint,
                           '%s_stage2_acc_%.4f_ppl_%.4f_e%d.pt'
                           % (save_path, valid_stats2.accuracy(),
                              valid_stats2.ppl(), epoch))

            
        # return model1_name, model2_name
    def _gradient_accumulation_basic_encdec(self, true_batchs, total_stats2, report_stats2, normalization):
        if self.grad_accum_count > 1:
            assert False
            self.model.zero_grad()

        device = torch.device('cuda') if self.cuda else torch.device('cpu')

        for batch in true_batchs:
            #Stage 1
            src = onmt.io.make_features(batch, 'src1', self.data_type)

            if self.mlb:
                _, src_lengths = batch.src1
            else:
                src_lengths = torch.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0]).to(device)

            # src_char is tuple of two dimension tensor (char_len, batch*seq_len) and length

            if 'src1_hist' in batch.__dict__:
                src_hist = onmt.io.make_features(batch, 'src1_hist', self.data_type)
            else:
                src_hist = None

            #Stage 2
            target_size = batch.tgt2.size(0)
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                assert False
                trunc_size = target_size

            dec_state = None
            report_stats2.n_src_words += src_lengths.sum().item()

            if self.data_type == 'text':
                tgt_outer = onmt.io.make_features(batch, 'tgt2')
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model2.zero_grad()
                outputs, attns, dec_state, _ = \
                    self.model2(src, tgt, src_lengths, dec_state, require_emb=True)

                # retain_graph is false for the final truncation
                retain_graph = (j + trunc_size) < (target_size - 1)
                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss2.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization, retain_graph=retain_graph)


                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim2.step()
                total_stats2.update(batch_stats)
                report_stats2.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

    def calculate_rwds(self, set_tgt_recs, set_tgt_ents, words, names, length, batch_no, pos_rwd, neg_rwd, ent_name_freqs, ent_freq_threshold, accu_rwd, lst_tgt_recs, order_rwd):
        gen_set_recs = set()
        gen_set_ents = set()
        tmp_ent_rwds = []
        tmp_rec_rwds = []

        tmp_all_gen_rec = []

        for len_no in range(length[batch_no]):
            # if len_no < tgt_real_len:
            # record prec reward
            gen_rec = words[len_no].squeeze(1)[batch_no].item()
            tmp_all_gen_rec.append(int(gen_rec))
            # print("gen_rec")
            # print(gen_rec)
            # tgt_rec = tgt[len_no][batch_no]
            if gen_rec in set_tgt_recs:
                tmp_rec_rwds.append(pos_rwd)
            else:
                tmp_rec_rwds.append(neg_rwd)
            # entity prec reward
            gen_rec_name = names[gen_rec].item()
            # tgt_rec_name = names[tgt_rec]
            # ent_got_weight = max(ent_name_freqs[batch_no][gen_rec], ent_freq_threshold) if ent_name_freqs is not None else 1.0 
            ent_got_weight = 1.0 
            if gen_rec_name in set_tgt_ents:
                tmp_ent_rwds.append(pos_rwd * ent_got_weight)
            else:
                tmp_ent_rwds.append(neg_rwd * ent_got_weight)

            gen_set_recs.add(gen_rec)
            gen_set_ents.add(gen_rec_name)

            # else:
            #     tmp_rec_rwds.append(neg_rwd)
            #     tmp_ent_rwds.append(neg_rwd)

        if accu_rwd is not None:
            gen_rwd = accu_rwd
        else:
            gen_rwd = 1.0

        # assert len(tmp_all_gen_rec) == len(lst_tgt_recs)
        gen_rec_order_rwd = order_rwd * self.calculate_order_score(lst_tgt_recs, tmp_all_gen_rec)
        gen_rec_recall_rwd = gen_rwd * (1.0 - (len(set_tgt_recs - gen_set_recs) / float(len(set_tgt_recs))))
        gen_ent_recall_rwd = gen_rwd * (1.0 - (len(set_tgt_ents - gen_set_ents) / float(len(set_tgt_ents))))
        return tmp_rec_rwds, tmp_ent_rwds, gen_rec_recall_rwd, gen_ent_recall_rwd, gen_rec_order_rwd

    def calculate_order_score(self, gold, pred):
        # l1 l2 is list of [elem, elem, elem]
        # change it to list of 0, 1, 2 ...
        # l1 l2: [0, 3, 5, 2, 7]
        def norm_dld(l1, l2):
            ascii_start = 0
            # make a string for l1
            # all triples are unique...
            s1 = ''.join((chr(ascii_start+i) for i in range(len(l1))))
            s2 = ''
            next_char = ascii_start + len(s1)
            invalid_flag = False
            for j in range(len(l2)):
                found = None
                #next_char = chr(ascii_start+len(s1)+j)
                for k in range(len(l1)):
                    if l2[j] == l1[k]:
                        found = s1[k]
                        #next_char = s1[k]
                        break
                if found is None:
                    s2 += chr(next_char)
                    next_char += 1
                    if next_char <= 128:
                        pass
                    else:
                        invalid_flag = True
                else:
                    s2 += found
            # return 1- , since this thing gives 0 to perfect matches etc
            if invalid_flag:
                return 1.0
            else:
                return 1.0-normalized_damerau_levenshtein_distance(s1, s2)

        def calc_dld(goldfi, predfi):
            total_score = norm_dld(goldfi, predfi)
            # print("avg score:", total_score)
            return total_score

        return calc_dld(gold, pred)

    def combine_scalar_rwds(self, tmp_rec_rwds, tmp_ent_rwds, recall_rec_rwds, recall_ent_rwds, length, prob, batch_no, rwd_weight1, rwd_weight2, rwd_weight3, rwd_weight4, gamma, baseline_rwd, baseline_mode, rwd_weight5, gen_rec_order_rwd, r_recall_beta):
        tmp_loss_func = []
        tmp_cal_rwds = []
        # init 1.0
        tmp_rwds_so_far = 0.0
        # tmp_rec_rwd_so_far = pos_rwd
        # tmp_ent_rwd_so_far = pos_rwd
        # if prob is not None:
        #     assert len(prob) == length[batch_no], (length[batch_no], len(prob))
        assert len(tmp_rec_rwds) == length[batch_no]
        assert len(tmp_ent_rwds) == length[batch_no]
        for len_no in range(-1, -(length[batch_no]+1), -1):
            now_rwds = (rwd_weight1 * tmp_rec_rwds[len_no] + rwd_weight2 * tmp_ent_rwds[len_no] + rwd_weight3 * recall_rec_rwds + rwd_weight4 * recall_ent_rwds + rwd_weight5 * gen_rec_order_rwd - r_recall_beta) - baseline_rwd
            tmp_rwds_so_far = tmp_rwds_so_far + now_rwds
            # if baseline_mode:
            tmp_cal_rwds.append(tmp_rwds_so_far)
            tmp_rwds_so_far = tmp_rwds_so_far * gamma
            # else:
            #     tmp_loss_func.append(-prob[len_no][batch_no] * now_rwds)
            # if len_no > -length[batch_no]:
            #     tmp_rec_rwd_so_far = tmp_rec_rwd_so_far * gamma + tmp_rec_rwds[len_no-1]
            #     tmp_ent_rwd_so_far = tmp_ent_rwd_so_far * gamma + tmp_ent_rwds[len_no-1]
        tmp_cal_rwds = tmp_cal_rwds[::-1]
        if baseline_mode:
            assert len(tmp_cal_rwds) == length[batch_no]
            return sum(tmp_cal_rwds) / len(tmp_cal_rwds), tmp_cal_rwds
        else:
            for each_id, each_rwd in enumerate(tmp_cal_rwds):
                tmp_loss_func.append(-prob[each_id][batch_no] * each_rwd)
            assert len(tmp_loss_func) == len(tmp_cal_rwds)
            assert len(tmp_loss_func) == length[batch_no]
            return sum(tmp_loss_func) / len(tmp_loss_func), tmp_cal_rwds

    def _reinforce_loss(self, words, prob, batch, length, src, pos_rwd, neg_rwd, gamma, rwd_weight1, rwd_weight2, rwd_weight3, rwd_weight4, ent_freq_threshold, baseline, greedy_words, greedy_length, greedy_prob, accu_rwd, order_rwd, rwd_weight5, r_recall_beta):
        # words <- list(target len) of [batch, 1] LongTensor
        # prob <- list(target len) of [batch] FloatTensor
        # length <- list(batch size) of len int
        # src <- LongTensor [len, batch, feat]
        # size: (len, batch) 
        # ent_freq <- [len, batch]
        tgt = batch.tgt1_planning.detach()[1:]
        words = [each.detach().cpu() for each in words]
        greedy_words = [each.detach().cpu() for each in greedy_words] if baseline else None
        # print("words")
        # print(words)
        # list of example (list of freq for each src)
        ent_name_freqs = batch.ent_freq
        # check how to get real length
        tgt_len, tgt_batch = tgt.size()
        assert tgt_batch == len(length)
        if isinstance(src, tuple):
            tmp_src = src[0]
        else:
            tmp_src = src
        ent_name_ids = tmp_src.detach().cpu()[:, :, ENT_NAME_ID]

        rein_int_results = {"scalar_rwd": [], "rec_rwds": [], "ent_rwds": [], "rec_rec_rwd": [], "ent_rec_rwd": [], "rec_order_rwd": []}
        rein_int_results["action"] = words
        rein_int_results["prob"] = [each.detach().cpu() for each in prob]

        rein_base_results = {"scalar_rwd": [], "rec_rwds": [], "ent_rwds": [], "rec_rec_rwd": [], "ent_rec_rwd": [], "rec_order_rwd": []}
        # BASE also need prob!!!
        if baseline:
            rein_base_results["action"] = greedy_words
            rein_base_results["prob"] = [each.detach().cpu() for each in greedy_prob]

        batch_loss_func = []
        for batch_no in range(tgt_batch):
            tgt_real_len = tgt[:, batch_no].ne(self.tgt1_padding_idx).sum().item()
            # tmp_ent_rwds = []
            # tmp_rec_rwds = []        
            # base_rec_rwds = []
            # base_ent_rwds = []
            names = ent_name_ids[:, batch_no]

            # calculate reward
            # obtain set of target records & entity name
            # assert tgt_real_len == length[batch_no], (tgt_real_len, length[batch_no])
            # TODO: ADD A LIST
            set_tgt_recs = set()
            set_tgt_ents = set()
            lst_tgt_recs = []
            for tgt_len_no in range(tgt_real_len):
                tgt_rec = tgt[tgt_len_no][batch_no].item()
                set_tgt_recs.add(tgt_rec)
                lst_tgt_recs.append(int(tgt_rec))
                set_tgt_ents.add(names[tgt_rec].item())


            tmp_rec_rwds, tmp_ent_rwds, gen_rec_recall_rwd, gen_ent_recall_rwd, gen_rec_order_rwd = self.calculate_rwds(set_tgt_recs, set_tgt_ents, words, names, length, batch_no, pos_rwd, neg_rwd, ent_name_freqs, ent_freq_threshold, accu_rwd, lst_tgt_recs, order_rwd)
            rein_int_results["rec_rwds"].append(tmp_rec_rwds)
            rein_int_results["ent_rwds"].append(tmp_ent_rwds)
            rein_int_results["rec_rec_rwd"].append(gen_rec_recall_rwd)
            rein_int_results["ent_rec_rwd"].append(gen_ent_recall_rwd)
            rein_int_results["rec_order_rwd"].append(gen_rec_order_rwd)
            # gen_set_recs = set()
            # gen_set_ents = set()

            # for len_no in range(length[batch_no]):
            #     # if len_no < tgt_real_len:
            #     # record prec reward
            #     gen_rec = words[len_no].squeeze(1)[batch_no]
            #     # tgt_rec = tgt[len_no][batch_no]
            #     if gen_rec in set_tgt_recs:
            #         tmp_rec_rwds.append(pos_rwd)
            #     else:
            #         tmp_rec_rwds.append(neg_rwd)
            #     # entity prec reward
            #     gen_rec_name = names[gen_rec]
            #     # tgt_rec_name = names[tgt_rec]
            #     if gen_rec_name in set_tgt_ents:
            #         tmp_ent_rwds.append(pos_rwd * ent_name_freqs[batch_no][gen_rec])
            #     else:
            #         tmp_ent_rwds.append(neg_rwd * ent_name_freqs[batch_no][gen_rec])

            #     gen_set_recs.add(gen_rec)
            #     gen_set_ents.add(gen_rec_name)

            #     # else:
            #     #     tmp_rec_rwds.append(neg_rwd)
            #     #     tmp_ent_rwds.append(neg_rwd)
            # gen_rec_recall_rwd = neg_rwd * (len(set_tgt_recs - gen_set_recs) / len(set_tgt_recs))
            # gen_ent_recall_rwd = neg_rwd * (len(set_tgt_ents - gen_set_ents) / len(set_tgt_ents))

            # calculate baseline reward
            if baseline:
                assert greedy_words is not None and greedy_length is not None
                b_tmp_rec_rwds, b_tmp_ent_rwds, b_gen_rec_recall_rwd, b_gen_ent_recall_rwd, b_gen_rec_order_rwd = self.calculate_rwds(set_tgt_recs, set_tgt_ents, greedy_words, names, greedy_length, batch_no, pos_rwd, neg_rwd, ent_name_freqs, ent_freq_threshold, accu_rwd, lst_tgt_recs, order_rwd)

                rein_base_results["rec_rwds"].append(b_tmp_rec_rwds)
                rein_base_results["ent_rwds"].append(b_tmp_ent_rwds)
                rein_base_results["rec_rec_rwd"].append(b_gen_rec_recall_rwd)
                rein_base_results["ent_rec_rwd"].append(b_gen_ent_recall_rwd)
                rein_base_results["rec_order_rwd"].append(b_gen_rec_order_rwd)

                baseline_rwd, baseline_int_rwds = self.combine_scalar_rwds(b_tmp_rec_rwds, b_tmp_ent_rwds, b_gen_rec_recall_rwd, b_gen_ent_recall_rwd, greedy_length, None, batch_no, rwd_weight1, rwd_weight2, rwd_weight3, rwd_weight4, gamma, 0.0, True, rwd_weight5, b_gen_rec_order_rwd, r_recall_beta)
                rein_base_results["scalar_rwd"].append(baseline_int_rwds)
                # for len_no in range(greedy_length[batch_no]):
                #     # if len_no < tgt_real_len:
                #     # record reward
                #     gen_rec = greedy_words[len_no].squeeze(1)[batch_no]
                #     tgt_rec = tgt[len_no][batch_no]
                #     if tgt_rec == gen_rec:
                #         base_rec_rwds.append(pos_rwd)
                #     else:
                #         base_rec_rwds.append(neg_rwd)
                #     # entity reward
                #     gen_rec_name = names[gen_rec]
                #     tgt_rec_name = names[tgt_rec]
                #     if gen_rec_name == tgt_rec_name:
                #         base_ent_rwds.append(pos_rwd * ent_name_freqs[batch_no][gen_rec])
                #     else:
                #         base_ent_rwds.append(neg_rwd * ent_name_freqs[batch_no][gen_rec])
                #     # else:
                #     #     base_rec_rwds.append(neg_rwd)
                #     #     base_ent_rwds.append(neg_rwd)
            else:
                baseline_rwd = 0.0

            # tmp_loss_func = []
            # tmp_rec_rwd_so_far = tmp_rec_rwds[-1]
            # tmp_ent_rwd_so_far = tmp_ent_rwds[-1]
            # for len_no in range(-1, -(length[batch_no]+1), -1):
            #     now_rwds = (rwd_weight * tmp_rec_rwd_so_far + (1.0 - rwd_weight) * tmp_ent_rwd_so_far) - baseline_rwd
            #     tmp_loss_func.append(-prob[-len_no - 1][batch_no] * now_rwds)
            #     tmp_rec_rwd_so_far = tmp_rec_rwd_so_far * gamma + tmp_rec_rwds[len_no-1]
            #     tmp_ent_rwd_so_far = tmp_ent_rwd_so_far * gamma + tmp_ent_rwds[len_no-1]
            # batch_loss_func.append(tmp_loss_func.mean())
            batch_tmp_rwd, batch_int_rwds = self.combine_scalar_rwds(tmp_rec_rwds, tmp_ent_rwds, gen_rec_recall_rwd, gen_ent_recall_rwd, length, prob, batch_no, rwd_weight1, rwd_weight2, rwd_weight3, rwd_weight4, gamma, baseline_rwd, False, rwd_weight5, gen_rec_order_rwd, r_recall_beta)
            batch_loss_func.append(batch_tmp_rwd)
            rein_int_results["scalar_rwd"].append(batch_int_rwds)

        loss = sum(batch_loss_func) / len(batch_loss_func)
        return loss, (rein_int_results, rein_base_results)
            

    # TODO: make sure stage2_use_pretrain
    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, total_stats2, report_stats2, normalization, sep_train, stage1_train, stage2_train, reinforce, opt, r_rein_weight, r_topk_sample, stage2_use_pretrain, use_pretrain, val_use_pretrain, ent_use_pretrain, epoch):

        assert self.grad_accum_count == 1, self.grad_accum_count
        if self.grad_accum_count > 1:
            assert False
            self.model.zero_grad()

        reinforce_int_results = []
        reinforce_base_results = []

        device = torch.device('cuda') if self.cuda else torch.device('cpu')

        for batch in true_batchs:
            if not sep_train or (sep_train and stage1_train) or reinforce:
                #Stage 1
                target_size = batch.tgt1.size(0)

                trunc_size = target_size

                dec_state = None

                src = onmt.io.make_features(batch, 'src1', self.data_type)

                if use_pretrain:
                    if val_use_pretrain:
                        val_src_pretrain = batch.__dict__["src1_pretrain"]
                    else:
                        val_src_pretrain = None

                    if ent_use_pretrain:
                        tmp_ent_name, tmp_ent_select_idx = onmt.io.make_pretrain_ent_features(batch, 'src1_ent_pretrain')
                        tmp_ent_data = onmt.io.make_pretrain_ent_features(batch, 'src1_ent_pretrain_data')
                        tmp_ent_data_idx = batch.__dict__["src1_ent_pretrain_data_idx"]
                        if tmp_ent_select_idx:
                            # print(batch.__dict__["src1_ent_pretrain_not_pretrain"].size())
                            # print(tmp_ent_select_idx[1].size())
                            eq_flag = (batch.__dict__["src1_ent_pretrain_not_pretrain"].transpose(0, 1).unsqueeze(2) == tmp_ent_select_idx[1]).all().item()
                            assert eq_flag == 1
                        ent_src_pretrain = (tmp_ent_name, tmp_ent_data, tmp_ent_data_idx, tmp_ent_select_idx)
                    else:
                        ent_src_pretrain = None
                    src = (src, val_src_pretrain, ent_src_pretrain)

                if self.mlb:
                    _, src_lengths = batch.src1
                    report_stats.n_src_words += src_lengths.sum().item()
                else:
                    src_lengths = torch.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0]).to(device)

                # print("src leng type:")
                # print(src_lengths.type())
                # print(device)

                # src_char is tuple of two dimension tensor (char_len, batch*seq_len) and length
                if 'src1_hist' in batch.__dict__:
                    src_hist = onmt.io.make_features(batch, 'src1_hist', self.data_type)
                else:
                    src_hist = None

                for j in range(0, target_size-1, trunc_size):
                    #setting to value of tgt_planning
                    tgt = batch.tgt1_planning[j: j + trunc_size].unsqueeze(2)

                    # 2. F-prop all but generator.
                    if self.grad_accum_count == 1:
                        self.model.zero_grad()


                    if reinforce:
                        max_length = opt.r_max_length
                        baseline = opt.r_baseline
                    else:
                        max_length = None
                        baseline = False
                        
                    # print(src.type())
                    # print(src_hist.type())
                    # print(tgt.type())
                    # print(src_lengths.type())
                    outputs, attns, dec_state, memory_bank = \
                        self.model((src, src_hist), tgt, src_lengths, dec_state, max_length=max_length, baseline=baseline, reinforce=reinforce, topk_sample=r_topk_sample, require_emb=True)

                    # 3. Compute loss in shards for memory efficiency.

                    if reinforce:
                        # now it needs to propogate the loss
                        all_embs_input, reinforce_prob, batch_real_length, base_all_embs_input, base_rein_prob, base_real_length, mle_outputs, mle_attns = outputs

                        # TOCHECK if this loss is valid
                        mle_loss, batch_stats = self.train_loss.monolithic_compute_loss(
                                batch, outputs, mle_attns, stage1=True, full_loss=True)

                        assert len(reinforce_prob) == len(all_embs_input)
                        # all_embs_input and reinf _prob are all generated including EOS (real length)
                        reinforce_loss, (tmp_rein_int_result, tmp_rein_base_result) = self._reinforce_loss(all_embs_input, reinforce_prob, batch, batch_real_length, src, opt.r_pos_rwd, opt.r_neg_rwd, opt.r_gamma, opt.rwd_weight1, opt.rwd_weight2, opt.rwd_weight3, opt.rwd_weight4, opt.r_ent_freq_threshold, opt.r_baseline, base_all_embs_input, base_real_length, base_rein_prob, opt.r_accu_rwd, opt.r_order_rwd, opt.rwd_weight5, opt.r_recall_beta)

                        reinforce_int_results.append(tmp_rein_int_result)
                        reinforce_base_results.append(tmp_rein_base_result)

                        if opt.r_join_loss:
                            loss = (1.0 - r_rein_weight) * (mle_loss.div(normalization)) + r_rein_weight * reinforce_loss
                        else:
                            loss = reinforce_loss
                        loss.backward()
                        # all_embs_input for reward <- list(target len) of [batch, 1]
                        # use prob and length to calculate loss
                        # prob <- list(target len) of batch
                        # real_length <- list(batch size) of len

                    else:
                        batch_stats = self.train_loss.sharded_compute_loss(
                                batch, outputs, attns, j,
                                trunc_size, self.shard_size, normalization, retain_graph=True if not (sep_train and stage1_train) else False)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                    assert trunc_size > target_size-1

                    # If truncated, don't backprop fully.
                    if dec_state is not None:
                        dec_state.detach()

            if not sep_train or (sep_train and stage2_train):
                # assert not reinforce
                #Stage 2
                target_size = batch.tgt2.size(0)
                if self.trunc_size:
                    trunc_size = self.trunc_size
                else:
                    assert False
                    trunc_size = target_size

                dec_state = None

                # len, batch, nfeat
                src2 = onmt.io.make_features(batch, 'src2', self.data_type)
                _, src_lengths = batch.src2

                if opt.s2_multi_attn and opt.s2_multi_attn_shuffle and epoch >= opt.s2_multi_attn_shuffle_start_epoch:
                    assert src_lengths.size(0) == src2.size(1)
                    sm_len, sm_batch, _ = batch.src_map.size()
                    assert sm_len == src2.size(0) and sm_batch == src2.size(1)

                    for b_id in range(src_lengths.size(0)):
                        if src_lengths[b_id].item() < opt.s2_multi_attn_shuffle_start_len:
                            continue
                        assert src2[:, b_id, 0].ne(PAD_INDEX).sum().item() == src_lengths[b_id].item()
                        tmp_res = []
                        tmp_real_len = 0
                        tmp_src_map = []

                        for len_id in range(src2.size(0)):
                            if src2[len_id, b_id, 0].item() != PAD_INDEX:
                                if random.random() <= opt.s2_multi_attn_shuffle_prob:
                                    pass
                                else:
                                    tmp_res.append(src2[len_id, b_id])
                                    tmp_real_len += 1
                                    tmp_src_map.append(batch.src_map[len_id, b_id])
                            else:
                                break

                        assert len(tmp_res) == tmp_real_len
                        src2[:, b_id, :].fill_(PAD_INDEX)
                        src2[:len(tmp_res), b_id, :] = torch.stack(tmp_res)
                        batch.src_map[:, b_id, :].fill_(0)
                        batch.src_map[:len(tmp_res), b_id, :] = torch.stack(tmp_src_map)
                        src_lengths[b_id] = tmp_real_len



                report_stats2.n_src_words += src_lengths.sum().item()

                #memory bank is of size src_len*batch_size*dim, inp_stage2 is of size inp_len*batch_size*1

                if sep_train and stage2_use_pretrain:
                    stage2_pretrain_src = onmt.io.make_features(batch, 'src1', self.data_type)
                    src_pretrain = batch.__dict__["src1_pretrain"]
                    stage2_pretrain_src = (stage2_pretrain_src, src_pretrain, None)
                    stage2_tgt = batch.tgt1_planning[0:batch.tgt1.size(0)].unsqueeze(2)[1:-1]
                    pretrain_val_result = self.model2.encoder.stage2_pretrain_model(stage2_pretrain_src)
                    pretrain_val_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                                    zip(torch.transpose(pretrain_val_result, 0, 1), torch.t(torch.squeeze(stage2_tgt, 2)))]
                    pretrain_val_input = torch.transpose(torch.cat(pretrain_val_select), 0, 1)

                if sep_train:
                    emb = src2
                else:
                    inp_stage2 = tgt[1:-1]
                    index_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                                    zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(inp_stage2, 2)))]
                    emb = torch.transpose(torch.cat(index_select), 0, 1)

                if sep_train and stage2_use_pretrain:
                    emb = (emb, pretrain_val_input, None)

                if self.s2_multi_attn:
                    src1_input = onmt.io.make_features(batch, 'src1', self.data_type)
                    assert sep_train and stage2_train
                    if stage2_use_pretrain:
                        src1_input = ((src1_input, pretrain_val_result, None), None)
                    emb = (emb, src1_input)

                    if self.mlb:
                        _, src1_lengths = batch.src1
                    else:
                        src1_lengths = torch.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0]).to(device)
                else:
                    src1_lengths = None


                if self.data_type == 'text':
                    tgt_outer = onmt.io.make_features(batch, 'tgt2')
                for j in range(0, target_size-1, trunc_size):
                    # 1. Create truncated target.
                    tgt = tgt_outer[j: j + trunc_size]

                    # 2. F-prop all but generator.
                    if self.grad_accum_count == 1:
                        self.model2.zero_grad()
                    outputs, attns, dec_state, _ = \
                        self.model2(emb, tgt, src_lengths, dec_state, sep_train if sep_train else False, src1_length=src1_lengths)

                    # retain_graph is false for the final truncation
                    retain_graph = (j + trunc_size) < (target_size - 1)
                    # 3. Compute loss in shards for memory efficiency.
                    # print(j, retain_graph)
                    batch_stats = self.train_loss2.sharded_compute_loss(
                            batch, outputs, attns, j,
                            trunc_size, self.shard_size, normalization, retain_graph=retain_graph)


                    # 4. Update the parameters and statistics.
                    if self.grad_accum_count == 1:
                        self.optim2.step()
                    total_stats2.update(batch_stats)
                    report_stats2.update(batch_stats)

                    # If truncated, don't backprop fully.
                    if dec_state is not None:
                        dec_state.detach()

            if not sep_train or (sep_train and stage1_train) or reinforce:
                # 4. Update the parameters and statistics.
                self.optim.step()

        if self.grad_accum_count > 1:
            assert False
            self.optim.step()

        return reinforce_int_results, reinforce_base_results


    def _pretrain_gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, opt):

        assert self.grad_accum_count == 1, self.grad_accum_count
        if self.grad_accum_count > 1:
            assert False
            self.model.zero_grad()

        device = torch.device('cuda') if self.cuda else torch.device('cpu')

        for batch in true_batchs:
            src = onmt.io.make_features(batch, 'src_pretrain', self.data_type)
            # tgt = onmt.io.make_features(batch, 'tgt_pretrain', self.data_type)

            # 2. F-prop all but generator.
            if self.grad_accum_count == 1:
                self.model.zero_grad()
            else:
                assert False, "Not supported yet"

            if opt.pretrain_emb_ent and opt.pre_ent_use_data:
                src = (src, onmt.io.make_features(batch, 'src_pretrain_data', self.data_type), batch.src_pretrain_data_idx)


            # output size: len*len, batch, prob
            pretrain_rep, outputs = self.model(src)

            # 3. Compute loss in shards for memory efficiency
            batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, None, None,
                    None, None, None, retain_graph=False, pretrain=True)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            self.optim.step()
