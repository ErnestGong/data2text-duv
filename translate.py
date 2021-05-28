#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts
from onmt.io.DatasetBase import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD


parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def makedir(path):
    path = os.path.split(path)[0]
    if not os.path.exists(path):
        os.makedirs(path)

makedir(opt.output)

def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the model.
    model = None
    model_opt = None
    
    if opt.model is not None:
        fields, model, model_opt = \
            onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__, stage1=True)


    model2 = None
    model_opt2 = None
    if opt.model2 is not None:
        fields2, model2, model_opt2 = \
            onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__, stage1=False)

        if model_opt2.basicencdec or (model_opt2.sep_train and model_opt2.stage2_train):
            fields = None
            model = None
            model_opt = None

        if model is None:
            fields = fields2
            model_opt = model_opt2

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    # Test data
    # print(opt.src1_hist)
    # print(fields)
    # print(dir(fields))
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src1, opt.src1_hist, opt.src1_pretrain, opt.tgt1,
                                 opt.src2, opt.tgt2,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False,
                                 pretrain_usage_ent_path=opt.src1_pretrain_ent, model_opt=model_opt)

    if not model_opt.pretrain_emb:
        def sort_minibatch_key(ex):
            """ Sort using length of source sentences and length of target sentence """
            #Needed for packed sequence
            if hasattr(ex, "tgt1"):
                return len(ex.src1), len(ex.tgt1)
            return len(ex.src1)
    else:
        def sort_minibatch_key(ex):
            """ Sort using length of source sentences and length of target sentence """
            #Needed for packed sequence
            if hasattr(ex, "tgt_pretrain"):
                return len(ex.src_pretrain), len(ex.tgt_pretrain)
            return len(ex.src_pretrain)

    # def sort_minibatch_key(ex):
    #     """ Sort using length of source sentences and length of target sentence """
    #     #Needed for packed sequence
    #     if hasattr(ex, "tgt1"):
    #         return len(ex.src1), len(ex.tgt1)
    #     return len(ex.src1)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    # need to be changed for rnn encoder of basic encoder-decoder framework
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=device,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_key=sort_minibatch_key,
        sort_within_batch=False, shuffle=False)

    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)
    tgt_plan_map = None

    if opt.src2 is None and not model_opt.pretrain_emb:
        tgt_plan_map = {}
        for j, entry in enumerate(fields["tgt1"].vocab.itos):
            if j<4:
                tgt_plan_map[j] = j
            else:
                tgt_plan_map[j] = int(entry)
    translator = onmt.translate.Translator(
        model, model2, fields, model_opt, model_opt2,
        beam_size=opt.beam_size,
        n_best=opt.n_best,
        global_scorer=scorer,
        max_length=opt.max_length,
        copy_attn=model_opt.copy_attn and tgt_plan_map is None,
        cuda=opt.cuda,
        beam_trace=opt.dump_beam != "",
        min_length=opt.min_length,
        stepwise_penalty=opt.stepwise_penalty)
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, has_tgt=False)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    stage1 = opt.stage1

    pretrain_rep_lst = []
    for batch in data_iter:
        if model_opt.pretrain_emb:
            src = onmt.io.make_features(batch, 'src_pretrain', opt.data_type)
            pretrain_rep, outputs = model(src)
            if opt.debug:
                pretrain_rep_lst.append(pretrain_rep.cpu())
            outputs = outputs.cpu()
            if outputs.size(2) == 2:
                outputs = outputs + torch.FloatTensor([1e-7, 0.0]).unsqueeze(0).unsqueeze(0).expand(outputs.size(0), outputs.size(1), -1)
                _, max_id = outputs.max(2)
                max_id = 1 - max_id
                # print("processed")
            else:
                assert False, "Not supported yet"

            # get the mask
            src_padding_idx = fields["src_pretrain"].vocab.stoi[PAD_WORD]
            inp_non_padding = batch.src_pretrain.ne(src_padding_idx)
            batch_inp_non_padding = inp_non_padding.transpose(0, 1)
            batch_inp_row_non_pad = batch_inp_non_padding.unsqueeze(2).expand(-1, -1, batch_inp_non_padding.size(1))
            batch_inp_col_non_pad = batch_inp_non_padding.repeat(1, batch_inp_non_padding.size(1)).view(batch_inp_non_padding.size(0), batch_inp_non_padding.size(1), batch_inp_non_padding.size(1))

            if model_opt.pre_no_self_loss:
                mask_self_index = list(range(batch_inp_row_non_pad.size(1)))
                # total_mask batch, len, len
                total_mask = (batch_inp_row_non_pad * batch_inp_col_non_pad)
                total_mask[:, mask_self_index, mask_self_index] = 0
                output_mask = total_mask.view(batch_inp_non_padding.size(0), batch_inp_row_non_pad.size(1) * batch_inp_row_non_pad.size(1)).cpu()
            else:
                output_mask = (batch_inp_row_non_pad * batch_inp_col_non_pad).view(batch_inp_non_padding.size(0), batch_inp_row_non_pad.size(1) * batch_inp_row_non_pad.size(1)).cpu()

            # print(max_id.size(), output_mask.size())
            results = [torch.masked_select(a, i) for a, i in
                                            zip(max_id.transpose(0, 1), output_mask)]

            for each_exp in results:
                out_file.write(" ".join([str(each_pred.item()) for each_pred in each_exp]))
                out_file.write("\n")
                out_file.flush()
                    

        else:
            batch_data = translator.translate_batch(batch, data, stage1)
            translations = builder.from_batch(batch_data, stage1)

            for trans in translations:
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if opt.tgt2:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent)

                if stage1:
                    n_best_preds = [" ".join([str(entry.item()) for entry in pred])
                                    for pred in trans.pred_sents[:opt.n_best]]
                else:
                    n_best_preds = [" ".join(pred)
                                    for pred in trans.pred_sents[:opt.n_best]]
                out_file.write('\n'.join(n_best_preds))
                out_file.write('\n')
                out_file.flush()

                if opt.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    os.write(1, output.encode('utf-8'))

    if opt.debug:
        torch.save(pretrain_rep_lst, opt.output + ".debug.pretrain.rep")
        
    if not hasattr(model_opt, "pretrain_emb") or not model_opt.pretrain_emb:
        _report_score('PRED', pred_score_total, pred_words_total)
        if opt.tgt2:
            _report_score('GOLD', gold_score_total, gold_words_total)
            if opt.report_bleu:
                _report_bleu()
            if opt.report_rouge:
                _report_rouge()

        if opt.dump_beam:
            import json
            json.dump(translator.beam_accum,
                      codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
