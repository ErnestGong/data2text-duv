import torch

import onmt.translate.Beam
import onmt.io


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, model2, fields, model_opt, model_opt2, 
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 cuda=False,
                 beam_trace=False,
                 min_length=0,
                 stepwise_penalty=False):
        self.model = model
        self.model2 = model2
        self.model_opt = model_opt
        self.model_opt2 = model_opt2
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty

        if model_opt:
            self.mlb = model_opt.mlb
        else:
            self.mlb = model_opt2.mlb

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data, stage1):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           tgt_plan_map (): mapping between tgt indices and tgt_planning


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        device = torch.device("cuda") if self.cuda else torch.device("cpu")
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        if stage1:
            tgt = "tgt1"
        else:
            tgt = "tgt2"

        vocab = self.fields[tgt].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return a

        def rvar(a):
            if isinstance(a, tuple):
                result_tuple = []
                for each_one in a:
                    if isinstance(each_one, tuple):
                        tmp_tuple = []
                        for each_tensor in each_one:
                            if each_tensor is not None:
                                tmp_tuple.append(var(each_tensor.data.repeat(1, beam_size, 1)))
                            else:
                                tmp_tuple.append(None)
                        result_tuple.append(tuple(tmp_tuple))
                    else:
                        if each_one is not None:
                            result_tuple.append(var(each_one.data.repeat(1, beam_size, 1)))
                        else:
                            result_tuple.append(None)
                return tuple(result_tuple)
            else: 
                return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)


        # (1) Run the encoder on the src.
        if self.model_opt2 and (self.model_opt2.basicencdec or (self.model_opt2.sep_train and self.model_opt2.stage2_train)):
            pass
        else:
            src = onmt.io.make_features(batch, 'src1', data_type)
            src_lengths = None


            device = torch.device("cuda") if self.cuda else torch.device("cpu")

            if self.mlb:
                _, src_lengths = batch.src1
            else:
                src_lengths = torch.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0]).to(device)

            # print("batch:\n")
            # print(batch.__dict__)

            if 'src1_hist' in batch.__dict__:
                src_hist = onmt.io.make_features(batch, 'src1_hist', data_type)
            else:
                src_hist = None
            # except:
            #     src_hist = None


        # assert 
        # assert src_hist is not None, (batch.__dict__)
        
        if self.model_opt is not None and (not self.model_opt.sep_train or (self.model_opt.sep_train and self.model_opt.stage1_train)) and self.model is not None:
            if self.model_opt.use_pretrain:
                # compatability
                if self.model_opt.val_use_pretrain or ((not self.model_opt.val_use_pretrain) and (not self.model_opt.ent_use_pretrain)):
                    val_src_pretrain = batch.__dict__["src1_pretrain"]
                else:
                    val_src_pretrain = None

                if self.model_opt.ent_use_pretrain:
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

            enc_states, memory_bank = self.model.encoder((src, src_hist), src_lengths, require_emb=True)
            if not stage1:
                memory_bank = memory_bank[2] if isinstance(memory_bank, tuple) else memory_bank
            tmp_memory_bank = memory_bank[2] if isinstance(memory_bank, tuple) else memory_bank

            if not self.mlb:
                src_lengths = torch.Tensor(batch_size).type_as(tmp_memory_bank.data) \
                    .long() \
                    .fill_(tmp_memory_bank.size(0))
            model = self.model

        if (self.model_opt2 and (self.model_opt2.basicencdec or (self.model_opt2.sep_train and self.model_opt2.stage2_train))):
            src = onmt.io.make_features(batch, 'src2', data_type)
            _, src_lengths = batch.src2
            emb = src
            if self.model_opt2.use_pretrain:
                stage2_pretrain_src = onmt.io.make_features(batch, 'src1', data_type)
                src_pretrain = batch.__dict__["src1_pretrain"]
                stage2_pretrain_src = (stage2_pretrain_src, src_pretrain, None)
                stage2_tgt = batch.tgt1_planning[0:batch.tgt1.size(0)].unsqueeze(2)[1:-1]
                pretrain_val_result = self.model2.encoder.stage2_pretrain_model(stage2_pretrain_src)
                pretrain_val_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                                zip(torch.transpose(pretrain_val_result, 0, 1), torch.t(torch.squeeze(stage2_tgt, 2)))]
                pretrain_val_input = torch.transpose(torch.cat(pretrain_val_select), 0, 1)  
                emb = (emb, pretrain_val_input, None)              


        elif self.model is None and self.model2 is not None:
            emb = (src, src_hist)
            if data_type == 'text':
                _, src_lengths = batch.src2

        elif not stage1:
            if data_type == 'text':
                _, src_lengths = batch.src2
            inp_stage2 = batch.tgt1_planning.unsqueeze(2)[1:-1]
            index_select = [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                            zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(inp_stage2, 2)))]
            emb = torch.transpose(torch.cat(index_select), 0, 1)

        if not stage1:
            if self.model_opt2 and (self.model_opt2.basicencdec or (self.model_opt2.sep_train and self.model_opt2.stage2_train)):
                if hasattr(self.model_opt2, "s2_multi_attn") and self.model_opt2.s2_multi_attn:
                    src1_input = onmt.io.make_features(batch, 'src1', data_type)
                    assert self.model_opt2.sep_train and self.model_opt2.stage2_train
                    if self.model_opt2.use_pretrain:
                        src1_input = ((src1_input, pretrain_val_result, None), None)
                    emb = (emb, src1_input)

                    if hasattr(self.model_opt2, "mlb") and self.model_opt2.mlb:
                        _, src1_lengths = batch.src1
                    else:
                        src1_lengths = torch.LongTensor(batch.src1.size()[1]).fill_(batch.src1.size()[0]).to(device)
                else:
                    src1_lengths = None


                enc_results = self.model2.encoder(emb, src_lengths, require_emb=True, src1_lengths=src1_lengths)
                if isinstance(enc_results, tuple):
                    if len(enc_results) == 2:
                        enc_states, memory_bank = enc_results
                        src1_result = None
                    elif len(enc_results) == 3:
                        enc_states, memory_bank, src1_result = enc_results
                    else:
                        assert False
            else:
                enc_states, memory_bank = self.model2.encoder(emb, src_lengths)
                src1_result = None
                src1_lengths = None

            model = self.model2

        else:
            src1_lengths = None
            src1_result = None
        dec_states = model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None

        src_map_multi_attn = rvar(batch.src_map_multi_attn.data) if "src_map_multi_attn" in batch.__dict__ else None
        
        memory_bank = rvar(memory_bank if isinstance(memory_bank, tuple) else memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)
        if src1_lengths is not None:
            mem_src1_lengths = src1_lengths.repeat(beam_size)
        else:
            mem_src1_lengths = None

        if src1_result is not None:
            src1_result = rvar(src1_result if isinstance(src1_result, tuple) else src1_result.data)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt2"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)
            # Run one step.
            dec_out, dec_states, attn = model.decoder(
                inp, memory_bank, dec_states, memory_lengths=memory_lengths, src1_result=src1_result, src1_memory_length=mem_src1_lengths)

            if not stage1:
                dec_out = dec_out.squeeze(0)

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                if stage1:
                    upd_attn = unbottle(attn["std"]).data
                    out = upd_attn
                else:
                    out = model.generator.forward(dec_out).data
                    out = unbottle(out)
                    # beam x tgt_vocab
                    beam_attn = unbottle(attn["std"])
            else:
                out = model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map, 
                                                   copy_attn=attn["copy_src1"].squeeze(0) if "copy_src1" in attn else None,
                                                   copy_src_map=src_map_multi_attn,
                                                   copy_gate_res=attn["copy_gate_res"].squeeze(0) if "copy_gate_res" in attn else None)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out[0].data),
                    batch, self.fields[tgt].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                if stage1:
                    b.advance(
                        out[:, j],
                        torch.exp(unbottle(attn["std"]).data[:, j, :memory_lengths[j]]))
                else:
                    b.advance(out[:, j],
                        beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)


        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size

        #if "tgt" in batch.__dict__:
        #    ret["gold_score"] = self._run_target(batch, data, indexes, unbottle)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)

            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        device = torch.device("cuda") if self.cuda else torch.device("cpu")

        gold_scores = torch.FloatTensor(batch.batch_size).fill_(0).to(device)
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
