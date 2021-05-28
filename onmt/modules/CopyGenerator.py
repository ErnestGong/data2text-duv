import torch.nn as nn
import torch.nn.functional as F
import torch


import onmt
import onmt.io
from onmt.Utils import aeq


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.

    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.

    The copy generator is an extended version of the standard
    generator that computse three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a
      word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary

    """
    def __init__(self, input_size, tgt_dict, marg, s2_multi_attn_multi_copy):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, len(tgt_dict))
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict
        self.marg = marg
        self.s2_multi_attn_multi_copy = s2_multi_attn_multi_copy

    def forward(self, hidden, attn, src_map, align=None, ptrs=None, copy_attn=None, copy_src_map=None, copy_gate_res=None):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        if slen != slen_:
            assert slen_ > slen
            assert (src_map[slen:, :, :] == 0).all().item() == 1
            src_map = src_map[:slen, :, :]
            slen_, batch, cvocab = src_map.size()

        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_dict.stoi[onmt.io.PAD_WORD]] = -float('inf')
        prob = F.softmax(logits)

        # Probability of copying p(z=1) batch.
        p_copy = F.sigmoid(self.linear_copy(hidden))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))

        if self.s2_multi_attn_multi_copy:
            assert not (not self.marg and self.training)

        if not self.marg and self.training:
            align_unk = align.eq(0).float().view(-1, 1)
            align_not_unk = align.ne(0).float().view(-1, 1)
            out_prob = torch.mul(prob, align_unk.expand_as(prob))
            mul_attn = torch.mul(attn, align_not_unk.expand_as(attn))
            mul_attn = torch.mul(mul_attn, ptrs.view(-1, slen_).float())
        else:
            out_prob = torch.mul(prob,  1 - p_copy.expand_as(prob))
            mul_attn = torch.mul(attn, p_copy.expand_as(attn))

        copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
                              .transpose(0, 1),
                              src_map.transpose(0, 1)).transpose(0, 1)

        if self.s2_multi_attn_multi_copy:
            if copy_gate_res is not None:
                cp_weight = copy_gate_res.squeeze(1)
                # print(copy_gate_res.size())
                # print(cp_weight.size())
            else:
                cp_weight = 0.5

            batch_by_tlen_c, slen_c = copy_attn.size()
            aeq(batch_by_tlen, batch_by_tlen_c)
            cmap_slen, cmap_batch, cmap_cvocab = copy_src_map.size()
            aeq(slen_c, cmap_slen)
            mul_copy_attn = torch.mul(copy_attn, p_copy.expand_as(copy_attn))
            copy_copy_prob = torch.bmm(mul_copy_attn.view(-1, cmap_batch, slen_c)
                              .transpose(0, 1),
                              copy_src_map.transpose(0, 1)).transpose(0, 1)
            assert copy_prob.size(2) <= copy_copy_prob.size(2)
            if copy_prob.size(2) < copy_copy_prob.size(2):
                need_more = copy_copy_prob.size(2) - copy_prob.size(2)
                cvocab = max([cvocab, cmap_cvocab])
                copy_prob = torch.cat([copy_prob, torch.zeros(copy_prob.size(0), copy_prob.size(1), need_more).to(copy_prob.device)], 2)

            # print(cp_weight.size())
            # print(copy_prob.size())
            copy_prob = cp_weight * copy_prob + (1.0 - cp_weight) * copy_copy_prob



        copy_prob = copy_prob.contiguous().view(-1, cvocab)

        return torch.cat([out_prob, copy_prob], 1), p_copy

class CopyGeneratorCriterion(object):
    def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.pad = pad

    def __call__(self, scores, align, target):
        # Compute unks in align and target for readability
        align_unk = align.eq(0).float()
        align_not_unk = align.ne(0).float()
        target_unk = target.eq(0).float()
        target_not_unk = target.ne(0).float()

        # Copy probability of tokens in source
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            out = out + tmp.mul(target_not_unk)
            # Add score for when word is unk in both align and tgt
            out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            # Forced copy. Add only probability for not-copied tokens
            out = out + tmp.mul(align_unk)

        # Drop padding.
        loss = -out.log().mul(target.ne(self.pad).float())
        return loss


class CopyGeneratorLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """
    def __init__(self, generator, tgt_vocab,
                 force_copy, normalize_by_length, marg, s2_multi_attn_multi_copy,
                 eps=1e-20):
        super(CopyGeneratorLossCompute, self).__init__(
            generator, tgt_vocab)

        # We lazily load datasets when there are more than one, so postpone
        # the setting of cur_dataset.
        self.cur_dataset = None
        self.marg = marg
        self.force_copy = force_copy
        self.normalize_by_length = normalize_by_length
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy,
                                                self.padding_idx)
        self.s2_multi_attn_multi_copy = s2_multi_attn_multi_copy
        if self.s2_multi_attn_multi_copy:
            assert self.marg

        if self.marg:
            print("don't use pointers")
        if not self.marg:
            self.switch_loss_criterion = nn.BCELoss(size_average=False)
        else:
            self.switch_loss_criterion = None

    def _make_shard_state(self, batch, output, range_, attns, print_full_loss=False):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        if self.marg:
            if self.s2_multi_attn_multi_copy:
                return {
                    "output": output,
                    "target": batch.tgt2[range_[0] + 1: range_[1]],
                    "copy_attn": attns.get("copy"),
                    "align": batch.alignment[range_[0] + 1: range_[1]],
                    "copy_copy_attn": attns.get("copy_src1"),
                    "copy_gate_res": attns.get("copy_gate_res")
                }
            else:
                return {
                    "output": output,
                    "target": batch.tgt2[range_[0] + 1: range_[1]],
                    "copy_attn": attns.get("copy"),
                    "align": batch.alignment[range_[0] + 1: range_[1]],
                    "copy_gate_res": attns.get("copy_gate_res")
                }
        else:
            return {
                "output": output,
                "target": batch.tgt2[range_[0] + 1: range_[1]],
                "copy_attn": attns.get("copy"),
                "align": batch.alignment[range_[0] + 1: range_[1]],
                "ptrs": batch.ptrs[range_[0] + 1: range_[1]],
                "copy_gate_res": attns.get("copy_gate_res")

            }

    def _compute_loss(self, batch, output, target, copy_attn, align, ptrs=None, copy_copy_attn=None, copy_gate_res=None):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        if self.marg:
            assert ptrs is None

        # print(copy_attn.size())
        # print(copy_copy_attn.size())
        # print(batch.__dict__.keys())
        scores, p_copy = self.generator(self._bottle(output),
                                self._bottle(copy_attn),
                                batch.src_map, align, ptrs, self._bottle(copy_copy_attn) if copy_copy_attn is not None else None, batch.src_map_multi_attn if "src_map_multi_attn" in batch.__dict__ else None, copy_gate_res)
        loss = self.criterion(scores, align, target)
        if not self.marg:
            switch_loss = self.switch_loss_criterion(p_copy, align.ne(0).float().view(-1, 1))
        else:
            switch_loss = None
        scores_data = scores.clone()
        scores_data = onmt.io.TextDataset.collapse_copy_scores(
                self._unbottle(scores_data, batch.batch_size),
                batch, self.tgt_vocab, self.cur_dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        correct_mask = target_data.eq(0) * align.detach().ne(0)
        correct_copy = (align.detach() + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy

        # Compute sum of perplexities for stats
        loss_data = loss.sum().clone()
        stats = self._stats(loss_data, scores_data, target_data)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            pad_ix = batch.dataset.fields['tgt2'].vocab.stoi[onmt.io.PAD_WORD]
            tgt_lens = batch.tgt2.ne(pad_ix).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        if not self.marg:
            loss = loss + switch_loss
        return loss, stats
