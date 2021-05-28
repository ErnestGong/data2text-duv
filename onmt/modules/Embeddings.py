import torch
import torch.nn as nn

from onmt.modules import Elementwise
from onmt.Utils import aeq
from onmt.Models import RNNEncoder
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000, pos_type="sincos", only_pe=False):
        assert pos_type is not None
        self.only_pe = only_pe
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))

        if pos_type == "sincos":
            print("using sincos position option...")
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                                 -(math.log(10000.0) / dim)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe = pe.unsqueeze(1)
            super(PositionalEncoding, self).__init__()
            self.register_buffer('pe', pe)
        elif pos_type == "posEmb":
            super(PositionalEncoding, self).__init__()
            print("using position embedding option...")
            self.pe = nn.Parameter(torch.Tensor(max_len, 1, dim))
            # self.pe = self.pe.unsqueeze(1)
        else:
            raise ValueError("wrong pos type")
            
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        emb = emb * math.sqrt(self.dim)
        if step is None:
            if self.only_pe:
                emb = self.pe[:emb.size(0)].expand(-1, emb.size(1), -1)

            else:
                emb = emb + self.pe[:emb.size(0)]
        else:
            if self.only_pe:
                emb = self.pe[step].expand(emb.size(1), -1)
            else:
                emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

class dumpEmb(object):
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

class ProcessChar(nn.Module):
    """
        args:
    """
    def __init__(self, char_vocab_size, char_emb_dim, char_pad_idx, char_rnn_type, char_hidden_size, char_num_layers, char_dropout, char_bidirection, emb_luts):
        super(ProcessChar, self).__init__()
        self.char_enabled_flag = (char_vocab_size is not None)
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=char_pad_idx) if self.char_enabled_flag else None
        self.char_enc = RNNEncoder(char_rnn_type, char_bidirection, char_num_layers, char_hidden_size, char_dropout, dumpEmb(char_emb_dim), sort_data=True) if self.char_enabled_flag else None
        self.char_bidirection = char_bidirection

        self.emb_luts = emb_luts

    def obtainChar(self, rep, batch_size):
        length = 2 if self.char_bidirection else 1
        rep = rep.narrow(0, rep.size(0)-length, length).transpose(0, 1).contiguous().view(rep.size(1), -1)
        rep = rep.view(batch_size, -1, rep.size(1)).transpose(0, 1).contiguous()
        return rep

    def forward(self, src):
        if self.char_enabled_flag:
            assert isinstance(src, tuple)
            orig_rep = self.emb_luts(src[0])
            char_input = src[1]
            # char_input: first is data (char_len, batch*seq_len), second is lengths (batch*seq_len)
            assert isinstance(char_input, tuple)
            char_enc_final, _ = self.char_enc(self.char_emb(char_input[0]), lengths=char_input[1])
            assert isinstance(char_enc_final, tuple)
            char_rep = char_enc_final[0]
            return torch.cat((orig_rep, self.obtainChar(char_rep, orig_rep.size(1))), 2)
        else:
            if isinstance(src, tuple):
                src = src[0]
            return self.emb_luts(src)

class HistScalarMap(nn.Module):
    """
        args:

    """
    def __init__(self, start_dim, end_dim, col_start_dim, col_end_dim, emb_size, num_interval, in_dim, enable_mlp=False):
        super(HistScalarMap, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.col_start_dim = col_start_dim
        self.col_end_dim = col_end_dim

        if enable_mlp:
            self.mlp = nn.Sequential(nn.Linear(in_dim, emb_size), nn.ReLU())
            score_in_dim = emb_size
        else:
            self.mlp = None
            score_in_dim = in_dim

        self.get_score = nn.Sequential(
            nn.Linear(score_in_dim, num_interval), 
            nn.Softmax(dim=2), 
            nn.Linear(num_interval, emb_size))

    """
    Computes the embeddings for mixed hist and rank.

    Args:
        input (`LongTensor`): index tensor `[len x batch x all_emb]`
    Return:
        `FloatTensor`: word embeddings (mapping hist scalar to rep) `[len x batch x all_emb]`
    """
    def forward(self, x):
        hist_rep = x.narrow(2, self.start_dim, (self.end_dim-self.start_dim))
        col_rep = x.narrow(2, self.col_start_dim, (self.col_end_dim-self.col_start_dim))

        concated_rep = torch.cat((col_rep, hist_rep), 2)
        if self.mlp is not None:
            concated_rep = self.mlp(concated_rep)
        concated_rep = self.get_score(concated_rep)

        concated_lst = []
        if self.start_dim > 0:
            concated_lst.append(x.narrow(2, 0, self.start_dim))
        concated_lst.append(concated_rep)
        if x.size(2)-self.end_dim > 0:
            concated_lst.append(x.narrow(2, self.end_dim, x.size(2)-self.end_dim))
        return torch.cat(concated_lst, 2)
        
class HistScalarAttnMap(nn.Module):
    """
        args:

    """
    def __init__(self, start_dim, end_dim, emb_size, num_interval, dim, attn_type="general"):
        super(HistScalarAttnMap, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

        # attn params
        self.hier_mix_attn = False
        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")
        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            if self.hier_mix_attn:
                self.linear_query = nn.Linear(dim*2, dim, bias=True)
            else:
                self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"

        self.sm = nn.Softmax(2)
        self.tanh = nn.Tanh()


        self.interval_rep = nn.Parameter(torch.FloatTensor(num_interval, emb_size))

        print("interval_rep size is {}\n".format(self.interval_rep.size()))

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        if self.hier_mix_attn:
            aeq(src_dim*2, tgt_dim)
        else:
            aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            target_input_dim = dim * 2 if self.hier_mix_attn else dim
            wq = self.linear_query(h_t.view(-1, target_input_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)


    """
    Computes the embeddings for mixed hist and rank.

    Args:
        input (`LongTensor`): index tensor `[len x batch x all_emb]`
    Return:
        `FloatTensor`: word embeddings (mapping hist scalar to rep) `[len x batch x all_emb]`
    """
    def forward(self, x):
        _, batch, _ = x.size()
        hist_rep = x.narrow(2, self.start_dim, (self.end_dim-self.start_dim))


        attn_score = self.score(hist_rep.transpose(0, 1).contiguous(), self.interval_rep.unsqueeze(0).expand(batch, -1, -1))

        attn_score = self.sm(attn_score)

        # align vector (batch, target_len, dim)
        align_vector = torch.bmm(
            attn_score,
            self.interval_rep.unsqueeze(0).expand(batch, -1, -1)
            )

        hist_map_result = align_vector.transpose(0, 1)

        concated_lst = []
        if self.start_dim > 0:
            concated_lst.append(x.narrow(2, 0, self.start_dim))
        concated_lst.append(hist_map_result)
        if x.size(2)-self.end_dim > 0:
            concated_lst.append(x.narrow(2, self.end_dim, x.size(2)-self.end_dim))
        return torch.cat(concated_lst, 2)

class MixHistRank(nn.Module):
    """
    args:
        start_dim is the index of first dimension of rank representation
        end_dim is the index of first dimension after history representation
        * [start_dim:end_dim] includes rank and history representation

    """
    def __init__(self, start_dim, end_dim):
        super(MixHistRank, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.inter_dim = self.start_dim+int((self.end_dim-self.start_dim)/2)

        in_dim = self.end_dim - self.start_dim
        self.gate = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    """
    Computes the embeddings for mixed hist and rank.

    Args:
        input (`LongTensor`): index tensor `[len x batch x embedding_size*n]`
    Return:
        `FloatTensor`: word embeddings `[len x batch x embedding_size*(n-1)]`
    """
    def forward(self, x):
        rank_rep = x.narrow(2, self.start_dim, (self.inter_dim-self.start_dim))
        hist_rep = x.narrow(2, self.inter_dim, (self.end_dim-self.inter_dim))
        gate_value = self.gate(torch.cat((rank_rep, hist_rep), 2))
        mixed_rep = gate_value * rank_rep + (1.0 - gate_value) * hist_rep
        concated_lst = []
        if self.start_dim > 0:
            concated_lst.append(x.narrow(2, 0, self.start_dim))
        concated_lst.append(mixed_rep)
        if x.size(2)-self.end_dim > 0:
            concated_lst.append(x.narrow(2, self.end_dim, x.size(2)-self.end_dim))
        return torch.cat(concated_lst, 2)


class UsePretrainModel(nn.Module):
    """
    args:
        start_dim is the index of first dimension of rank representation
        end_dim is the index of first dimension after history representation
        * [start_dim:end_dim] includes rank and history representation

    """
    def __init__(self, pretrain_model, not_pretrain_emb, feat_merge, embeddings, only_val=False, stage2=False, ent_pretrain_model=None, ent_not_pretrain_emb=None):
        super(UsePretrainModel, self).__init__()
        if stage2:
            self.pretrain_model, self.not_pretrain_emb = None, None
            assert not only_val
        else:
            self.pretrain_model = pretrain_model
            self.not_pretrain_emb = not_pretrain_emb

        self.ent_pretrain_model = ent_pretrain_model
        self.ent_not_pretrain_emb = ent_not_pretrain_emb
        self.enable_ent = ent_pretrain_model is not None
        self.enable_val = pretrain_model is not None
        assert self.enable_val or self.enable_ent or stage2

        self.stage2 = stage2

        # self.feat_merge = feat_merge
        # self.embeddings = embeddings
        if only_val:
            assert feat_merge is None and embeddings is None
            self.emb_luts = None
        else:
            assert isinstance(embeddings, list)

            if self.enable_ent and self.enable_val:
                self.emb_luts = Elementwise(feat_merge, embeddings[2:])
            elif self.enable_ent:
                self.ent_val_luts = Elementwise(feat_merge, embeddings[0:1])
                self.emb_luts = Elementwise(feat_merge, embeddings[2:])
            elif self.enable_val or self.stage2:
                self.emb_luts = Elementwise(feat_merge, embeddings[1:])
            else:
                assert False

    def load_pretrained_contextual_rep(self, model_file, fixed, ent_model_file, ent_fixed):
        if model_file:
            pretrained_model = torch.load(model_file,
                            map_location=lambda storage, loc: storage)
            self.pretrain_model.load_state_dict(pretrained_model['model'])
            if fixed:
                for param in self.pretrain_model.parameters():
                    param.requires_grad = False

        if ent_model_file:
            ent_pretrained_model = torch.load(ent_model_file,
                            map_location=lambda storage, loc: storage)
            self.ent_pretrain_model.load_state_dict(ent_pretrained_model['model'])
            if ent_fixed:
                for param in self.ent_pretrain_model.parameters():
                    param.requires_grad = False                


    """
    Computes the embeddings for mixed hist and rank.

    Args:
        input (`LongTensor`): index tensor `[len x batch x embedding_size*n]`
    Return:
        `FloatTensor`: word embeddings `[len x batch x embedding_size*(n-1)]`
    """
    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == 3
        # orig_src: len x batch x nfeat
        orig_src, pretrain_input, ent_pretrain_input = x
        # batch, seq_no, len
        # batch, len
        # batch, len
        # TODO: Take care the input data first
        # ent_pretrain_input should include:

        # ent name
        # batch, seq_no, seq_len, nfeats+1 -> tensor
        # ent data
        # batch, seq_no, seq_len, nfeats+1 -> tensor
        # ent data index
        # batch, seq_no, valid_tuples -> list of var element length
        # reverse idx
        # batch, tuple_num -> tensor -> elem is about idx in ent name (seq_no * seq_len) 

        if not self.stage2:
            if self.enable_val:
                padded_pre, padded_not_pre, reverse_idxs = pretrain_input
                batch, seq_no, seq_len = padded_pre.size()
                # size (len, batch * seq_no, hidden)
                attn_res, _ = self.pretrain_model(padded_pre.view(padded_pre.size(0) * padded_pre.size(1), padded_pre.size(2)).transpose(0, 1).unsqueeze(2).contiguous())
                pretrain_res = attn_res.transpose(0, 1).contiguous().view(batch, seq_no * seq_len, attn_res.size(2))
                # batch, len, hidden
                not_pretrain_res = self.not_pretrain_emb(padded_not_pre.unsqueeze(2))
                final_val_res = torch.cat((pretrain_res, not_pretrain_res), 1)
                assert final_val_res.size(0) == reverse_idxs.size(0)
                # size tuple_len, batch, hidden
                final_val_res = torch.cat([res.index_select(0, ind).unsqueeze(0) for res, ind in zip(final_val_res, reverse_idxs)], 0).transpose(0, 1)
            else:
                final_val_res = None

            if self.enable_ent:
                ent_name, ent_data, ent_data_idx, ent_select_idx = ent_pretrain_input
                batch_size, ent_seq_no, ent_seq_len, _ = ent_name.size()
                ent_inp_src = ent_name.view(ent_name.size(0) * ent_name.size(1), ent_name.size(2), ent_name.size(3)).transpose(0, 1)
                ent_inp_src_data = ent_data.view(ent_data.size(0) * ent_data.size(1), ent_data.size(2), ent_data.size(3)).transpose(0, 1)

                # process ent_data_idx -> should be batch * ent_seq_no, ent_nums
                tmp_ent_data_idx = []
                for tmp_bid, tmp_batch in enumerate(ent_data_idx):
                    for tmp_each_seq in tmp_batch:
                        tmp_ent_data_idx.append(tmp_each_seq)
                    for tmp_pad in range(ent_seq_no - len(tmp_batch)):
                        tmp_ent_data_idx.append([])

                assert len(tmp_ent_data_idx) == batch_size * ent_seq_no
                # attn_res -> size (len, batch * seq_no, hidden)

                ent_attn_res, _ = self.ent_pretrain_model((ent_inp_src, ent_inp_src_data, tmp_ent_data_idx))
                batch_ent_attn_res = ent_attn_res.transpose(0, 1).contiguous().view(batch_size, ent_seq_no * ent_seq_len, ent_attn_res.size(2))
                #TODO get necessary emb for tmp_not_pretrain and use it
                ent_select_idx, tmp_not_pretrain = ent_select_idx
                ent_not_pretrain_res = self.ent_not_pretrain_emb(tmp_not_pretrain)
                assert ent_select_idx.size(0) == batch_ent_attn_res.size(0)
                assert tmp_not_pretrain.size(0) == batch_ent_attn_res.size(0)
                final_ent_res = torch.cat([tmp_ent_res.index_select(0, tmp_sel_idx).unsqueeze(0) for tmp_ent_res, tmp_sel_idx in zip(torch.cat((batch_ent_attn_res, ent_not_pretrain_res), 1), ent_select_idx)], 0).transpose(0, 1)
            else:
                final_ent_res = None

        else:
            final_val_res = pretrain_input
            final_ent_res = ent_pretrain_input

        if self.enable_val:
            assert final_val_res.size(0) == orig_src.size(0)
            assert final_val_res.size(1) == orig_src.size(1)
        if self.enable_ent:
            final_ent_res.size(0) == orig_src.size(0)
            final_ent_res.size(1) == orig_src.size(1)

        if self.emb_luts is None:
            assert not self.stage2
            if not final_ent_res:
                return final_val_res
            else:
                return final_val_res, final_ent_res
        else:
            if self.enable_val and self.enable_ent:
                return torch.cat((final_val_res, final_ent_res, self.emb_luts(orig_src[:, :, 2:])), 2)
            elif self.enable_val or self.stage2:
                # print(self.emb_luts)
                # print(orig_src.size())
                # print(orig_src.max())
                return torch.cat((final_val_res, self.emb_luts(orig_src[:, :, 1:])), 2)
            elif self.enable_ent:
                return torch.cat((self.ent_val_luts(orig_src[:, :, 0:1]), final_ent_res, self.emb_luts(orig_src[:, :, 2:])), 2)
            else:
                assert False
                

class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.

        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`

        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    `-feat_merge mlp`
        dropout (float): dropout probability.
    """
    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7, feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 emb_for_hier_hist=False, 
                 external_embedding=None,
                 use_pretrain=False,
                 pretrain_model=None,
                 not_pretrain_emb=None,
                 stage2_pretrain_input=False,
                 pos_type=None,
                 only_pos=False,
                 ent_pretrain_model=None,
                 ent_not_pretrain_emb=None):

        self.word_padding_idx = word_padding_idx
        self.only_pos = only_pos


        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        if external_embedding is not None:
            assert emb_for_hier_hist
            embeddings = external_embedding
        else:
            emb_params = zip(vocab_sizes, emb_dims, pad_indices)
            embeddings = [nn.Embedding(vocab, dim, padding_idx=pad)
                          for vocab, dim, pad in emb_params]
        self.store_emb_info = embeddings
        emb_luts = Elementwise(feat_merge, embeddings)



        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else max(word_vec_size, feat_vec_size))

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()

        # pretrain
        self.pretrain_model = pretrain_model
        self.ent_pretrain_model = ent_pretrain_model
        if use_pretrain or stage2_pretrain_input:
            self.pretrain_model_layer = UsePretrainModel(pretrain_model, not_pretrain_emb, feat_merge, embeddings, stage2=stage2_pretrain_input, ent_pretrain_model=ent_pretrain_model, ent_not_pretrain_emb=ent_not_pretrain_emb)
        else:
            self.pretrain_model_layer = None

        self.make_embedding = nn.Sequential()

        if emb_for_hier_hist:
            in_dim = sum(emb_dims)
            out_dim = max(word_vec_size, feat_vec_size)
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            if not only_pos:
                if use_pretrain or stage2_pretrain_input:
                    self.make_embedding.add_module('pretrain_module', self.pretrain_model_layer)
                else:
                    self.make_embedding.add_module('emb_luts', emb_luts)

                self.make_embedding.add_module('mlp', mlp)
        else:
            if not only_pos:
                if use_pretrain or stage2_pretrain_input:
                    self.make_embedding.add_module('pretrain_module', self.pretrain_model_layer)
                else:
                    self.make_embedding.add_module('emb_luts', emb_luts)

                if feat_merge == 'mlp' and len(feat_vocab_sizes)>0:
                    in_dim = sum(emb_dims)
                    out_dim = max(word_vec_size, feat_vec_size)

                    mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
                    self.make_embedding.add_module('mlp', mlp)

        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size, pos_type=pos_type, only_pe=only_pos)
            self.make_embedding.add_module('pe', pe)

    @property
    def word_lut(self):
        if isinstance(self.make_embedding[0], Elementwise):
            return self.make_embedding[0][0]
        else:
            return self.make_embedding[0].emb_luts[0]

    @property
    def emb_luts(self):
        if isinstance(self.make_embedding[0], Elementwise):
            return self.make_embedding[0]
        else:
            return self.make_embedding[0].emb_luts

    def load_pretrained_contextual_rep(self, model_file, fixed, ent_model_file, ent_fixed):
        if model_file:
            pretrained_model = torch.load(model_file,
                            map_location=lambda storage, loc: storage)
            self.pretrain_model.load_state_dict(pretrained_model['model'])
            if fixed:
                for param in self.pretrain_model.parameters():
                    param.requires_grad = False

        if ent_model_file:
            ent_pretrained_model = torch.load(ent_model_file,
                            map_location=lambda storage, loc: storage)
            self.ent_pretrain_model.load_state_dict(ent_pretrained_model['model'])
            if ent_fixed:
                for param in self.ent_pretrain_model.parameters():
                    param.requires_grad = False  
                    
    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    @property
    def get_feat_emb(self):
        return self.store_emb_info[1:]

    @property
    def get_val_emb(self):
        return self.store_emb_info[0]    

    def forward(self, input):
        """
        Computes the embeddings for words and features.

        Args:
            tuple of tensors: tensor `[len x batch x nfeat]`, ((char_len, batch*seq_len), batch*seq_len)
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        if isinstance(input, tuple):
            check_input = input[0]
        else:
            check_input = input
        in_length, in_batch, nfeat = check_input.size()
        if not self.only_pos:
            if self.pretrain_model_layer is not None:
                pass
                # aeq(nfeat - 1, len(self.emb_luts))
            else:
                aeq(nfeat, len(self.emb_luts))

        emb = self.make_embedding(input)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)

        return emb
