from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import onmt
import onmt.Loss
from onmt.Utils import aeq
from torch.distributions import Categorical
# from onmt.modules import TransformerEncoder
# import onmt.modules.Transformer.PositionwiseFeedForward as 
# from onmt.modules.Transformer import PositionwiseFeedForward
POINTER_PAD_IDX=1

class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = onmt.modules.LayerNorm(size)
        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
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
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        if isinstance(input, tuple):
            input = input[0]
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            encoder_state (rnn-class specific):
               initial encoder_state state.

        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError

class dumpEmb(object):
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

class PositionEmb(nn.Module):
    def __init__(self, windows, hidden, dropout, nohistpos):
        super(PositionEmb, self).__init__()
        self.pos_emb = nn.Parameter(torch.FloatTensor(windows, 1, hidden))
        print("pos emb size is {}".format(self.pos_emb.size()))
        self.dropout = nn.Dropout(p=dropout)
        self.nohistpos = nohistpos
        print("status for nohistpos: {}".format(self.nohistpos))

    # x has size of window_size, batch, hidden
    def forward(self, x):
        if self.nohistpos:
            return x
        else:
            return self.dropout(x + self.pos_emb.expand(-1, x.size(1), -1))



class MultiLayerMHA(nn.Module):
    def __init__(self, n_layer, head_count, size, dropout):
        super(MultiLayerMHA, self).__init__()
        self.n_layer = n_layer
        self.models = nn.ModuleList([onmt.modules.MultiHeadedAttention(
            head_count, size, dropout=dropout) for i in range(n_layer)])
        self.feed_forward = PositionwiseFeedForward(size,
                                                    2048,
                                                    dropout)
        self.layer_norm = onmt.modules.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory_bank):
        for i in range(self.n_layer):
            memory_bank_norm = self.layer_norm(memory_bank.contiguous())
            tmp, _ = self.models[i](memory_bank_norm, memory_bank_norm, memory_bank_norm)
            memory_bank = self.dropout(tmp) + memory_bank
            memory_bank = self.feed_forward(memory_bank)

            # memory_bank = memory_bank.transpose(0, 1)

        return self.layer_norm(memory_bank), None

class MultiLayerSelfAttention(nn.Module):
    def __init__(self, n_layer, emb_size, attn_type, attn_hidden, only_ply_level_gate, residual, no_gate_relu, no_gate_bias, dropout):
        super(MultiLayerSelfAttention, self).__init__()
        self.n_layer = n_layer
        self.models = nn.ModuleList([onmt.modules.GlobalSelfAttention(emb_size, coverage=False, attn_type=attn_type, attn_hidden=attn_hidden, no_gate=only_ply_level_gate, residual=residual, no_gate_relu=no_gate_relu, no_gate_bias=no_gate_bias) for i in range(n_layer)])
        self.feed_forward = PositionwiseFeedForward(emb_size,
                                                    2048,
                                                    dropout)
        self.layer_norm = onmt.modules.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, memory_bank, memory_lengths=None, coverage=None):
        for i in range(self.n_layer):
            memory_bank_norm = self.layer_norm(memory_bank.contiguous())
            tmp, _ = self.models[i](memory_bank_norm, memory_bank_norm, memory_lengths=memory_lengths)
            memory_bank = self.dropout(tmp) + memory_bank.transpose(0, 1)
            memory_bank = self.feed_forward(memory_bank)

            memory_bank = memory_bank.transpose(0, 1)

        return self.layer_norm(memory_bank).transpose(0, 1).contiguous(), None

class MultiLayerCNN(nn.Module):
    """
    Encoder built on CNN based on
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """
    def __init__(self, num_layers, hidden_size,
                 cnn_kernel_width, dropout, input_size):
        super(MultiLayerCNN, self).__init__()

        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = onmt.modules.Conv2Conv.StackedCNN(num_layers, hidden_size,
                              cnn_kernel_width, dropout)

    def shape_transform(self, x):
        """ Tranform the size of the tensors to fit for conv input. """
        return torch.unsqueeze(torch.transpose(x, 1, 2), 3)

    def forward(self, emb, lengths=None, hidden=None):
        """ See :obj:`onmt.modules.EncoderBase.forward()`"""
        # self._check_args(input, lengths, hidden)

        s_len, batch, emb_dim = emb.size()

        emb = emb.transpose(0, 1).contiguous()
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = self.shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return emb_remap.squeeze(3).transpose(0, 1).contiguous(),\
            out.squeeze(3).transpose(0, 1).transpose(0, 2).contiguous()

class PretrainModel(EncoderBase):
    def __init__(self, embeddings, num_layers, d_model, heads, d_ff, dropout, max_relative_positions, categories, loss_type, pre_use_mlp, pretrain_emb_ent, pre_ent_type, pre_ent_use_data):
        super(PretrainModel, self).__init__()

        self.pre_ent = pretrain_emb_ent
        self.pre_ent_type = pre_ent_type
        self.ent_use_data = pretrain_emb_ent and pre_ent_use_data
        if pretrain_emb_ent and pre_ent_use_data:
            self.embeddings, self.data_embeddings = embeddings
        else:
            self.embeddings = embeddings
        if pre_use_mlp:
            self.mlp_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        else:
            self.transformer = onmt.modules.Transformer.TransformerEncoder(num_layers, d_model, heads, d_ff, dropout, self.embeddings, max_relative_positions, pretrain_emb_ent and pre_ent_use_data)
        self.dropout = nn.Dropout(p=dropout)

        self.categories = categories
        self.pre_use_mlp = pre_use_mlp

        if pretrain_emb_ent:
            self.loss_type = loss_type
            if pre_ent_type == "rank":
                self.ranking_tgt_dim = 1
                self.inf_layer = nn.Sequential(
                        nn.Linear(d_model, self.ranking_tgt_dim, bias=False)
                        )
                self.inf_softmax = nn.Softmax(-1)  
            elif pre_ent_type == "bin":   
                self.ranking_tgt_dim = 1
                self.inf_layer = nn.Sequential(
                        nn.Linear(d_model, self.ranking_tgt_dim, bias=False),
                        nn.Sigmoid()
                    )
            else:
                assert False         
        else:
            self.loss_type = loss_type
            if loss_type == "nll":
                assert categories == 3 or categories == 2
                self.inf_layer = nn.Sequential(
                        nn.Linear(2 * d_model, categories, bias=False),
                        nn.LogSoftmax(-1)
                    )
            elif loss_type == "ranking":
                self.ranking_tgt_dim = 1
                self.inf_layer = nn.Sequential(
                        nn.Linear(d_model, self.ranking_tgt_dim, bias=False)
                        )
                self.inf_softmax = nn.Softmax(-1)

    def forward(self, src, lengths=None, encoder_state=None, memory_lengths=None, require_emb=False):
        # if ent & data -> src = (srcs, datas, idxs)
        # if isinstance(src, tuple) and len(src) == 2:
        #     src, _ = src
        # else:
        #     assert False, (isinstance(src, tuple), len(src))

        self._check_args(src, lengths, encoder_state)

        if self.ent_use_data:
            src, src_data, src_data_idx = src

        # src size: (len, batch, nfeat)
        # attn_res size: (len, batch, hidden)
        if self.pre_use_mlp:
            emb_res = self.mlp_layer(self.embeddings(src))
            attn_res = emb_res
            assert False
        else:
            if self.ent_use_data:
                src = self.embeddings(src)
                src_data = self.data_embeddings(src_data)
                tmp_batches_result = []
                for bidxid, batch_idx in enumerate(src_data_idx):
                    tmp_elem_result = []
                    for elemid, each_elem in enumerate(batch_idx):
                        tmp_elem_result.append(src_data[:, bidxid:bidxid+1, :].narrow(0, each_elem[0], each_elem[1]).mean(0, keepdim=True))
 
                    if len(tmp_elem_result) < src.size(0):
                        for tmp_num in range(src.size(0) - len(tmp_elem_result)):
                            tmp_elem_result.append(torch.FloatTensor(1, 1, src_data.size(2)).fill_(0).to(src_data.device))

                    tmp_batches_result.append(torch.cat(tmp_elem_result, 0))
                tmp_batches_result = torch.cat(tmp_batches_result, 1)
                src = src + tmp_batches_result
            emb_res, attn_res, lengths = self.transformer(src, lengths)

        batch_first_attn_res = attn_res.transpose(0, 1)
        comp_first = batch_first_attn_res.unsqueeze(2).expand(-1, -1, batch_first_attn_res.size(1), -1)
        comp_second = batch_first_attn_res.repeat(1, batch_first_attn_res.size(1), 1).view(batch_first_attn_res.size(0), batch_first_attn_res.size(1), batch_first_attn_res.size(1), batch_first_attn_res.size(2))
        comp_input = torch.cat((comp_first, comp_second), 3)
        # size: -> batch, len*len, prob -> len*len, batch, prob
        if self.pre_ent and self.pre_ent_type == "bin":
            inf_result = self.inf_layer(attn_res)
        elif self.loss_type == "ranking" or (self.pre_ent and self.pre_ent_type == "rank"):
            inf_result = self.inf_softmax(torch.cat((self.inf_layer(comp_first), self.inf_layer(comp_second)), 3)).view(comp_first.size(0), comp_first.size(1) * comp_first.size(1), self.ranking_tgt_dim * 2).transpose(0, 1)

        elif self.loss_type == "nll":
            inf_result = self.inf_layer(comp_input).view(comp_input.size(0), comp_input.size(1) * comp_input.size(1), self.categories).transpose(0, 1)
        
        return attn_res, inf_result


class HierarchicalEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, meta, num_layers, embeddings, emb_size, attn_hidden, rnn_type, bidirectional, hidden_size, dropout=0.0, attn_type="general", coverage_attn=False, two_dim_record=False, row_self_attn_type="normal", col_self_attn_type="normal", multi_head_count=None, multi_head_dp=None, hier_history=False, hier_history_seq_type=None, hier_history_seq_window=None, hier_bi=False, hier_num_layers=None, hier_hist_attn_type=None, hier_hist_attn_pos_type=None, norow=False, nocolumn=False, nofusion=False, nohierstructure=False, nohistpos=False, is_cnn_cell=False, cnn_kernel_width=None, mlp_hier=False, two_mlp_hier=False, ply_level_gate=False, two_dim_score=None):
        super(HierarchicalEncoder, self).__init__()

        self.mlp_hier = mlp_hier
        self.two_mlp_hier = two_mlp_hier
        if self.two_mlp_hier:
            self.two_mlp_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        else:
            self.two_mlp_layer = None
        self.isCNN = is_cnn_cell
        self.cnn_kernel_width = cnn_kernel_width

        self.hier_history = hier_history
        self.hier_history_seq_type = hier_history_seq_type
        self.hier_history_seq_window = hier_history_seq_window
        self.hier_bi = hier_bi
        self.hier_num_layers = hier_num_layers
        self.hier_hist_attn_type = hier_hist_attn_type
        self.nohierstructure = nohierstructure
        # self.noallthreedimrep = noallthreedimrep
        print("status for no hier strc: {}".format(self.nohierstructure))
        # print("status for no all three dim: {}".format(self.noallthreedimrep))

        self.norow = norow
        self.nocolumn = nocolumn
        self.nofusion = nofusion

        print("status for hier seq history | hier_history {}, hier_history_seq_type {}, hier_history_seq_window {}, hier_bi {}, hier_num_layers {}, hier_hist_attn_type {}, hier_hist_attn_pos_type {}".format(self.hier_history, self.hier_history_seq_type, self.hier_history_seq_window, self.hier_bi, self.hier_num_layers, self.hier_hist_attn_type, hier_hist_attn_pos_type))

        if self.hier_history:
            if hier_history_seq_type in ["rnn", "RNN"]:
                self.hier_history_seq_rnn_layer = RNNEncoder(rnn_type, hier_bi, hier_num_layers, hidden_size, dropout, dumpEmb(emb_size)) 
            elif hier_history_seq_type in ["sa", "SA"]:
                # if nohistpos:
                #     self.hier_hist_attn_pos_layer = None
                if hier_hist_attn_pos_type == "posEmb":
                    self.hier_hist_attn_pos_layer = PositionEmb(self.hier_history_seq_window, hidden_size, dropout, nohistpos)
                elif hier_hist_attn_pos_type == "posEncoding":
                    self.hier_hist_attn_pos_layer = PositionalEncoding(dropout, emb_size, nohistpos)
                else:
                    raise ValueError("invalid hier_hist_attn_pos_type")

                self.hier_history_seq_attn_layer = onmt.modules.GlobalSelfAttention(emb_size, coverage=False, attn_type=hier_hist_attn_type, attn_hidden=attn_hidden, no_gate=True, residual=False, no_gate_relu=False, no_gate_bias=True)
            else:
                raise ValueError("invalid hier_history_seq_type")

        self.two_dim_score = two_dim_score

        if two_dim_record:
            self.two_dim_gen_layer = nn.Sequential()
            if self.hier_history:
                if self.norow or self.nocolumn:
                    two_dim_gen_layer_input_dim = hidden_size*2
                else:
                    two_dim_gen_layer_input_dim = hidden_size*3
            else:
                two_dim_gen_layer_input_dim = hidden_size*2
            self.two_dim_gen_layer.add_module('linear_transform', nn.Linear(two_dim_gen_layer_input_dim, hidden_size, bias=False))
            
            self.two_dim_gen_layer.add_module('tanh', nn.Tanh())

            assert two_dim_score is not None and two_dim_score in ["mlp", "general", "dot"]
            if not self.nofusion:
                if two_dim_score == "mlp":
                    self.two_dim_mlp_score_layer = nn.Sequential(
                        nn.Linear(hidden_size*2, hidden_size, bias=False),
                        nn.Tanh(),
                        nn.Linear(hidden_size, 1, bias=False)
                        )
                elif two_dim_score == "general":
                    self.two_dim_general_score_layer = nn.Linear(hidden_size, hidden_size, bias=False)

        self.num_layers = num_layers

        if isinstance(embeddings, tuple):
            assert hier_history and len(embeddings) == 2
            self.embeddings = embeddings[0]
            self.hist_embeddings = embeddings[1]
        else:
            self.embeddings = embeddings
            self.hist_embeddings = None

        self.meta = meta
        self.dropout = nn.Dropout(p=dropout)
        bidirectional = bidirectional if bidirectional is not None else False
        self.bidirectional = bidirectional

        assert row_self_attn_type in ["normal", "multi-head-attn", None] and col_self_attn_type in ["normal", "multi-head-attn", None]
        self.row_multi_head_attn = (row_self_attn_type == "multi-head-attn")
        self.col_multi_head_attn = (col_self_attn_type == "multi-head-attn")
        
        self.hidden_size = hidden_size

        if not self.norow:
            if self.mlp_hier:
                self.row_rnn = None
            elif self.isCNN:
                self.row_rnn = MultiLayerCNN(self.hier_num_layers, hidden_size, self.cnn_kernel_width, dropout, emb_size)
            else:
                assert hidden_size == emb_size
                if row_self_attn_type == "multi-head-attn":
                    assert multi_head_count is not None and isinstance(multi_head_count, int)
                    # self.row_rnn = onmt.modules.MultiHeadedAttention(multi_head_count, emb_size, dropout if multi_head_dp is None else multi_head_dp, mha_concat, mha_residual, mha_norm)
                    self.row_rnn = MultiLayerMHA(self.hier_num_layers, multi_head_count, emb_size, dropout if multi_head_dp is None else multi_head_dp)
                elif row_self_attn_type == "normal":
                    if self.hier_num_layers and self.hier_num_layers > 1:
                        self.row_rnn = MultiLayerSelfAttention(self.hier_num_layers, emb_size, attn_type, attn_hidden, True, False, False, True, dropout)
                    else:
                        self.row_rnn = onmt.modules.GlobalSelfAttention(emb_size, coverage=coverage_attn, attn_type=attn_type, attn_hidden=attn_hidden, no_gate=True, residual=False, no_gate_relu=False, no_gate_bias=True)

        if self.isCNN:
            self.row_attn = MultiLayerCNN(self.hier_num_layers, hidden_size, self.cnn_kernel_width, dropout, emb_size)
        elif row_self_attn_type == "multi-head-attn":
            assert multi_head_count is not None and isinstance(multi_head_count, int)
            # self.row_attn = onmt.modules.MultiHeadedAttention(multi_head_count, hidden_size, dropout if multi_head_dp is None else multi_head_dp, mha_concat, mha_residual, mha_norm)
            self.row_attn = MultiLayerMHA(self.hier_num_layers, multi_head_count, hidden_size, dropout if multi_head_dp is None else multi_head_dp)
        elif row_self_attn_type == "normal":
            self.row_attn = onmt.modules.GlobalSelfAttention(hidden_size, coverage=coverage_attn, attn_type=attn_type, attn_hidden=attn_hidden, no_gate=False if ply_level_gate else True, residual=False, no_gate_relu=False, no_gate_bias=True)

        # col dim record
        self.two_dim_record = two_dim_record
        self.col_sigmoid = self.two_dim_mlp = None

        if two_dim_record:
            if not self.nocolumn:
                if col_self_attn_type == "multi-head-attn":
                    assert multi_head_count is not None and isinstance(multi_head_count, int)
                    self.col_attn = MultiLayerMHA(self.hier_num_layers, multi_head_count, emb_size, dropout if multi_head_dp is None else multi_head_dp)
                elif col_self_attn_type == "normal":
                    self.col_attn = onmt.modules.GlobalSelfAttention(emb_size, coverage=coverage_attn, attn_type=attn_type, attn_hidden=attn_hidden, no_gate=True, residual=False, no_gate_relu=False, no_gate_bias=True)

            self.col_sigmoid = nn.Sequential(
                    nn.Linear(hidden_size+emb_size, hidden_size),
                    nn.Sigmoid()
                  )
        else:
            self.col_attn = None

        self.register_buffer("row_attn_length", torch.LongTensor([int(self.meta['home_ply_num']) + int(self.meta['vis_ply_num']) + int(self.meta['team_num'])]))
        self.register_buffer("home_ply_length", torch.LongTensor([int(self.meta['home_ply_num'])]))
        self.register_buffer("vis_ply_length", torch.LongTensor([int(self.meta['vis_ply_num'])]))
        self.register_buffer("team_length", torch.LongTensor([int(self.meta['team_num'])]))

        self.register_buffer("home_ply_num_kw_num", torch.LongTensor([int(self.meta["home_ply_kw_num"])]))
        self.register_buffer("vis_ply_num_kw_num", torch.LongTensor([int(self.meta["vis_ply_kw_num"])]))
        self.register_buffer("team_kw_num", torch.LongTensor([int(self.meta["team_kw_num"])]))

        if self.hier_history_seq_window is not None:
            self.register_buffer("hist_seq_window", torch.LongTensor([int(self.hier_history_seq_window)]))

    def obtainLastLayer(self, rep, batch_size, hier_hist_seq=False):
        if hier_hist_seq:
            length = 2 if self.hier_bi else 1
        else:
            length = 2 if self.bidirectional else 1
        rep = rep.narrow(0, rep.size(0)-length, length).transpose(0, 1).contiguous().view(rep.size(1), -1)
        rep = rep.view(batch_size, -1, rep.size(1)).transpose(0, 1).contiguous()
        return rep

    def obtainHierRep(self, emb, start_id, end_id, num, num_tensor, kw_num, kw_num_tensor, batch, hist_emb, hist_start_id, hist_end_id):
        home_ply_rep = emb.narrow(0, start_id, (end_id - start_id + 1))

        # size are (kw_num, batch*ply_num, hidden)
        home_ply_rep = home_ply_rep.transpose(0, 1).contiguous().view(home_ply_rep.size(1)*int(num), kw_num, home_ply_rep.size(2)).transpose(0, 1)

        if self.hier_history:
            # place need to change
            home_ply_rep_hist = hist_emb.narrow(0, hist_start_id, (hist_end_id - hist_start_id + 1))
            assert home_ply_rep_hist.size(1) == batch

            # size is windows, batch*ply_num*ply_keys, hidden
            home_ply_rep_hist = home_ply_rep_hist.transpose(0, 1).contiguous().view(-1, self.hier_history_seq_window, home_ply_rep_hist.size(2)).transpose(0, 1).contiguous()
            if self.hier_history_seq_type == "rnn" or self.hier_history_seq_type == "RNN":
                (home_ply_rep_hist, _), _ = self.hier_history_seq_rnn_layer(home_ply_rep_hist)
                home_ply_rep_hist = self.obtainLastLayer(home_ply_rep_hist, home_ply_rep.size(1), True)
            else:
                home_ply_rep_hist, _ = self.hier_history_seq_attn_layer(self.hier_hist_attn_pos_layer(home_ply_rep_hist).transpose(0, 1).contiguous(), self.hier_hist_attn_pos_layer(home_ply_rep_hist).transpose(0, 1), memory_lengths=self.hist_seq_window.expand(home_ply_rep_hist.size(1)))
                home_ply_rep_hist = home_ply_rep_hist.narrow(0, home_ply_rep_hist.size(0)-1, 1).squeeze(0)
                assert len(home_ply_rep_hist.size()) == 2

                home_ply_rep_hist = home_ply_rep_hist.view(home_ply_rep.size(1), -1, home_ply_rep_hist.size(1)).transpose(0, 1).contiguous()
            assert home_ply_rep_hist.size(2) == home_ply_rep.size(2) and home_ply_rep_hist.size(1) == home_ply_rep.size(1) and home_ply_rep_hist.size(0) == home_ply_rep.size(0)


        else:
            home_ply_rep_hist = None

        if not self.norow:
            if self.mlp_hier:
                home_ply_memory_bank = home_ply_rep
                home_ply_row_rep = home_ply_memory_bank.mean(0).view(batch, -1, home_ply_memory_bank.size(2)).transpose(0, 1).contiguous()
                if self.two_mlp_hier:
                    home_ply_row_rep = self.two_mlp_layer(home_ply_row_rep)

            elif self.isCNN:
                # print("homeplyrep")
                # print(home_ply_rep.size())
                _, home_ply_memory_bank = self.row_rnn(home_ply_rep)
                # print("homeplymembank")
                # print(home_ply_memory_bank.size())
            elif self.row_multi_head_attn:
                # home_ply_memory_bank, _ = self.row_rnn(home_ply_rep.transpose(0, 1), home_ply_rep.transpose(0, 1), home_ply_rep.transpose(0, 1))
                home_ply_memory_bank, _ = self.row_rnn(home_ply_rep.transpose(0, 1))
                home_ply_memory_bank = home_ply_memory_bank.transpose(0, 1).contiguous()
            else:
                home_ply_memory_bank, _ = self.row_rnn(home_ply_rep.transpose(0, 1).contiguous(),
                    home_ply_rep.transpose(0, 1),
                    memory_lengths=kw_num_tensor.expand(home_ply_rep.size(1)))

            home_ply_row_rep = home_ply_memory_bank.mean(0).view(batch, -1, home_ply_memory_bank.size(2)).transpose(0, 1).contiguous()

        else:
            home_ply_memory_bank = None

        # if not self.two_dim_record and not self.true_two_dim_fusion:

        if self.two_dim_record:
            home_ply_memory_bank = self.getTwoDimRep(home_ply_memory_bank, home_ply_rep, batch, num_tensor, False, False, home_ply_rep_hist)

        home_ply_row_rep = home_ply_memory_bank.mean(0).view(batch, -1, home_ply_memory_bank.size(2)).transpose(0, 1).contiguous()

        return home_ply_row_rep, home_ply_memory_bank

    # both rnn_output, memory_bank size are (kw_num, batch*ply_num, hidden)
    def getTwoDimRep(self, rnn_output, memory_bank, batch_size, memory_lengths, use_mlp, direct_concat, home_ply_rep_hist):
        # size is (ply_num, batch, kw_num, hidden)
        if not self.nocolumn:
            col_memory_bank = memory_bank.contiguous().view(memory_bank.size(0), batch_size, -1, memory_bank.size(2)).transpose(0, 2).contiguous()
            # size is (ply_num, batch*kw_num, hidden)
            col_memory_bank = col_memory_bank.view(col_memory_bank.size(0), -1, col_memory_bank.size(3))
            # size is [ply_num x batch*kw_num x dim]
            if self.col_multi_head_attn:
                col_rep, _ = self.col_attn(col_memory_bank.transpose(0, 1), col_memory_bank.transpose(0, 1), col_memory_bank.transpose(0, 1))
                col_rep = col_rep.transpose(0, 1).contiguous()
            else:
                col_rep, _ = self.col_attn(col_memory_bank.transpose(0, 1).contiguous(), col_memory_bank.transpose(0, 1), memory_lengths=memory_lengths.expand(col_memory_bank.size(1)))

            col_rep = col_rep.view(col_rep.size(0), batch_size, -1, col_rep.size(2)).transpose(0, 2).contiguous()
            # size is (kw_num, batch*ply_num, dim)
            col_rep = col_rep.view(col_rep.size(0), -1, col_rep.size(3))

        if self.hier_history:
            if self.norow:
                gen_rep = self.two_dim_gen_layer(torch.cat((col_rep, home_ply_rep_hist), 2))
                row_dim1, row_dim2, row_dim3 = col_rep.size()
            elif self.nocolumn:
                gen_rep = self.two_dim_gen_layer(torch.cat((rnn_output, home_ply_rep_hist), 2))
                row_dim1, row_dim2, row_dim3 = rnn_output.size()
            else:
                gen_rep = self.two_dim_gen_layer(torch.cat((rnn_output, col_rep, home_ply_rep_hist), 2))
                row_dim1, row_dim2, row_dim3 = rnn_output.size()
        else:
            if self.norow:
                gen_rep = self.two_dim_gen_layer(torch.cat((col_rep), 2))
                row_dim1, row_dim2, row_dim3 = col_rep.size()
            elif self.nocolumn:
                gen_rep = self.two_dim_gen_layer(torch.cat((rnn_output), 2))
                row_dim1, row_dim2, row_dim3 = rnn_output.size()
            else:
                gen_rep = self.two_dim_gen_layer(torch.cat((rnn_output, col_rep), 2))
                row_dim1, row_dim2, row_dim3 = rnn_output.size()

        if self.nofusion:
            return gen_rep
        else:
            if self.two_dim_score == "mlp":
                if not self.norow:
                    row_score = self.two_dim_mlp_score_layer(torch.cat((rnn_output, gen_rep), 2))
                else:
                    row_score = None
                if not self.nocolumn:
                    col_score = self.two_dim_mlp_score_layer(torch.cat((col_rep, gen_rep), 2))
                else:
                    col_score = None
                if self.hier_history:
                    hist_score = self.two_dim_mlp_score_layer(torch.cat((home_ply_rep_hist, gen_rep), 2))
                else:
                    hist_score = None
            elif self.two_dim_score == "general":
                if not self.norow:
                    row_score = torch.bmm(
                        rnn_output.view(-1, row_dim3).unsqueeze(1),
                        self.two_dim_general_score_layer(gen_rep).view(-1, row_dim3).unsqueeze(2)
                        ).view(row_dim1, row_dim2, 1)
                else:
                    row_score = None
                if not self.nocolumn:
                    col_score = torch.bmm(
                        col_rep.view(-1, row_dim3).unsqueeze(1),
                        self.two_dim_general_score_layer(gen_rep).view(-1, row_dim3).unsqueeze(2)
                        ).view(row_dim1, row_dim2, 1)
                else:
                    col_score = None
                if self.hier_history:
                    hist_score = torch.bmm(
                        home_ply_rep_hist.view(-1, row_dim3).unsqueeze(1),
                        self.two_dim_general_score_layer(gen_rep).view(-1, row_dim3).unsqueeze(2)
                        ).view(row_dim1, row_dim2, 1)
                else:
                    hist_score = None
            elif self.two_dim_score == "dot":
                if not self.norow:
                    row_score = torch.bmm(
                        rnn_output.view(-1, row_dim3).unsqueeze(1),
                        gen_rep.view(-1, row_dim3).unsqueeze(2)
                        ).view(row_dim1, row_dim2, 1)
                else:
                    row_score = None
                if not self.nocolumn:
                    col_score = torch.bmm(
                        col_rep.view(-1, row_dim3).unsqueeze(1),
                        gen_rep.view(-1, row_dim3).unsqueeze(2)
                        ).view(row_dim1, row_dim2, 1)
                else:
                    col_score = None
                if self.hier_history:
                    hist_score = torch.bmm(
                        home_ply_rep_hist.view(-1, row_dim3).unsqueeze(1),
                        gen_rep.view(-1, row_dim3).unsqueeze(2)
                        ).view(row_dim1, row_dim2, 1)
                else:
                    hist_score = None

            if self.hier_history:
                if not self.norow and not self.nocolumn:
                    two_dim_weight = F.softmax(torch.cat((row_score, col_score, hist_score), 2), dim=2)
                    rep_concat = torch.cat((rnn_output.unsqueeze(2), col_rep.unsqueeze(2), home_ply_rep_hist.unsqueeze(2)), 2)
                    assert rep_concat.size(2) == 3 and rep_concat.size(3) == row_dim3
                    rep_view = rep_concat.view(-1, 3, row_dim3)
                    assert two_dim_weight.size(2) == 3
                    score_view = two_dim_weight.view(-1, 3).unsqueeze(1)
                elif self.norow:
                    two_dim_weight = F.softmax(torch.cat((col_score, hist_score), 2), dim=2)
                    rep_concat = torch.cat((col_rep.unsqueeze(2), home_ply_rep_hist.unsqueeze(2)), 2)
                    assert rep_concat.size(2) == 2 and rep_concat.size(3) == row_dim3
                    rep_view = rep_concat.view(-1, 2, row_dim3)
                    assert two_dim_weight.size(2) == 2
                    score_view = two_dim_weight.view(-1, 2).unsqueeze(1) 
                elif self.nocolumn:
                    two_dim_weight = F.softmax(torch.cat((row_score, hist_score), 2), dim=2)
                    rep_concat = torch.cat((rnn_output.unsqueeze(2), home_ply_rep_hist.unsqueeze(2)), 2)
                    assert rep_concat.size(2) == 2 and rep_concat.size(3) == row_dim3
                    rep_view = rep_concat.view(-1, 2, row_dim3)
                    assert two_dim_weight.size(2) == 2
                    score_view = two_dim_weight.view(-1, 2).unsqueeze(1)
            else:
                assert not self.norow and not self.nocolumn
                two_dim_weight = F.softmax(torch.cat((row_score, col_score), 2), dim=2)
                rep_concat = torch.cat((rnn_output.unsqueeze(2), col_rep.unsqueeze(2)), 2)
                assert rep_concat.size(2) == 2 and rep_concat.size(3) == row_dim3
                rep_view = rep_concat.view(-1, 2, row_dim3)
                assert two_dim_weight.size(2) == 2
                score_view = two_dim_weight.view(-1, 2).unsqueeze(1)

            return torch.bmm(score_view, rep_view).squeeze(1).view(row_dim1, row_dim2, row_dim3)

    def forward(self, src, lengths=None, encoder_state=None, memory_lengths=None, require_emb=False):
        "See :obj:`EncoderBase.forward()`"

        if isinstance(src, tuple) and len(src) == 2:
            src, hist_src = src
        else:
            assert False, (isinstance(src, tuple), len(src))
            hist_src = None
            
        self._check_args(src, lengths, encoder_state)

        if self.hier_history:
            assert hist_src is not None
            hist_emb = self.dropout(self.hist_embeddings(hist_src))
            assert (self.hier_history_seq_window * (self.meta["tuple_num"] - (self.meta["special_end"] - self.meta["special_start"] + 1)) + (self.meta["special_end"] - self.meta["special_start"] + 1)) == hist_emb.size(0)
        else:
            hist_emb = None

        emb = self.dropout(self.embeddings(src))

        s_len, batch, emb_dim = emb.size()

        # obtain record level representation
        assert s_len == self.meta['tuple_num']
        special_rep = emb.narrow(0, self.meta['special_start'], (self.meta['special_end'] - self.meta['special_start'] + 1))

        if self.hier_history:
            hist_home_ply_start = self.meta['home_ply_start']
            hist_home_ply_end = hist_home_ply_start + ((self.meta['home_ply_end'] - self.meta['home_ply_start'] + 1) * self.hier_history_seq_window) - 1
        else:
            hist_home_ply_start = hist_home_ply_end = None

        home_ply_row_rep, home_ply_memory_bank = self.obtainHierRep(emb, self.meta['home_ply_start'], self.meta['home_ply_end'], self.meta['home_ply_num'], self.home_ply_length, self.meta['home_ply_kw_num'], self.home_ply_num_kw_num, batch, hist_emb, hist_home_ply_start, hist_home_ply_end)

        if self.hier_history:
            hist_vis_ply_start = hist_home_ply_end + 1
            hist_vis_ply_end = hist_vis_ply_start + ((self.meta['vis_ply_end'] - self.meta['vis_ply_start'] + 1) * self.hier_history_seq_window) - 1
        else:
            hist_vis_ply_start = hist_vis_ply_end = None

        vis_ply_row_rep, vis_ply_memory_bank = self.obtainHierRep(emb, self.meta['vis_ply_start'], self.meta['vis_ply_end'], self.meta['vis_ply_num'], self.vis_ply_length, self.meta['vis_ply_kw_num'], self.vis_ply_num_kw_num, batch, hist_emb, hist_vis_ply_start, hist_vis_ply_end)


        if self.hier_history:
            hist_team_start = hist_vis_ply_end + 1
            hist_team_end = hist_team_start + ((self.meta['team_end'] - self.meta['team_start'] + 1) * self.hier_history_seq_window) - 1
        else:
            hist_team_start = hist_team_end = None

        if self.hier_history:
            assert hist_team_end == hist_emb.size(0) - 1

        team_row_rep, team_memory_bank = self.obtainHierRep(emb, self.meta['team_start'], self.meta['team_end'], self.meta['team_num'], self.team_length, self.meta['team_kw_num'], self.team_kw_num, batch, hist_emb, hist_team_start, hist_team_end)
      
        total_row_rep = torch.cat((home_ply_row_rep, vis_ply_row_rep, team_row_rep), 0)

        if self.isCNN:
            _, total_row_rep = self.row_attn(total_row_rep)
        else:
            if self.row_multi_head_attn:
                # total_row_rep, _ = self.row_attn(total_row_rep.transpose(0, 1), total_row_rep.transpose(0, 1), total_row_rep.transpose(0, 1))
                total_row_rep, _ = self.row_attn(total_row_rep.transpose(0, 1))
                total_row_rep = total_row_rep.transpose(0, 1).contiguous()
            else:
                total_row_rep, _ = self.row_attn(total_row_rep.transpose(0, 1).contiguous(), total_row_rep.transpose(0, 1), memory_lengths=self.row_attn_length.expand(batch))


        home_ply_row_rep = total_row_rep.narrow(0, 0, home_ply_row_rep.size(0))
        vis_ply_row_rep = total_row_rep.narrow(0, home_ply_row_rep.size(0), vis_ply_row_rep.size(0))
        team_row_rep = total_row_rep.narrow(0, home_ply_row_rep.size(0) + vis_ply_row_rep.size(0), team_row_rep.size(0))

        mean = total_row_rep.mean(0).expand(self.num_layers, batch, self.hidden_size)

        # print(special_rep.size(), home_ply_memory_bank.size(), vis_ply_memory_bank.size(), team_memory_bank.size())

        mem_bank_as_orig = torch.cat((special_rep, torch.cat([tmp_i.transpose(0, 1).contiguous().view(batch, -1, tmp_i.size(2)).transpose(0, 1) for tmp_i in (home_ply_memory_bank, vis_ply_memory_bank, team_memory_bank)], 0)), 0)

        memory_bank = ((special_rep, home_ply_memory_bank, vis_ply_memory_bank, team_memory_bank), (special_rep,home_ply_row_rep, vis_ply_row_rep, team_row_rep), mem_bank_as_orig)
        if self.nohierstructure:
            mean = mem_bank_as_orig.mean(0).expand(self.num_layers, batch, self.hidden_size)
            memory_bank = (None, None, memory_bank[2])
        encoder_final = (mean, mean)
        return encoder_final, memory_bank

class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, meta, num_layers, embeddings, emb_size, attn_hidden, dropout=0.0, attn_type="general",
                 coverage_attn=False, enable_attn=True, no_gate=False, residual=False, no_gate_relu=False, no_gate_bias=False, mode="MLP", trans_layer=None, rnn_type=None, bidirectional=False, hidden_size=None, cnn_kernel_width=None, multi_head_count=None, multi_head_dp=None):
        super(MeanEncoder, self).__init__()
        self.meta = meta
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=dropout)
        assert mode in ["MLP", "LSTM", "CNN", "SA", "MHSA"]
        self.enc_mode = mode
        if mode == "LSTM":
            self.transform_layer = RNNEncoder(rnn_type, bidirectional, trans_layer, hidden_size, dropout, dumpEmb(emb_size))
        elif mode == "CNN":
            self.transform_layer = MultiLayerCNN(trans_layer, hidden_size, cnn_kernel_width, dropout, emb_size)
        elif mode == "SA":
            self.transform_layer = MultiLayerSelfAttention(trans_layer, emb_size, attn_type, attn_hidden, no_gate, True, residual, no_gate_relu, no_gate_bias, dropout)
        elif mode == "MHSA":
            self.transform_layer = MultiLayerMHA(trans_layer, multi_head_count, emb_size, dropout if multi_head_dp is None else multi_head_dp)
        else:
            self.transform_layer = None

        if self.meta:
            self.register_buffer("record_num_tensor", torch.LongTensor([int(self.meta["tuple_num"])]))

        if enable_attn:
            self.attn = onmt.modules.GlobalSelfAttention(emb_size, coverage=coverage_attn, attn_type=attn_type, attn_hidden=attn_hidden, no_gate=no_gate, residual=residual, no_gate_relu=no_gate_relu, no_gate_bias=no_gate_bias)
        else:
            self.attn = None
    def forward(self, src, lengths=None, encoder_state=None, memory_lengths=None, require_emb=False, src1_lengths=None):
        "See :obj:`EncoderBase.forward()`"

        if isinstance(src, tuple) and len(src) == 2:
            src, _ = src
        else:
            assert False, (isinstance(src, tuple), len(src))

        self._check_args(src, lengths, encoder_state)

        emb = self.dropout(self.embeddings(src))
        s_len, batch, emb_dim = emb.size()

        if self.enc_mode == "LSTM":
            _, decoder_output = self.transform_layer(emb)

        elif self.enc_mode == "CNN":
            _, decoder_output = self.transform_layer(emb)

        elif self.enc_mode == "SA":
            assert int(self.meta["tuple_num"]) == emb.size(0)
            decoder_output, _ = self.transform_layer(emb.transpose(0, 1).contiguous(),
                emb.transpose(0, 1),
                memory_lengths=self.record_num_tensor.expand(emb.size(1)))
        elif self.enc_mode == "MHSA":
            decoder_output, _ = self.transform_layer(emb.transpose(0, 1))
            decoder_output = decoder_output.transpose(0, 1).contiguous()

        elif self.attn is not None:
            decoder_output, p_attn = self.attn(emb.transpose(0, 1).contiguous(), emb.transpose(0, 1), memory_lengths=lengths)
        else:
            decoder_output = emb

        mean = decoder_output.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = decoder_output
        encoder_final = (mean, mean)
        return encoder_final, memory_bank


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False, sort_data=False, stage2_model=None, inp_hid_size=None, src1_encoder=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.sort_data = sort_data

        if stage2_model is not None:
            self.stage2_pretrain_model = stage2_model

        self.src1_encoder = src1_encoder

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=inp_hid_size if inp_hid_size is not None else embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def sort_input(self, input_data, input_length):
        assert len(input_length.size()) == 1
        sorted_input_length, sorted_idx = torch.sort(input_length, dim=0, descending=True)

        _, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)

        real_input_data = input_data.index_select(1, sorted_idx)
        return real_input_data, sorted_input_length, reverse_idx

    def reverse_sort_result(self, output, reverse_idx):
        return output.index_select(1, reverse_idx)

    def forward(self, src, lengths=None, encoder_state=None, require_emb=False, src1_lengths=None):
        "See :obj:`EncoderBase.forward()`"
        
        if self.src1_encoder is not None:
            assert isinstance(src, tuple) and len(src) == 2
            src, src1 = src
            assert require_emb

            _, src1_result = self.src1_encoder(src1, lengths=src1_lengths, require_emb=True)
        else:
            src1_result = None
            

        if isinstance(src, tuple) and len(src) == 2:
            src, _ = src

        self._check_args(src, lengths, encoder_state)

        emb = src
        
        packed_emb = self.dropout(self.embeddings(emb)) if require_emb else emb
        s_len, batch, emb_dim = packed_emb.size()

        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            if self.sort_data:
                packed_emb, lengths, reverse_idx = self.sort_input(packed_emb, lengths)
            lengths = lengths.view(-1).tolist()
            # print(lengths)
            # print(packed_emb.size())
            packed_emb = pack(packed_emb, lengths)


        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
            if self.sort_data:
                memory_bank = self.reverse_sort_result(memory_bank, reverse_idx)
                encoder_final = tuple([self.reverse_sort_result(enc_hidden, reverse_idx) 
                  for enc_hidden in encoder_final])

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        if src1_result is not None:
            return encoder_final, memory_bank, src1_result
        else:
            return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for i in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs



class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, pointer_decoder_type=None, hier_attn=False, eos_token=None, s2_multi_attn=False, s2_multi_attn_gate=False, s2_multi_attn_diff_gate=False, rl_no_repetition=False):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.eos_token = eos_token
        self.s2_multi_attn = s2_multi_attn
        self.s2_multi_attn_gate = s2_multi_attn_gate
        self.s2_multi_attn_diff_gate = s2_multi_attn_diff_gate
        self.rl_no_repetition = rl_no_repetition
        if self.rl_no_repetition:
            print("no repetition for rl")

        if self.s2_multi_attn_gate:
            self.multi_cont_gate = nn.Sequential(
                        nn.Linear(hidden_size*3, 1),
                        nn.Sigmoid()
                    )
            if self.s2_multi_attn_diff_gate:
                self.multi_copy_prob_gate = nn.Sequential(
                        nn.Linear(hidden_size*3, 1),
                        nn.Sigmoid()
                    )
            else:
                self.multi_copy_prob_gate = self.multi_cont_gate
        else:
            self.multi_copy_prob_gate, self.multi_cont_gate = None, None
        

        self.pointer_decoder_type = pointer_decoder_type

        self.hier_attn = hier_attn
        print("hier_attn for decoder: {}".format(self.hier_attn))
        if hier_attn:
            if pointer_decoder_type == 'pointer':
                self.row_attn = onmt.modules.PointerAttention(hidden_size, attn_type=attn_type)
            else:
                self.row_attn = onmt.modules.GlobalAttention(
                    hidden_size, attn_type=attn_type
                )
        else:
            self.row_attn = None

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        
        if pointer_decoder_type == 'pointer':
            self.attn = onmt.modules.PointerAttention(hidden_size, attn_type=attn_type)
        else:
            if self.s2_multi_attn:
                # for s2
                self.attn = onmt.modules.GlobalAttention(
                    hidden_size, coverage=coverage_attn,
                    attn_type=attn_type, multi_attn='s2attn', cont_gate=self.multi_cont_gate, copy_gate=self.multi_copy_prob_gate)
                # for s1              
                self.multi_attn = onmt.modules.GlobalAttention(
                    hidden_size, coverage=coverage_attn,
                    attn_type=attn_type, multi_attn='s1attn')
            else:
                self.attn = onmt.modules.GlobalAttention(
                    hidden_size, coverage=coverage_attn,
                    attn_type=attn_type)
                self.multi_attn = None
                

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            if hier_attn:
                self.copy_row_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            else:
                self.copy_row_attn = None
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, state, memory_lengths=None, max_length=None, baseline=False, reinforce=False, topk_sample=None, src1_result=None, src1_memory_length=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        assert isinstance(state, RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        if isinstance(memory_bank, tuple):
            _, memory_batch, _ = memory_bank[2].size()
        else:
            _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        if src1_result is not None:
            aeq(tgt_batch, src1_result.size(1))
        # END

        # Run the forward pass of the RNN.
        if reinforce:
            if baseline:
                _, baseline_outputs, _ = self._reinforce_forward_pass(
                tgt, memory_bank, state, memory_lengths=memory_lengths, max_length=max_length, greedy=True, topk_sample=topk_sample, no_repetition=self.rl_no_repetition)
            else:
                baseline_outputs = (None, None, None)
            decoder_final, decoder_outputs, attns = self._reinforce_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths, max_length=max_length, greedy=False, topk_sample=topk_sample, no_repetition=self.rl_no_repetition)

            _, mle_outputs, mle_attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)

            for k in mle_attns:
                if isinstance(mle_attns[k], list) or isinstance(mle_attns[k], tuple):
                    mle_attns[k] = torch.stack(mle_attns[k])

            decoder_outputs = decoder_outputs + baseline_outputs + (mle_outputs, mle_attns)
        else:
            decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths, src1_result=src1_result, src1_memory_length=src1_memory_length)

        special_output_flag = decoder_outputs is None or reinforce
        if special_output_flag:
            final_output = None
        else:
            # Update the state with the result.
            final_output = decoder_outputs[-1].unsqueeze(0)

        coverage = None
        state.update_state(decoder_final, final_output, coverage)
        if not special_output_flag:
            # Concatenates sequence of tensors along a new dimension.
            if isinstance(decoder_outputs, list) or isinstance(decoder_outputs, tuple):
                decoder_outputs = torch.stack(decoder_outputs)

        for k in attns:
            if isinstance(attns[k], list) or isinstance(attns[k], tuple):
                attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))

class PointerRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _reinforce_forward_pass(self, tgt, memory_bank, state, memory_lengths=None, max_length=None, greedy=False, topk_sample=None, no_repetition=False):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.
        assert max_length is not None
        assert self.eos_token is not None
        assert isinstance(self.eos_token, int)
        # Initialize local and return variables.
        if isinstance(memory_bank, tuple) and len(memory_bank) == 3:
            record_bank, row_bank, memory_bank = memory_bank

        # # Check
        # tgt_len, tgt_batch, _ = tgt.size()
        # output_len, output_batch, _ = rnn_output.size()
        # aeq(tgt_len, output_len)
        # aeq(tgt_batch, output_batch)
        # # END

        attns = {}
        std_attns = []
        all_embs_input = []
        reinforce_prob = []
        batch_eos_flag = [False for tmp_i in range(tgt.size(1))]
        batch_real_length = [0 for tmp_i in range(tgt.size(1))]
        if no_repetition or self.rl_no_repetition:
            batch_already_gen_id = [[POINTER_PAD_IDX] for tmp_i in range(tgt.size(1))]

        emb_input_id = torch.t(torch.squeeze(tgt, 2).narrow(0, 0, 1))
        # all_embs_input.append(emb_input_id)

        mem_trans = torch.transpose(memory_bank, 0, 1)

        init_states = state.hidden[0] if isinstance(self.rnn, nn.GRU) else state.hidden

        # loop the generation process
        for lno in range(max_length):
            emb = torch.transpose(torch.cat(
                [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                 zip(mem_trans, emb_input_id)]), 0, 1)
            # Run the starting word for forward RNN
            if isinstance(self.rnn, nn.GRU):
                rnn_output, decoder_final = self.rnn(emb, init_states)
            else:
                rnn_output, decoder_final = self.rnn(emb, init_states)

            init_states = decoder_final[0] if isinstance(self.rnn, nn.GRU) else decoder_final

            _, tgt_batch, _ = tgt.size()
            _, output_batch, _ = rnn_output.size()
            aeq(tgt_batch, output_batch)

            # Calculate the attention.
            if self.hier_attn:
                rnn_output_batch_first = rnn_output.transpose(0, 1).contiguous()
                batch_size, rnn_output_len, rnn_output_dim = rnn_output_batch_first.size()
                row_attn = self.row_attn(
                    rnn_output_batch_first,
                    torch.cat(row_bank, 0).transpose(0, 1)
                )

                special_row_attn = row_attn.narrow(2, 0, row_bank[0].size(0))
                home_ply_row_attn = row_attn.narrow(2, row_bank[0].size(0), row_bank[1].size(0))
                vis_ply_row_attn = row_attn.narrow(2, row_bank[0].size(0)+row_bank[1].size(0), row_bank[2].size(0))
                team_row_attn = row_attn.narrow(2, row_bank[0].size(0)+row_bank[1].size(0)+row_bank[2].size(0), row_bank[3].size(0))

                home_ply_attn = self.attn(
                    rnn_output_batch_first.unsqueeze(1).expand(-1, int(record_bank[1].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
                    record_bank[1].transpose(0, 1)
                )
                vis_ply_attn = self.attn(
                    rnn_output_batch_first.unsqueeze(1).expand(-1, int(record_bank[2].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
                    record_bank[2].transpose(0, 1)
                )

                team_attn = self.attn(
                    rnn_output_batch_first.unsqueeze(1).expand(-1, int(record_bank[3].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
                    record_bank[3].transpose(0, 1)
                )

                p_attn = torch.cat((special_row_attn, 
                  (home_ply_attn.view(home_ply_attn.size(0), batch_size, -1, home_ply_attn.size(2)) + home_ply_row_attn.unsqueeze(3).expand(-1, -1, -1, home_ply_attn.size(2))).view(home_ply_attn.size(0), batch_size, -1),
                  (vis_ply_attn.view(vis_ply_attn.size(0), batch_size, -1, vis_ply_attn.size(2)) + vis_ply_row_attn.unsqueeze(3).expand(-1, -1, -1, vis_ply_attn.size(2))).view(vis_ply_attn.size(0), batch_size, -1),
                  (team_attn.view(team_attn.size(0), batch_size, -1, team_attn.size(2)) + team_row_attn.unsqueeze(3).expand(-1, -1, -1, team_attn.size(2))).view(team_attn.size(0), batch_size, -1)
                  ), 2)
                # print(p_attn.sum(2))
                assert p_attn.size(2) == memory_bank.size(0)

            else:
                assert self.pointer_decoder_type == 'pointer'
                p_attn = self.attn(
                    rnn_output.transpose(0, 1).contiguous(),
                    memory_bank.transpose(0, 1),
                    memory_lengths=None
                )
            assert p_attn.size(0) == 1
            std_attns.append(p_attn)

            # use p_attn to sample an output for the next input
            # this result should be documented

            if no_repetition or self.rl_no_repetition:
                assert p_attn.size(1) == len(batch_already_gen_id)
                for tmp_bid in range(p_attn.size(1)):
                    p_attn[:, tmp_bid, batch_already_gen_id[tmp_bid]] = -float('inf')


            r_prob = Categorical(torch.exp(p_attn).squeeze(0))
            if greedy:
                r_action = p_attn.squeeze(0).max(dim=1)[1]
            else:
                if topk_sample is not None:
                    probs, indexes = torch.exp(p_attn).squeeze(0).topk(topk_sample, 1)
                    r_prob = Categorical(probs)
                    r_action = r_prob.sample()
                else:
                    r_action = r_prob.sample()

            action_prob = r_prob.probs.gather(-1, r_action).log() if r_action.dim() == 1 and r_prob.probs.dim() == 1 else r_prob.probs.gather(-1, r_action.unsqueeze(-1)).squeeze(-1).log()
            # action_prob = r_prob.log_prob(r_action)

            if not greedy and topk_sample is not None:
                # FIGUREOUT HOW TO DO ADVANCED INDEX
                r_action = torch.cat(
                [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                 zip(indexes, r_action.unsqueeze(1))]).squeeze(1)

            if no_repetition or self.rl_no_repetition:
                for each_bid, each_gen in enumerate(r_action):
                    batch_already_gen_id[each_bid].append(each_gen.item())
            emb_input_id = r_action.unsqueeze(1)
            all_embs_input.append(emb_input_id)
            reinforce_prob.append(action_prob)

            # whether to stop or not
            for each_id, each_one in enumerate(emb_input_id.squeeze(1)):
                each_one = each_one.item()

                if not batch_eos_flag[each_id]:
                    batch_real_length[each_id] += 1

                assert isinstance(each_one, int)
                if each_one == self.eos_token:
                    batch_eos_flag[each_id] = True
                

            if all(batch_eos_flag):
                break

        attns["std"] = torch.cat(std_attns, 0)

        #decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, (all_embs_input, reinforce_prob, batch_real_length), attns

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None, src1_result=None, src1_memory_length=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.
        # Initialize local and return variables.
        if isinstance(memory_bank, tuple) and len(memory_bank) == 3:
            record_bank, row_bank, memory_bank = memory_bank

        attns = {}
        # print(tgt.size())
        # print(tgt)
        # print(memory_bank.size())
        emb = torch.transpose(torch.cat(
            [torch.index_select(a, 0, i).unsqueeze(0) for a, i in
             zip(torch.transpose(memory_bank, 0, 1), torch.t(torch.squeeze(tgt,2)))]), 0, 1)
        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            # print(emb.size())
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        if self.hier_attn:
            rnn_output_batch_first = rnn_output.transpose(0, 1).contiguous()
            batch_size, rnn_output_len, rnn_output_dim = rnn_output_batch_first.size()
            row_attn = self.row_attn(
                rnn_output_batch_first,
                torch.cat(row_bank, 0).transpose(0, 1)
            )

            special_row_attn = row_attn.narrow(2, 0, row_bank[0].size(0))
            home_ply_row_attn = row_attn.narrow(2, row_bank[0].size(0), row_bank[1].size(0))
            vis_ply_row_attn = row_attn.narrow(2, row_bank[0].size(0)+row_bank[1].size(0), row_bank[2].size(0))
            team_row_attn = row_attn.narrow(2, row_bank[0].size(0)+row_bank[1].size(0)+row_bank[2].size(0), row_bank[3].size(0))

            home_ply_attn = self.attn(
                rnn_output_batch_first.unsqueeze(1).expand(-1, int(record_bank[1].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
                record_bank[1].transpose(0, 1)
            )
            vis_ply_attn = self.attn(
                rnn_output_batch_first.unsqueeze(1).expand(-1, int(record_bank[2].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
                record_bank[2].transpose(0, 1)
            )

            team_attn = self.attn(
                rnn_output_batch_first.unsqueeze(1).expand(-1, int(record_bank[3].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
                record_bank[3].transpose(0, 1)
            )

            p_attn = torch.cat((special_row_attn, 
              (home_ply_attn.view(home_ply_attn.size(0), batch_size, -1, home_ply_attn.size(2)) + home_ply_row_attn.unsqueeze(3).expand(-1, -1, -1, home_ply_attn.size(2))).view(home_ply_attn.size(0), batch_size, -1),
              (vis_ply_attn.view(vis_ply_attn.size(0), batch_size, -1, vis_ply_attn.size(2)) + vis_ply_row_attn.unsqueeze(3).expand(-1, -1, -1, vis_ply_attn.size(2))).view(vis_ply_attn.size(0), batch_size, -1),
              (team_attn.view(team_attn.size(0), batch_size, -1, team_attn.size(2)) + team_row_attn.unsqueeze(3).expand(-1, -1, -1, team_attn.size(2))).view(team_attn.size(0), batch_size, -1)
              ), 2)
            # print(p_attn.sum(2))
            assert p_attn.size(2) == memory_bank.size(0)

        else:
            assert self.pointer_decoder_type == 'pointer'
            p_attn = self.attn(
                rnn_output.transpose(0, 1).contiguous(),
                memory_bank.transpose(0, 1),
                memory_lengths=None
            )
        attns["std"] = p_attn


        #decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, None, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.hidden_size
        # return self.embeddings.embedding_size


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        decoder_outputs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                decoder_outputs.view(-1, decoder_outputs.size(2))
            )
            decoder_outputs = \
                decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def get_hier_rep_and_attn(self, rnn_output, record_bank, row_bank, orig_memory_bank, attn_func, row_attn_func):
        rnn_output_batch_first = rnn_output.unsqueeze(1)
        
        row_attn_vec, row_attn = row_attn_func(
            rnn_output_batch_first,
            torch.cat(row_bank, 0).transpose(0, 1)
        )

        rnn_output_batch_for_record_attn = rnn_output_batch_first

        batch_size, rnn_output_len, rnn_output_dim = rnn_output_batch_for_record_attn.size()

        special_row_attn = row_attn.narrow(2, 0, row_bank[0].size(0))

        home_ply_row_attn = row_attn.narrow(2, row_bank[0].size(0), row_bank[1].size(0))
        vis_ply_row_attn = row_attn.narrow(2, row_bank[0].size(0)+row_bank[1].size(0), row_bank[2].size(0))
        team_row_attn = row_attn.narrow(2, row_bank[0].size(0)+row_bank[1].size(0)+row_bank[2].size(0), row_bank[3].size(0))

        _, home_ply_attn = attn_func(
            rnn_output_batch_for_record_attn.unsqueeze(1).expand(-1, int(record_bank[1].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
            record_bank[1].transpose(0, 1)
        )
        _, vis_ply_attn = attn_func(
            rnn_output_batch_for_record_attn.unsqueeze(1).expand(-1, int(record_bank[2].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
            record_bank[2].transpose(0, 1)
        )

        _, team_attn = attn_func(
            rnn_output_batch_for_record_attn.unsqueeze(1).expand(-1, int(record_bank[3].size(1) / batch_size), -1, -1).contiguous().view(-1, rnn_output_len, rnn_output_dim),
            record_bank[3].transpose(0, 1)
        )

        p_attn = torch.cat((special_row_attn, 
          (home_ply_attn.view(home_ply_attn.size(0), batch_size, -1, home_ply_attn.size(2)) * home_ply_row_attn.unsqueeze(3).expand(-1, -1, -1, home_ply_attn.size(2))).view(home_ply_attn.size(0), batch_size, -1),
          (vis_ply_attn.view(vis_ply_attn.size(0), batch_size, -1, vis_ply_attn.size(2)) * vis_ply_row_attn.unsqueeze(3).expand(-1, -1, -1, vis_ply_attn.size(2))).view(vis_ply_attn.size(0), batch_size, -1),
          (team_attn.view(team_attn.size(0), batch_size, -1, team_attn.size(2)) * team_row_attn.unsqueeze(3).expand(-1, -1, -1, team_attn.size(2))).view(team_attn.size(0), batch_size, -1)
          ), 2)
        attn_h, _ = attn_func(
            rnn_output_batch_for_record_attn, 
            orig_memory_bank.transpose(0, 1),
            align_score=p_attn)
        # print(p_attn.sum(2))
        return attn_h.squeeze(0), p_attn.squeeze(0)

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None, src1_result=None, src1_memory_length=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        if isinstance(memory_bank, tuple) and len(memory_bank) == 3:
            record_bank, row_bank, memory_bank = memory_bank

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if src1_result is not None:
            attns["std_src1"] = []
        if self._copy:
            attns["copy"] = []
            if src1_result is not None:
                attns["copy_src1"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            if self.hier_attn:
                if src1_result is not None:
                    assert False, "Not yet supported"
                decoder_output, p_attn = self.get_hier_rep_and_attn(rnn_output, record_bank, row_bank, memory_bank, self.attn, self.row_attn)
            else:
                if src1_result is None:
                    decoder_output, p_attn = self.attn(
                        rnn_output,
                        memory_bank.transpose(0, 1),
                        memory_lengths=memory_lengths)
                    s1_p_attn = None
                else:
                    s1_c, s1_p_attn = self.multi_attn(
                        rnn_output,
                        src1_result.transpose(0, 1),
                        memory_lengths=src1_memory_length
                        )
                    # copy_gate_res: (batch, tgt_len, 1)
                    decoder_output, p_attn, copy_gate_res = self.attn(
                        rnn_output,
                        memory_bank.transpose(0, 1),
                        memory_lengths=memory_lengths,
                        c1=s1_c)
                    if copy_gate_res is not None:
                        if "copy_gate_res" not in attns:
                            attns["copy_gate_res"] = []
                        attns["copy_gate_res"] += [copy_gate_res.transpose(0, 1).contiguous()]


            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]
            if src1_result is not None:
                attns["std_src1"] += [s1_p_attn]


            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                if src1_result is not None:
                    assert False, "Not supported yet"
                if self.hier_attn:
                    _, copy_attn = self.get_hier_rep_and_attn(decoder_output, record_bank, row_bank, memory_bank, self.copy_attn, self.copy_row_attn)
                else:
                    _, copy_attn = self.copy_attn(decoder_output,
                                                  memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
                if "std_src1" in attns:
                    attns["copy_src1"] = attns["std_src1"]
        # Return result.
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None, require_emb=False, max_length=None, baseline=False, reinforce=False, topk_sample=None, src1_length=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_results = self.encoder(src, lengths=lengths, require_emb=require_emb, src1_lengths=src1_length)
        if isinstance(enc_results, tuple):
            if len(enc_results) == 2:
                enc_final, memory_bank = enc_results
                src1_result = None
            elif len(enc_results) == 3:
                enc_final, memory_bank, src1_result = enc_results
            else:
                assert False
        else:
            assert False

        # print("mem bank")
        # print(memory_bank.size())
        # print(src[0][0].size())
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths, 
                         max_length=max_length,
                         baseline=baseline,
                         reinforce=reinforce,
                         topk_sample=topk_sample,
                         src1_result=src1_result,
                         src1_memory_length=src1_length)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state, memory_bank[2] if isinstance(memory_bank, tuple) else memory_bank


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        # for h in self._all:
        #     if h is not None:
        #         h = h.detach()
        h_d = []
        for tmp_i in self.hidden:
            if tmp_i is not None:
                h_d.append(tmp_i.detach())
            else:
                h_d.append(tmp_i)
        self.hidden = tuple(h_d)

        self.input_feed = self.input_feed.detach() if self.input_feed is not None else self.input_feed

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = self.hidden[0].detach().new(*h_size).zero_().unsqueeze(0)
        self.input_feed.requires_grad=False
                                   

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        if input_feed is not None:
            self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [e.data.repeat(1, beam_size, 1)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
