"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        PointerRNNDecoder, HierarchicalEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder, PretrainModel
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder, UsePretrainModel
from onmt.Utils import use_gpu

PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True, hist_dict=None, use_hier_hist=False, use_pretrain=False, tgt_opt_num=None, pretrain_word_dict=None, stage2_pretrain_input=False, position_encoding=False, pos_type=None, only_pos=False, ent_use_pretrain=False, text_ent_tgt_opt_num=None, ent_not_pretrain_vocab=None):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    feat_vec_size = opt.feat_vec_size
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    # construct and load the Pretrain Model / Embedding for other words in order for Embeddings to calculate 
    if use_pretrain:
        pretrain_ckpt = torch.load(opt.pretrain_model_path,
                            map_location=lambda storage, loc: storage)
        pretrain_model_opt = pretrain_ckpt['opt']
        pretrain_fields = onmt.io.load_fields_from_vocab(
                pretrain_ckpt['vocab'], "pretrain", pretrain_model_opt)

        pretrain_model = onmt.ModelConstructor.make_base_model(pretrain_model_opt, pretrain_fields,
                                                      use_gpu(opt), pretrain_ckpt, stage1=False, pretrain=True, tgt_opt_num=tgt_opt_num)

        # how to determine num word embeddings and emb dim
        pretrain_word_padding_idx = pretrain_word_dict["not_pretrain"].stoi[onmt.io.PAD_WORD]
        pretrain_num_word_embeddings = len(pretrain_word_dict["not_pretrain"])
        not_pretrain_emb = Embeddings(word_vec_size=embedding_dim,
                      position_encoding=False,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=pretrain_word_padding_idx,
                      feat_padding_idx=[],
                      word_vocab_size=pretrain_num_word_embeddings,
                      feat_vocab_sizes=[]
                      )
    else:
        pretrain_model = None
        not_pretrain_emb = None

    if ent_use_pretrain:
        ent_pretrain_ckpt = torch.load(opt.ent_pretrain_model_path,
                            map_location=lambda storage, loc: storage)
        ent_pretrain_model_opt = ent_pretrain_ckpt['opt']
        ent_pretrain_fields = onmt.io.load_fields_from_vocab(
                ent_pretrain_ckpt['vocab'], "pretrain", ent_pretrain_model_opt)


        ent_pretrain_word_padding_idx = ent_not_pretrain_vocab.stoi[onmt.io.PAD_WORD]
        ent_pretrain_num_word_embeddings = len(ent_not_pretrain_vocab)

        ent_not_pretrain_emb = Embeddings(
                      word_vec_size=embedding_dim,
                      position_encoding=False,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=ent_pretrain_word_padding_idx,
                      feat_padding_idx=[],
                      word_vocab_size=ent_pretrain_num_word_embeddings,
                      feat_vocab_sizes=[]
                      )


        ent_pretrain_model = onmt.ModelConstructor.make_base_model(ent_pretrain_model_opt, ent_pretrain_fields,
                                                      use_gpu(opt), ent_pretrain_ckpt, stage1=False, pretrain=True, tgt_opt_num=text_ent_tgt_opt_num)
     
    else:
        ent_pretrain_model = None
        ent_not_pretrain_emb = None
        

    main_emb = Embeddings(word_vec_size=embedding_dim,
                      position_encoding=position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      use_pretrain=use_pretrain or ent_use_pretrain,
                      pretrain_model=pretrain_model,
                      not_pretrain_emb=not_pretrain_emb,
                      stage2_pretrain_input=stage2_pretrain_input,
                      pos_type=pos_type,  
                      only_pos=only_pos,
                      ent_pretrain_model=ent_pretrain_model,
                      ent_not_pretrain_emb=ent_not_pretrain_emb
                      )

    if use_hier_hist:
        assert for_encoder and hist_dict is not None
        assert word_dict == hist_dict
        hist_padding_idx = hist_dict.stoi[onmt.io.PAD_WORD]
        num_hist_embeddings = len(hist_dict)
        assert len(feats_padding_idx) == 3
        assert len(main_emb.get_feat_emb) == 3
        external_embedding = [main_emb.get_val_emb] + main_emb.get_feat_emb[:2]
        hier_hist_emb = Embeddings(word_vec_size=embedding_dim,
                      position_encoding=position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=hist_padding_idx,
                      feat_padding_idx=feats_padding_idx[:2],
                      word_vocab_size=num_hist_embeddings,
                      feat_vocab_sizes=num_feat_embeddings[:2],
                      emb_for_hier_hist=True,
                      external_embedding=external_embedding,
                      pos_type=pos_type,  
                      only_pos=only_pos
          )
        return (main_emb, hier_hist_emb)
    else:
        return main_emb

def make_encoder(opt, embeddings, stage1=True, basic_enc_dec=False, no_cs_gate=False, pretrain=False, tgt_opt_num=None, stage2_use_pretrain=False, pretrain_word_dict=None, stage2_model=None, s2_multi_attn=False):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
        stage1: stage1 encoder
    """

    if pretrain:
        # find the right categories from data - should save in the dataset
        return PretrainModel(embeddings, opt.pre_num_layers, opt.pre_d_model, opt.pre_heads, opt.pre_d_ff, opt.pre_dropout, opt.pre_max_relative_positions, tgt_opt_num, opt.pre_loss_type, opt.pre_use_mlp if hasattr(opt, "pre_use_mlp") else False, opt.pretrain_emb_ent if hasattr(opt, 'pretrain_emb_ent') else False, opt.pre_ent_type if hasattr(opt, 'pre_ent_type') else None, opt.pre_ent_use_data if hasattr(opt, 'pre_ent_use_data') else False)
    elif stage2_use_pretrain:
        pretrain_ckpt = torch.load(opt.pretrain_model_path,
                            map_location=lambda storage, loc: storage)
        pretrain_model_opt = pretrain_ckpt['opt']
        pretrain_fields = onmt.io.load_fields_from_vocab(
                pretrain_ckpt['vocab'], "pretrain", pretrain_model_opt)

        pretrain_model = onmt.ModelConstructor.make_base_model(pretrain_model_opt, pretrain_fields,
                                                      use_gpu(opt), pretrain_ckpt, stage1=False, pretrain=True, tgt_opt_num=tgt_opt_num)

        # how to determine num word embeddings and emb dim
        pretrain_word_padding_idx = pretrain_word_dict["not_pretrain"].stoi[onmt.io.PAD_WORD]
        pretrain_num_word_embeddings = len(pretrain_word_dict["not_pretrain"])
        embedding_dim = opt.src_word_vec_size
        feat_vec_size = opt.feat_vec_size
        not_pretrain_emb = Embeddings(word_vec_size=embedding_dim,
                      position_encoding=False,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=pretrain_word_padding_idx,
                      feat_padding_idx=[],
                      word_vocab_size=pretrain_num_word_embeddings,
                      feat_vocab_sizes=[]
                      )
        # assert False
        return UsePretrainModel(pretrain_model, not_pretrain_emb, None, None, only_val=True)

    elif stage1 or basic_enc_dec:
        if opt.mlb:
            if hasattr(opt, "mlb_mean_enc") and opt.mlb_mean_enc:
                return MeanEncoder(opt.hier_meta, opt.enc_layers1, embeddings, opt.src_word_vec_size, opt.attn_hidden, opt.dropout, enable_attn=not basic_enc_dec and not no_cs_gate, no_gate=opt.gsa_no_gate if opt.gsa_no_gate is not None else False, residual=opt.gsa_residual if opt.gsa_residual is not None else False, no_gate_relu=opt.gsa_no_gate_relu if opt.gsa_no_gate_relu is not None else False, no_gate_bias=opt.gsa_no_gate_bias if opt.gsa_no_gate_bias is not None else False, mode=opt.mean_enc_mode if opt.mean_enc_mode is not None else "MLP", trans_layer=opt.mean_trans_layer, rnn_type=opt.mean_rnn_type, bidirectional=opt.mean_rnn_bi if opt.mean_rnn_bi is not None else False, hidden_size=opt.mean_hidden, cnn_kernel_width=opt.cnn_kernel_width, multi_head_count=opt.multi_head_count, multi_head_dp=opt.multi_head_dp)
            else:
                return RNNEncoder(opt.rnn_type, opt.brnn2, opt.enc_layers2,
                              opt.rnn_size, opt.dropout, embeddings,
                              opt.bridge, sort_data=True)            
        elif opt.hier_model1:
            return HierarchicalEncoder(opt.hier_meta, opt.enc_layers1, embeddings, opt.src_word_vec_size, opt.attn_hidden, opt.hier_rnn_type, opt.hier_bidirectional, opt.hier_rnn_size, dropout=opt.dropout, attn_type=opt.global_attention, two_dim_record=opt.hier_two_dim_record if opt.hier_two_dim_record is not None else False, row_self_attn_type=opt.row_self_attn_type, col_self_attn_type=opt.col_self_attn_type, multi_head_count=opt.multi_head_count, multi_head_dp=opt.multi_head_dp, two_dim_score=opt.two_dim_score, ply_level_gate=opt.ply_level_gate if opt.ply_level_gate is not None else False, hier_history=opt.hier_history if opt.hier_history is not None else False, hier_history_seq_type=opt.hier_history_seq_type, hier_history_seq_window=opt.hier_history_seq_window, hier_bi=opt.hier_bi if opt.hier_bi is not None else False, hier_num_layers=opt.hier_num_layers, hier_hist_attn_type=opt.hier_hist_attn_type, hier_hist_attn_pos_type=opt.hier_hist_attn_pos_type, norow=opt.norow if opt.norow is not None else False, nocolumn=opt.nocolumn if opt.nocolumn is not None else False, nofusion=opt.nofusion if opt.nofusion is not None else False, nohierstructure=opt.nohierstructure if opt.nohierstructure is not None else False, nohistpos=opt.nohistpos if opt.nohistpos is not None else False, is_cnn_cell=opt.cnncell if opt.cnncell is not None else False, cnn_kernel_width=opt.cnn_kernel_width, mlp_hier=opt.mlp_hier if opt.mlp_hier is not None else False, two_mlp_hier=opt.two_mlp_hier if opt.two_mlp_hier is not None else False)
        else:
            return MeanEncoder(opt.hier_meta, opt.enc_layers1, embeddings, max(opt.src_word_vec_size, opt.feat_vec_size), opt.attn_hidden, opt.dropout, enable_attn=not basic_enc_dec and not no_cs_gate, no_gate=opt.gsa_no_gate if opt.gsa_no_gate is not None else False, residual=opt.gsa_residual if opt.gsa_residual is not None else False, no_gate_relu=opt.gsa_no_gate_relu if opt.gsa_no_gate_relu is not None else False, no_gate_bias=opt.gsa_no_gate_bias if opt.gsa_no_gate_bias is not None else False, mode=opt.mean_enc_mode if opt.mean_enc_mode is not None else "MLP", trans_layer=opt.mean_trans_layer, rnn_type=opt.mean_rnn_type, bidirectional=opt.mean_rnn_bi if opt.mean_rnn_bi is not None else False, hidden_size=opt.mean_hidden, cnn_kernel_width=opt.cnn_kernel_width, multi_head_count=opt.multi_head_count, multi_head_dp=opt.multi_head_dp)
    else:
        # "rnn" or "brnn"
        if s2_multi_attn:
            if opt.mlb:
                src1_encoder = RNNEncoder(opt.rnn_type, opt.brnn2, opt.enc_layers2,
                              opt.rnn_size, opt.dropout, embeddings,
                              opt.bridge, sort_data=True)
            else:
                src1_encoder = MeanEncoder(opt.hier_meta, opt.enc_layers1, embeddings, opt.src_word_vec_size, opt.attn_hidden, opt.dropout, enable_attn=not basic_enc_dec and not no_cs_gate, no_gate=opt.gsa_no_gate if opt.gsa_no_gate is not None else False, residual=opt.gsa_residual if opt.gsa_residual is not None else False, no_gate_relu=opt.gsa_no_gate_relu if opt.gsa_no_gate_relu is not None else False, no_gate_bias=opt.gsa_no_gate_bias if opt.gsa_no_gate_bias is not None else False, mode=opt.mean_enc_mode if opt.mean_enc_mode is not None else "MLP", trans_layer=opt.mean_trans_layer, rnn_type=opt.mean_rnn_type, bidirectional=opt.mean_rnn_bi if opt.mean_rnn_bi is not None else False, hidden_size=opt.mean_hidden, cnn_kernel_width=opt.cnn_kernel_width, multi_head_count=opt.multi_head_count, multi_head_dp=opt.multi_head_dp)
        else:
            src1_encoder = None
        return RNNEncoder(opt.rnn_type, opt.brnn2, opt.enc_layers2,
                          opt.rnn_size, opt.dropout, embeddings,
                          opt.bridge, sort_data=True, stage2_model=stage2_model, inp_hid_size=opt.rnn_size if not opt.sep_train else None, src1_encoder=src1_encoder)


def make_decoder(opt, embeddings, stage1, basic_enc_dec):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
        stage1: stage1 decoder
    """
    if stage1:
        if hasattr(opt, 'mlb_mean_enc'):
            tmp_flag = opt.mlb_mean_enc
        else:
            tmp_flag = False

        return PointerRNNDecoder(opt.rnn_type, opt.brnn or (opt.mlb and not tmp_flag),
                             opt.dec_layers1, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             False,
                             opt.dropout,
                             embeddings,
                             False,
                             opt.decoder_type1, 
                             opt.hier_model1 and not opt.nohierstructure, 
                             EOS_INDEX,
                             rl_no_repetition=opt.no_repetition if hasattr(opt, "no_repetition") else False)
    else:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn2,
                                   opt.dec_layers2, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   True,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn,
                                   hier_attn=opt.hier_model2 and not opt.nohierstructure if opt.hier_model2 is not None and basic_enc_dec else False, s2_multi_attn=opt.s2_multi_attn if hasattr(opt, "s2_multi_attn") else False, s2_multi_attn_gate=opt.s2_multi_attn_gate if hasattr(opt, "s2_multi_attn_gate") else False, s2_multi_attn_diff_gate=opt.s2_multi_attn_diff_gate if hasattr(opt, "s2_multi_attn_diff_gate") else False
                                   )

def load_test_model(opt, dummy_opt, stage1=False):
    opt_model = opt.model if stage1 else opt.model2
    checkpoint = torch.load(opt_model,
                            map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']


    if model_opt.pretrain_model_path:
        if model_opt.pretrain_model_path == "/home/pcl_user/hg/mlb-codes/mlb_orig/gen_model/f_pretrain_orig_dim_300_model_no_pos_enc_2cat_noself_lay2_h3_ff1024_dp0.3_rank_0.2_20200520/mlb_stage1_acc_99.9952_ppl_1.0483_e148.pt":
            model_opt.pretrain_model_path = "/users6/hgong/data2text-rl-two-stage/mlb_orig/gen_model/f_pretrain_orig_dim_300_model_no_pos_enc_2cat_noself_lay2_h3_ff1024_dp0.3_rank_0.2_20200520/mlb_stage1_acc_99.9952_ppl_1.0483_e148.pt"

    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], opt.data_type, model_opt)

    
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    # find tgt_opt_num from model parameters
    tgt_opt_num = None
    text_ent_tgt_opt_num = None
    if model_opt.use_pretrain or model_opt.pretrain_emb:
        if model_opt.pretrain_emb:
            tgt_num, dmodel = checkpoint['model']['inf_layer.0.weight'].size()

            real_dmodel, = checkpoint['model']["transformer.layer_norm.weight"].size()            
        elif model_opt.sep_train and model_opt.stage2_train:
            tgt_num, dmodel = checkpoint['model']['encoder.stage2_pretrain_model.pretrain_model.inf_layer.0.weight'].size()

            real_dmodel, = checkpoint['model']["encoder.stage2_pretrain_model.pretrain_model.transformer.layer_norm.weight"].size()
        else:
            if model_opt.val_use_pretrain:
                tgt_num, dmodel = checkpoint['model']['encoder.embeddings.make_embedding.pretrain_module.pretrain_model.inf_layer.0.weight'].size()

                real_dmodel, = checkpoint['model']["encoder.embeddings.make_embedding.pretrain_module.pretrain_model.transformer.layer_norm.weight"].size()
        
                if dmodel != real_dmodel:
                    tgt_opt_num = tgt_num

            if model_opt.ent_use_pretrain:
                tgt_num, dmodel = checkpoint['model']['encoder.embeddings.make_embedding.pretrain_module.ent_pretrain_model.inf_layer.0.weight'].size()

                real_dmodel, = checkpoint['model']["encoder.embeddings.make_embedding.pretrain_module.ent_pretrain_model.transformer.layer_norm.weight"].size()
        
                if dmodel != real_dmodel:
                    text_ent_tgt_opt_num = tgt_num                

    if model_opt.pretrain_emb:
        model = make_base_model(model_opt, fields, use_gpu(opt), checkpoint, stage1=False, pretrain=True, tgt_opt_num=tgt_opt_num)

    else:
        model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint, stage1, model_opt.basicencdec, tgt_opt_num=tgt_opt_num, use_pretrain=model_opt.use_pretrain and (not (model_opt.sep_train and model_opt.stage2_train)) , pretrain_word_dict=fields["src1_pretrain"].vocab, stage2_use_pretrain=(model_opt.sep_train and model_opt.stage2_train) and model_opt.use_pretrain, text_ent_tgt_opt_num=text_ent_tgt_opt_num, ent_use_pretrain=model_opt.ent_use_pretrain, val_use_pretrain=model_opt.val_use_pretrain)
    model.eval()
    if hasattr(model, "generator"):
        model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None, stage1=True, basic_enc_dec=False, pretrain=False, tgt_opt_num=None, use_pretrain=False, pretrain_word_dict=None, stage2_use_pretrain=False, text_ent_tgt_opt_num=None, ent_use_pretrain=False, val_use_pretrain=False):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    # compatability -> previous only val
    if use_pretrain:
        if (not ent_use_pretrain) and (not val_use_pretrain):
            val_use_pretrain = True

    device = torch.device("cuda") if gpu else torch.device("cpu")

    if pretrain:
        src = "src_pretrain"
        tgt = "tgt_pretrain"
        if hasattr(model_opt, 'pretrain_emb_ent') and model_opt.pretrain_emb_ent and model_opt.pre_ent_use_data:
            src_data = "src_pretrain_data"
            src_data_idx = "src_pretrain_data_idx"
        else:
            src_data = None
            src_data_idx = None

    elif stage1 and not basic_enc_dec:
        src = "src1"
        # src_char = "src1_char"
        tgt = "tgt1"
    else:
        src = "src2"
        # src_char = "src2_char"
        tgt = "tgt2"

    src_hist = "src1_hist" if (basic_enc_dec or stage1) else None
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        no_cs_gate_flag = model_opt.nocsgate if model_opt.nocsgate is not None else False
        if stage2_use_pretrain:
            assert not pretrain
            stage2_model = make_encoder(model_opt, None, stage1, basic_enc_dec, no_cs_gate_flag, pretrain, tgt_opt_num, True, pretrain_word_dict)
        else:
            stage2_model = None
        src_dict = fields[src].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, src)
        # src_char_dict = fields[src_char].vocab if model_opt.char_enable else None

        assert not (use_pretrain and pretrain)

        # print(fields.keys())

        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts, hist_dict=fields[src_hist].vocab if model_opt.hier_history and (basic_enc_dec or stage1) and not pretrain else None, use_hier_hist=model_opt.hier_history and (basic_enc_dec or stage1) and not pretrain, use_pretrain=use_pretrain and val_use_pretrain, tgt_opt_num=tgt_opt_num, pretrain_word_dict=pretrain_word_dict,
                                            stage2_pretrain_input=stage2_model is not None, position_encoding=pretrain and model_opt.position_encoding, pos_type=model_opt.pos_type if hasattr(model_opt, "pos_type") else None, only_pos=model_opt.only_pos if hasattr(model_opt, "only_pos") else None, ent_use_pretrain=use_pretrain and ent_use_pretrain, text_ent_tgt_opt_num=text_ent_tgt_opt_num,
                                            ent_not_pretrain_vocab=fields["src1_ent_pretrain"].vocab['not_pretrain'] if "src1_ent_pretrain" in fields else None)
        enc_inp_emb = src_embeddings

        if hasattr(model_opt, 'pretrain_emb_ent') and model_opt.pretrain_emb_ent and model_opt.pre_ent_use_data:
            src_data_dict = fields[src_data].vocab
            feature_data_dicts = onmt.io.collect_feature_vocabs(fields, src_data)
            src_data_embeddings = make_embeddings(model_opt, src_data_dict,
                                         feature_data_dicts, hist_dict=None, use_hier_hist=False, use_pretrain=use_pretrain, tgt_opt_num=tgt_opt_num, pretrain_word_dict=pretrain_word_dict,
                                            stage2_pretrain_input=stage2_model is not None, position_encoding=pretrain and model_opt.position_encoding, pos_type=model_opt.pos_type if hasattr(model_opt, "pos_type") else None, only_pos=model_opt.only_pos if hasattr(model_opt, "only_pos") else None)
            enc_inp_emb = (src_embeddings, src_data_embeddings)

        
        encoder = make_encoder(model_opt, enc_inp_emb, stage1, basic_enc_dec, no_cs_gate_flag, pretrain, tgt_opt_num, stage2_model=stage2_model, s2_multi_attn=model_opt.s2_multi_attn if hasattr(model_opt, "s2_multi_attn") else False)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    if not pretrain:
        # Make decoder.
        tgt_dict = fields[tgt].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, tgt)
        tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                         feature_dicts, for_encoder=False)

        # Share the embedding matrix - preprocess with share_vocab required.
        if model_opt.share_embeddings:
            # src/tgt vocab should be the same if `-share_vocab` is specified.
            if src_dict != tgt_dict:
                raise AssertionError('The `-share_vocab` should be set during '
                                     'preprocess if you use share_embeddings!')

            tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

        decoder = make_decoder(model_opt, tgt_embeddings, stage1 and not basic_enc_dec, basic_enc_dec)

        # Make NMTModel(= encoder + decoder).
        model = NMTModel(encoder, decoder)
    else:
        model = encoder
    model.model_type = model_opt.model_type

    # Make Generator.
    if not pretrain:
        if stage1 and not basic_enc_dec:
            generator = nn.Sequential(
                nn.Linear(model_opt.rnn_size, len(fields["tgt1"].vocab)),
                nn.LogSoftmax())
            if model_opt.share_decoder_embeddings:
                generator[0].weight = decoder.embeddings.word_lut.weight
        else:
            generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt2"].vocab, (model_opt.mlb or model_opt.marginal) if (hasattr(model_opt, "mlb") and hasattr(model_opt, "marginal")) else False, model_opt.s2_multi_attn_multi_copy if hasattr(model_opt, "s2_multi_attn_multi_copy") else None)
    else:
        generator = None

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            if generator is not None:
                for p in generator.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)

        if not pretrain:
            if hasattr(model.encoder, 'embeddings'):
                model.encoder.embeddings.load_pretrained_vectors(
                        model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
            if hasattr(model.decoder, 'embeddings'):
                model.decoder.embeddings.load_pretrained_vectors(
                        model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
        else:
            if hasattr(model, 'embeddings'):
                model.embeddings.load_pretrained_vectors(
                        model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)    

        if use_pretrain:
            model.encoder.embeddings.load_pretrained_contextual_rep(model_opt.pretrain_model_path, model_opt.fix_use_pretrain, model_opt.ent_pretrain_model_path, model_opt.ent_fix_use_pretrain)

        if stage2_use_pretrain:
            model.encoder.stage2_pretrain_model.load_pretrained_contextual_rep(model_opt.pretrain_model_path, model_opt.fix_use_pretrain, model_opt.ent_pretrain_model_path, model_opt.ent_fix_use_pretrain)
    # Add generator to model (this registers it as parameter of model).
    if generator is not None:
        model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    model.to(device)
    # if gpu:
    #     model.cuda()
    # else:
    #     model.cpu()

    return model
