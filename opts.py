import argparse
from onmt.modules.SRU import CheckSRU


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('-mlb', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-mlb_mean_enc', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-marginal', action='store_true',
                       help="codes on mlb dataset")

    group.add_argument('-s2_multi_attn', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_multi_copy', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_gate', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_diff_gate', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_shuffle', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_shuffle_prob', type=float, default=0.2,
                       help='Word embedding size for src.')
    group.add_argument('-s2_multi_attn_shuffle_start_len', type=int, default=10,
                       help='Word embedding size for src.')
    group.add_argument('-s2_multi_attn_shuffle_start_epoch', type=int, default=10,
                       help='Word embedding size for src.')
    
    group.add_argument('-src_word_vec_size', type=int, default=500,
                       help='Word embedding size for src.')
    group.add_argument('-tgt_word_vec_size', type=int, default=-1,
                       help='Word embedding size for tgt.')
    group.add_argument('-word_vec_size', type=int, default=-1,
                       help='Word embedding size for src and tgt.')
    group.add_argument('-attn_hidden', type=int, default=-1,
                       help='Attn hidden size for self attention on input')

    # hier model 1 options
    group.add_argument('-hier_model1', action="store_true",
                       help="""Activate hier model 1""")
    group.add_argument('-hier_model2', action="store_true",
                       help="""Activate hier model 1""")
    group.add_argument('-mlp_hier', action="store_true",
                       help="""Activate hier model 1""")
    group.add_argument('-two_mlp_hier', action="store_true",
                       help="""Activate hier model 1""")
    group.add_argument('-hier_mix_attn', action="store_true",
                       help="""Activate mix hier attn for hier model 1""")
    group.add_argument('-hier_record_level_use_attn', action="store_true",
                       help="""Activate self attention for record level rep of hier model 1""")
    group.add_argument('-hier_row_rnn', action="store_true",
                       help="""Activate rnn for row level rep of hier model 1""")
    group.add_argument('-hier_two_dim_record', action="store_true",
                       help="""Activate two dim record for hier model 1""")
    group.add_argument('-two_dim_use_mlp', action="store_true",
                       help="""Activate mlp for two dim rep""")
    group.add_argument('-two_dim_concat', action="store_true",
                       help="""Activate direct concat for two dim rep""")
    group.add_argument('-row_self_attn_type', type=str, default="normal", choices=["normal", "multi-head-attn"],
                       help='row_self_attn_type')
    group.add_argument('-col_self_attn_type', type=str, default="normal", choices=["normal", "multi-head-attn"],
                       help='col_self_attn_type')
    group.add_argument('-multi_head_count', type=int, default=None,
                       help='multi_head_count')
    group.add_argument('-multi_head_dp', type=float, default=0.1,
                       help='multi_head_dp')
    group.add_argument('-true_two_dim_fusion', action="store_true",
                       help="""Activate true_two_dim_fusion""")
    group.add_argument('-ply_level_gate', action="store_true",
                       help="""Activate ply_level_gate""")
    group.add_argument('-gsa_no_gate', action="store_true",
                       help="""Activate gsa_no_gate""")
    group.add_argument('-gsa_no_gate_relu', action="store_true",
                       help="""Activate gsa_no_gate_relu""")
    group.add_argument('-gsa_no_gate_bias', action="store_true",
                       help="""Activate gsa_no_gate_bias""")
    group.add_argument('-gsa_residual', action="store_true",
                       help="""Activate gsa_residual""")
    group.add_argument('-mha_concat', action="store_true",
                       help="""Activate mha_concat""")
    group.add_argument('-mha_residual', action="store_true",
                       help="""Activate mha_residual""")
    group.add_argument('-mha_norm', action="store_true",
                       help="""Activate mha_norm""")
    group.add_argument('-two_dim_softmax', action="store_true",
                       help="""Activate two_dim_softmax""")
    group.add_argument('-two_dim_gen_tanh', action="store_true",
                       help="""Activate two_dim_gen_tanh""")
    group.add_argument('-two_dim_score', type=str, default=None, choices=["general", "mlp", "dot", None],
                       help='two_dim_score')                          
    group.add_argument('-true_hier_record_level_use_attn', action="store_true",
                       help="""Activate true_hier_record_level_use_attn""")
    group.add_argument('-cnncell', action="store_true",
                       help="""Activate true_hier_record_level_use_attn""")
    # group.add_argument('-cnn_kernel_width', type=int, default=None,
    #                    help='multi_head_count')

    # hier history
    group.add_argument('-hier_history', action="store_true",
                       help="""Activate hier_history""")
    group.add_argument('-hist_scalar_attn_map', action="store_true",
                       help="""Activate hist_scalar_attn_map""")
    group.add_argument('-hist_scalar_attn_type', type=str, default=None,
                       choices=['dot', 'general', 'mlp', None],
                       help="""The attention type for hier history""")
    group.add_argument('-hist_scalar_attn_num_interval', type=int, default=None,
                       help="""hist_scalar_attn_num_interval""")
    
    # hier seq history
    group.add_argument('-hier_history_seq', action="store_true",
                       help="""Activate hier_history_seq""")
    group.add_argument('-hier_history_seq_type', type=str, default=None,
                       choices=['rnn', 'RNN', "SA", "sa", None])
    group.add_argument('-hier_history_seq_window', type=int, default=None,
                       help="""hier_history_seq_window""")
    group.add_argument('-hier_bi', action="store_true",
                       help="""Activate hier_bi""")
    group.add_argument('-hier_num_layers', type=int, default=None)
    group.add_argument('-hier_hist_attn_type', type=str, default=None,
                       choices=['dot', 'general', 'mlp', None])
    group.add_argument('-hier_hist_attn_pos_type', type=str, default=None,
                       choices=["posEmb", "posEncoding", None])

    group.add_argument('-two_dim_gate_direct_scalar', action="store_true",
                       help="""Activate direct_scalar gate for two dim rep""")
    group.add_argument('-two_dim_gate_activation_scalar', action="store_true",
                       help="""Activate activation_scalar gate for two dim rep""")

    group.add_argument('-hier_col_dim_rnn', action="store_true",
                       help="""Activate rnn for col dim rep of two dim record encoder""")
    group.add_argument('-hier_meta', type=str, default=None,
                       help="""hier meta path""")
    group.add_argument('-hier_rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       action=CheckSRU,
                       help="""The gate type to use in the hier RNNs""")
    group.add_argument('-hier_rnn_size', type=int, default=600,
                       help='Size of hier rnn hidden states')
    group.add_argument('-hier_bidirectional', action="store_true",
                       help="""Activate biRNN for hier model""")

    # mean encoders
    group.add_argument('-mean_enc_mode', type=str, default='MLP',
                       choices=["MLP", "LSTM", "CNN", "SA", "MHSA"],
                       help="""The gate type to use in the hier RNNs""")
    group.add_argument('-mean_trans_layer', type=int, default=None,
                       help='Size of hier rnn hidden states')
    group.add_argument('-mean_rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       action=CheckSRU,
                       help="""The gate type to use in the hier RNNs""")
    group.add_argument('-mean_rnn_bi', action="store_true",
                       help="""Activate biRNN for hier model""")
    group.add_argument('-mean_hidden', type=int, default=600,
                       help='Size of hier rnn hidden states')
    

    # pretrain opts
    group.add_argument('-pretrain_emb', action="store_true",
                       help="""Activate biRNN for hier model""")
    group.add_argument('-pre_num_layers', type=int, default=12,
                       help='Size of hier rnn hidden states')
    group.add_argument('-pre_d_model', type=int, default=600,
                       help='Size of hier rnn hidden states')
    group.add_argument('-pre_heads', type=int, default=8,
                       help='Size of hier rnn hidden states')
    group.add_argument('-pre_d_ff', type=int, default=600,
                       help='Size of hier rnn hidden states')
    group.add_argument('-pre_dropout', type=float, default=0.1,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('-pre_max_relative_positions', type=int, default=0,
                       help='Size of hier rnn hidden states')
    group.add_argument('-pre_no_self_loss', action="store_true", default=True,
                       help="""Activate biRNN for hier model""")
    group.add_argument('-pre_only_neighbour_loss', action="store_true",
                       help="""Activate biRNN for hier model""")
    
    group.add_argument('-pre_loss_type', type=str, default="ranking",                 
                        choices=["nll", "ranking"],
                       help='row_self_attn_type')
    group.add_argument('-pre_hinge_thre', type=float, default=0.1,
                       help='Size of hier rnn hidden states')
    group.add_argument('-use_pretrain', action="store_true",
                       help="""Activate biRNN for hier model""")
    group.add_argument('-pretrain_model_path', type=str, default=None,
                       help='Size of hier rnn hidden states')
    group.add_argument('-fix_use_pretrain', action="store_true",
                       help="""Activate biRNN for hier model""")
    group.add_argument('-pre_use_mlp', action="store_true",
                       help="""Activate biRNN for hier model""")

    group.add_argument('-val_use_pretrain', action="store_true",
                       help="""Activate biRNN for hier model""")
    group.add_argument('-ent_use_pretrain', action="store_true",
                       help="""Activate biRNN for hier model""")
    group.add_argument('-ent_pretrain_model_path', type=str, default=None,
                       help='Size of hier rnn hidden states')
    group.add_argument('-ent_fix_use_pretrain', action="store_true",
                       help="""Activate biRNN for hier model""")
    group.add_argument('-ent_pre_use_mlp', action="store_true",
                       help="""Activate biRNN for hier model""")


    # pretrain ent
    group.add_argument('-pretrain_emb_ent', action="store_true",
                       help="""Activate biRNN for hier model""")
    group.add_argument('-pre_ent_type', type=str, default=None,                     choices=["bin", "rank"],
                       help='row_self_attn_type')
    group.add_argument('-pre_ent_use_data', action="store_true",
                       help="""Activate biRNN for hier model""")
                 
    group.add_argument('-save_best', action="store_true",
                       help="""Activate biRNN for hier model""")

    # ablation opts
    group.add_argument('-norow', action="store_true",
                       help="""Activate biRNN for hier model""")  
    group.add_argument('-nocolumn', action="store_true",
                       help="""Activate biRNN for hier model""")    
    group.add_argument('-nohierstructure', action="store_true",
                       help="""Activate biRNN for hier model""")    
    group.add_argument('-nofusion', action="store_true",
                       help="""Activate biRNN for hier model""") 
    group.add_argument('-nohistpos', action="store_true",
                       help="""Activate biRNN for hier model""")   
    group.add_argument('-cohan18', action="store_true",
                       help="""Activate biRNN for hier model""") 
    group.add_argument('-cohan18_hidden_size', type=int, default=1200,
                       help='Size of hier rnn hidden states')

    # char options
    group.add_argument('-char_enable', action="store_true",
                       help="""Activate character-level embedding for value""")
    group.add_argument('-char_vec_size', type=int, default=50,
                       help='Word embedding size for char.')
    group.add_argument('-char_rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       action=CheckSRU,
                       help="""The gate type to use in the RNNs""")
    group.add_argument('-char_rnn_size', type=int, default=50,
                       help='Size of rnn hidden states')
    group.add_argument('-char_layers', type=int, default=1,
                       help='Number of layers for char encoder')
    group.add_argument('-char_dropout', type=float, default=0.3,
                       help="Dropout probability; applied in char LSTM stacks.")
    group.add_argument('-char_bidirectional', action="store_true",
                       help="""Activate character-level embedding for biRNN""")

    # content selection gate
    group.add_argument('-nocsgate', action="store_true",
                       help="""No content selection gate""")    

    group.add_argument('-share_decoder_embeddings', action='store_true',
                       help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
    group.add_argument('-share_embeddings', action='store_true',
                       help="""Share the word embeddings between encoder
                       and decoder. Need to use shared dictionary for this
                       option.""")
    group.add_argument('-position_encoding', action='store_true',
                       help="""Use a sin to mark relative words positions.
                       Necessary for non-RNN style models.
                       """)
    group.add_argument('-only_pos', action='store_true',
                       help="""Use a sin to mark relative words positions.
                       Necessary for non-RNN style models.
                       """)
    group.add_argument('-pos_type', type=str, default="sincos", choices=["sincos", "posEmb"],
                       help='row_self_attn_type')

    group = parser.add_argument_group('Model-Embedding Features')
    group.add_argument('-feat_merge', type=str, default='concat',
                       choices=['concat', 'sum', 'mlp'],
                       help="""Merge action for incorporating features embeddings.
                       Options [concat|sum|mlp].""")
    group.add_argument('-feat_vec_size', type=int, default=-1,
                       help="""If specified, feature embedding sizes
                       will be set to this. Otherwise, feat_vec_exponent
                       will be used.""")
    group.add_argument('-feat_vec_exponent', type=float, default=0.7,
                       help="""If -feat_merge_size is not set, feature
                       embedding sizes will be set to N^feat_vec_exponent
                       where N is the number of values the feature takes.""")

    # Encoder-Deocder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add_argument('-model_type', default='text',
                       help="""Type of source model to use. Allows
                       the system to incorporate non-text inputs.
                       Options are [text|img|audio].""")

    group.add_argument('-basicencdec', action="store_true",
                       help="""Activate basic encoder-decoder model""")

    group.add_argument('-mixrankandhist', action="store_true",
                       help="""Activate mixture of rank and history""")

    # hist scalar map
    group.add_argument('-hist_scalar_map', action="store_true",
                       help="""Activate history scalar map""")

    group.add_argument('-num_interval', type=int, default=5,
                       help="""If -hist_scalar_map is set, num interval will be used.""")

    group.add_argument('-hist_scalar_map_enable_mlp', action="store_true",
                       help="""Activate mlp layer for scalar map""")

    group.add_argument('-encoder_type1', type=str, default='rnn',
                       choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                       help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn].""")
    group.add_argument('-decoder_type1', type=str, default='rnn',
                       choices=['rnn', 'transformer', 'cnn', 'pointer'],
                       help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|transformer|cnn].""")
    group.add_argument('-encoder_type2', type=str, default='rnn',
                       choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                       help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn].""")
    group.add_argument('-decoder_type2', type=str, default='rnn',
                       choices=['rnn', 'transformer', 'cnn', 'pointer'],
                       help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|transformer|cnn].""")

    group.add_argument('-layers', type=int, default=-1,
                       help='Number of layers in enc/dec.')
    group.add_argument('-enc_layers1', type=int, default=2,
                       help='Number of layers in the encoder')
    group.add_argument('-dec_layers1', type=int, default=2,
                       help='Number of layers in the decoder')
    group.add_argument('-enc_layers2', type=int, default=2,
                       help='Number of layers in the encoder')
    group.add_argument('-dec_layers2', type=int, default=2,
                       help='Number of layers in the decoder')
    group.add_argument('-rnn_size', type=int, default=500,
                       help='Size of rnn hidden states')
    group.add_argument('-cnn_kernel_width', type=int, default=3,
                       help="""Size of windows in the cnn, the kernel_size is
                       (cnn_kernel_width, 1) in conv layer""")

    group.add_argument('-input_feed', type=int, default=1,
                       help="""Feed the context vector at each time step as
                       additional input (via concatenation with the word
                       embeddings) to the decoder.""")
    group.add_argument('-bridge', action="store_true",
                       help="""Have an additional layer between the last encoder
                       state and the first decoder state""")
    group.add_argument('-rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'SRU'],
                       action=CheckSRU,
                       help="""The gate type to use in the RNNs""")
    # group.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    group.add_argument('-brnn', action=DeprecateAction,
                       help="Deprecated, use `encoder_type`.")
    group.add_argument('-brnn_merge', default='concat',
                       choices=['concat', 'sum'],
                       help="Merge action for the bidir hidden states")

    group.add_argument('-context_gate', type=str, default=None,
                       choices=['source', 'target', 'both'],
                       help="""Type of context gate to use.
                       Do not select for no context gate.""")

    # Attention options
    group = parser.add_argument_group('Model- Attention')
    group.add_argument('-global_attention', type=str, default='general',
                       choices=['dot', 'general', 'mlp'],
                       help="""The attention type to use:
                       dotprod or general (Luong) or MLP (Bahdanau)""")

    # Genenerator and loss options.
    group.add_argument('-copy_attn', action="store_true",
                       help='Train copy attention layer.')
    group.add_argument('-copy_attn_force', action="store_true",
                       help='When available, train to copy.')
    group.add_argument('-reuse_copy_attn', action="store_true",
                       help="Reuse standard attention for copy")
    group.add_argument('-copy_loss_by_seqlength', action="store_true",
                       help="Divide copy loss by length of sequence")
    group.add_argument('-coverage_attn', action="store_true",
                       help='Train a coverage attention layer.')
    group.add_argument('-lambda_coverage', type=float, default=1,
                       help='Lambda value for coverage.')

    group.add_argument('-stage1', action="store_true",
                       help="Stage1 pre process")


def preprocess_opts(parser):
    # Data options
    group = parser.add_argument_group('Data')

    group.add_argument('-mlb', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-marginal', action='store_true',
                       help="codes on mlb dataset")


    group.add_argument('-s2_multi_attn', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_multi_copy', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_gate', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_shuffle', action='store_true',
                       help="codes on mlb dataset")

    group.add_argument('-ent_use_data', action='store_true',
                       help="codes on mlb dataset")


    group.add_argument('-data_type', default="text",
                       help="""Type of the source input.
                       Options are [text|img].""")
    group.add_argument('-train',
                       help="Path to the training data")
    group.add_argument('-valid',
                       help="Path to the validation data")

    group.add_argument('-mode', required=True,
                        choices=['pretrain', 'textgen'],
                       help="Path to the training source data")
    group.add_argument('-pretrain_vocab', default=None,
                       help="Path to the training source data")
    group.add_argument('-pretrain_ent_vocab', default=None,
                       help="Path to the training source data")

    group.add_argument('-train_src1', required=True,
                       help="Path to the training source data")
    group.add_argument('-train_src1_pretrain', default=None,
                       help="Path to the training source data")
    group.add_argument('-train_src1_pretrain_ent', default=None,
                       help="Path to the training source data")

    group.add_argument('-train_src1_hist', default=None,
                       help="Path to the train_src1_hist")

    group.add_argument('-train_tgt1', required=True,
                       help="Path to the training target data")
    group.add_argument('-valid_src1', required=True,
                       help="Path to the validation source data")
    group.add_argument('-valid_src1_pretrain', default=None,
                       help="Path to the training source data")
    group.add_argument('-valid_src1_pretrain_ent', default=None,
                       help="Path to the training source data")

    group.add_argument('-valid_src1_hist', default=None,
                       help="Path to the valid_src1_hist")

    group.add_argument('-valid_tgt1', required=True,
                       help="Path to the validation target data")

    group.add_argument('-train_src2', required=True,
                       help="Path to the training source data")
    group.add_argument('-train_tgt2', required=True,
                       help="Path to the training target data")
    group.add_argument('-valid_src2', required=True,
                       help="Path to the validation source data")
    group.add_argument('-valid_tgt2', required=True,
                       help="Path to the validation target data")

    group.add_argument('-train_ptr', default=None,
                       help="Path to the training pointers data")
    group.add_argument('-train_ent_freq', default=None,
                       help="Path to the training pointers data")

    group.add_argument('-train_pretrain_src', default=None,
                       help="Path to the training pointers data")
    group.add_argument('-train_pretrain_tgt', default=None,
                       help="Path to the training pointers data")
    group.add_argument('-train_pretrain_data', default=None,
                       help="Path to the training pointers data")   
    group.add_argument('-train_pretrain_data_idx', default=None,
                       help="Path to the training pointers data")                            

    group.add_argument('-valid_pretrain_src', default=None,
                       help="Path to the training pointers data")
    group.add_argument('-valid_pretrain_tgt', default=None,
                       help="Path to the training pointers data")
    group.add_argument('-valid_pretrain_data', default=None,
                       help="Path to the training pointers data")   
    group.add_argument('-valid_pretrain_data_idx', default=None,
                       help="Path to the training pointers data")   

    group.add_argument('-src_dir', default="",
                       help="Source directory for image or audio files.")

    group.add_argument('-save_data', required=True,
                       help="Output file for the prepared data")

    group.add_argument('-max_shard_size', type=int, default=0,
                       help="""For text corpus of large volume, it will
                       be divided into shards of this size to preprocess.
                       If 0, the data will be handled as a whole. The unit
                       is in bytes. Optimal value should be multiples of
                       64 bytes.""")

    group.add_argument('-players_per_team', type=int, default=13,
                       help="""Max players per team""")

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab',
                       help="Path to an existing source vocabulary")
    group.add_argument('-tgt_vocab',
                       help="Path to an existing target vocabulary")
    group.add_argument('-features_vocabs_prefix', type=str, default='',
                       help="Path prefix to existing features vocabularies")
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    group.add_argument('-src_words_min_frequency', type=int, default=0)
    group.add_argument('-tgt_words_min_frequency', type=int, default=0)

    group.add_argument('-dynamic_dict', action='store_true',
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-src_seq_length', type=int, default=50,
                       help="Maximum source sequence length")
    group.add_argument('-pretrain_src_seq_length', type=int, default=50,
                       help="Maximum source sequence length")
    group.add_argument('-pretrain_src_data_seq_length', type=int, default=100000,
                       help="Maximum source sequence length")
    group.add_argument('-src_seq_length_trunc', type=int, default=0,
                       help="Truncate source sequence length.")
    group.add_argument('-src_data_seq_length_trunc', type=int, default=0,
                       help="Truncate source sequence length.")
    group.add_argument('-tgt_seq_length', type=int, default=50,
                       help="Maximum target sequence length to keep.")
    group.add_argument('-pretrain_tgt_seq_length', type=int, default=100000,
                       help="Maximum target sequence length to keep.")
    group.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                       help="Truncate target sequence length.")
    group.add_argument('-lower', action='store_true', help='lowercase data')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('-shuffle', type=int, default=1,
                       help="Shuffle data")
    group.add_argument('-seed', type=int, default=3435,
                       help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100000,
                       help="Report status every this many sentences")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help="Window size for spectrogram in seconds.")
    group.add_argument('-window_stride', type=float, default=.01,
                       help="Window stride for spectrogram in seconds.")
    group.add_argument('-window', default='hamming',
                       help="Window type for spectrogram generation.")


def train_opts(parser):
    # Model loading/saving options

    group = parser.add_argument_group('General')
    group.add_argument('-data', required=True,
                       help="""Path prefix to the ".train.pt" and
                       ".valid.pt" file path from preprocess.py""")

    # group.add_argument('-mlb', action='store_true',
    #                    help="codes on mlb dataset")
    # group.add_argument('-marginal', action='store_true',
    #                    help="codes on mlb dataset")

    group.add_argument('-val_pretrain_data', default=None,
                       help="""Path prefix to the ".train.pt" and
                       ".valid.pt" file path from preprocess.py""")
    group.add_argument('-ent_pretrain_data', default=None,
                       help="""Path prefix to the ".train.pt" and
                       ".valid.pt" file path from preprocess.py""")

    group.add_argument('-save_model', default='model',
                       help="""Model filename (the model will be saved as
                       <save_model>_epochN_PPL.pt where PPL is the
                       validation perplexity""")


    # Reinforce
    group.add_argument('-finetunefromstart', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")
    group.add_argument('-reinforce', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")
    group.add_argument('-r_join_loss', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")
    group.add_argument('-debug', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")
    group.add_argument('-no_repetition', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")

    # group.add_argument('-r_topk_sample', action="store_true",
    #                    help="""enable reinforcement learning for stage 1.""")
    group.add_argument('-r_topk_sample', type=int, default=2,
                       help="""max length for sampling""")

    group.add_argument('-r_rein_weight', type=float, default=0.5,
                       help="""join weight""")
    group.add_argument('-r_rein_not_weight_decay', type=float, default=1.0,
                       help="""not reinforce weight decay""")
    group.add_argument('-r_max_length', type=int, default=50,
                       help="""max length for sampling""")

    group.add_argument('-r_pos_rwd', type=float, default=1.0,
                       help="""join weight""")
    group.add_argument('-r_neg_rwd', type=float, default=-1.0,
                       help="""join weight""")
    group.add_argument('-r_accu_rwd', type=float, default=None,
                       help="""join weight""")
    group.add_argument('-r_order_rwd', type=float, default=1.0,
                       help="""join weight""")
    group.add_argument('-r_recall_beta', type=float, default=1.0,
                       help="""join weight""")

    group.add_argument('-r_gamma', type=float, default=0.8,
                       help="""join weight""")
    group.add_argument('-rwd_weight1', type=float, default=0.2,
                       help="""join weight""")
    group.add_argument('-rwd_weight2', type=float, default=0.2,
                       help="""join weight""")
    group.add_argument('-rwd_weight3', type=float, default=0.2,
                       help="""join weight""")
    group.add_argument('-rwd_weight4', type=float, default=0.2,
                       help="""join weight""")
    group.add_argument('-rwd_weight5', type=float, default=0.2,
                       help="""join weight""")

    group.add_argument('-r_ent_freq_threshold', type=float, default=1.0,
                       help="""join weight""")
    group.add_argument('-r_baseline', action="store_true",
                       help="""enable rl baseline""")
    group.add_argument('-r_only_baseline', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")


    group.add_argument('-sep_train', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")
    group.add_argument('-stage1_train', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")
    group.add_argument('-stage2_train', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")
    # GPU
    group.add_argument('-gpuid', default=[], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")

    group.add_argument('-seed', type=int, default=-1,
                       help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add_argument('-start_epoch', type=int, default=1,
                       help='The epoch from which to start')
    group.add_argument('-param_init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    group.add_argument('-train_from1', default=None, type=str,
                       help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")
    group.add_argument('-train_from2', default=None, type=str,
                       help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")

    # Pretrained word vectors
    group.add_argument('-pre_word_vecs_enc',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")
    group.add_argument('-pre_word_vecs_dec',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the decoder side.
                       See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add_argument('-fix_word_vecs_enc',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")
    group.add_argument('-fix_word_vecs_dec',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-batch_size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('-batch_type', default='sents',
                       choices=["sents", "tokens"],
                       help="""Batch grouping for batch_size. Standard
                               is sents. Tokens will do dynamic batching""")
    group.add_argument('-normalization', default='sents',
                       choices=["sents", "tokens"],
                       help='Normalization method of the gradient.')
    group.add_argument('-accum_count', type=int, default=1,
                       help="""Accumulate gradient this many times.
                       Approximately equivalent to updating
                       batch_size * accum_count batches at once.
                       Recommended for Transformer.""")
    group.add_argument('-valid_batch_size', type=int, default=32,
                       help='Maximum batch size for validation')
    group.add_argument('-max_generator_batches', type=int, default=32,
                       help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    group.add_argument('-epochs', type=int, default=13,
                       help='Number of training epochs')
    group.add_argument('-optim', default='sgd',
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                       help="""Optimization method.""")
    group.add_argument('-adagrad_accumulator_init', type=float, default=0,
                       help="""Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).
                       """)
    group.add_argument('-max_grad_norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
    group.add_argument('-dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('-truncated_decoder', type=int, default=0,
                       help="""Truncated bptt.""")
    group.add_argument('-adam_beta1', type=float, default=0.9,
                       help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add_argument('-adam_beta2', type=float, default=0.999,
                       help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
    group.add_argument('-label_smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-learning_rate', type=float, default=1.0,
                       help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add_argument('-learning_rate_decay', type=float, default=0.5,
                       help="""If update_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) epoch has gone past
                       start_decay_at""")
    group.add_argument('-start_decay_at', type=int, default=8,
                       help="""Start decaying every epoch after and including this
                       epoch""")
    group.add_argument('-start_checkpoint_at', type=int, default=0,
                       help="""Start checkpointing every epoch after and including
                       this epoch""")
    group.add_argument('-decay_method', type=str, default="",
                       choices=['noam'], help="Use a custom decay rate.")
    group.add_argument('-warmup_steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=50,
                       help="Print stats at this interval.")
    group.add_argument('-exp_host', type=str, default="",
                       help="Send logs to this crayon server.")
    group.add_argument('-exp', type=str, default="",
                       help="Name of the experiment for logging.")
    # Use TensorboardX for visualization during training
    group.add_argument('-tensorboard', action="store_true",
                       help="""Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.""")
    group.add_argument("-tensorboard_log_dir", type=str, default="runs/onmt",
                       help="""Log directory for Tensorboard.
                       This is also the name of the run.
                       """)

    group = parser.add_argument_group('Speech')
    # Options most relevant to speech
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help="Window size for spectrogram in seconds.")


def translate_opts(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('-mlb', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-marginal', action='store_true',
                       help="codes on mlb dataset")
    
    group.add_argument('-debug', action="store_true",
                       help="""enable reinforcement learning for stage 1.""")

    group.add_argument('-s2_multi_attn', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_multi_copy', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_gate', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_diff_gate', action='store_true',
                       help="codes on mlb dataset")
    group.add_argument('-s2_multi_attn_shuffle', action='store_true',
                       help="codes on mlb dataset")    

    group.add_argument('-model',
                       help='Path to model .pt file')
    group.add_argument('-model2',
                       help='Path to second model .pt file')

    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="Type of the source input. Options: [text|img].")

    group.add_argument('-src1',
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-src1_hist',
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-src1_pretrain', default=None,
                       help="Path to the training source data")
    group.add_argument('-src1_pretrain_ent', default=None,
                       help="Path to the training source data")
    
    group.add_argument('-src_dir',   default="",
                       help='Source directory for image or audio files')
    group.add_argument('-tgt1',
                       help='True target sequence (optional), required for two stage inference')
    group.add_argument('-src2',
                       help="""Second source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-tgt2',
                       help='True target sequence (optional)')
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")
    group.add_argument('-report_bleu', action='store_true',
                       help="""Report bleu score after translation,
                       call tools/multi-bleu.perl on command line""")
    group.add_argument('-report_rouge', action='store_true',
                       help="""Report rouge 1/2/3/L/SU4 score after translation
                       call tools/test_rouge.py on command line""")

    # Options most relevant to summarization.
    group.add_argument('-dynamic_dict', action='store_true',
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    group = parser.add_argument_group('Beam')
    group.add_argument('-beam_size',  type=int, default=5,
                       help='Beam size')
    group.add_argument('-min_length', type=int, default=0,
                       help='Minimum prediction length')
    group.add_argument('-max_length', type=int, default=100,
                       help='Maximum prediction length.')
    group.add_argument('-max_sent_length', action=DeprecateAction,
                       help="Deprecated, use `-max_length` instead")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument('-stepwise_penalty', action='store_true',
                       help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
    group.add_argument('-length_penalty', default='none',
                       choices=['none', 'wu', 'avg'],
                       help="""Length Penalty to use.""")
    group.add_argument('-coverage_penalty', default='none',
                       choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")
    group.add_argument('-alpha', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add_argument('-beta', type=float, default=-0.,
                       help="""Coverage penalty parameter""")
    group.add_argument('-replace_unk', action="store_true",
                       help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-verbose', action="store_true",
                       help='Print scores and predictions for each sentence')
    group.add_argument('-attn_debug', action="store_true",
                       help='Print best attn for each word')
    group.add_argument('-dump_beam', type=str, default="",
                       help='File to dump beam information to.')
    group.add_argument('-n_best', type=int, default=1,
                       help="""If verbose is set, will output the n_best
                       decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-batch_size', type=int, default=30,
                       help='Batch size')
    group.add_argument('-gpu', type=int, default=-1,
                       help="Device to run on")

    # Options most relevant to speech.
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help='Window size for spectrogram in seconds')
    group.add_argument('-window_stride', type=float, default=.01,
                       help='Window stride for spectrogram in seconds')
    group.add_argument('-window', default='hamming',
                       help='Window type for spectrogram generation')

    group.add_argument('-stage1', action="store_true",
                       help="Stage1 pre process")

def add_md_help_argument(parser):
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self)\
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class DeprecateAction(argparse.Action):
    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)
