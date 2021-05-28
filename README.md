# data2text-duv

This repo contains code for [Enhancing Content Planning for Table-to-Text Generation with Data Understanding and Verification](https://www.aclweb.org/anthology/2020.findings-emnlp.262/) (Gong, H., Bi, W., Feng, X., Qin, B., Liu, X., & Liu, T.; Findings of EMNLP 2020); this code is based on [data2text-plan-py](https://github.com/ratishsp/data2text-plan-py).


## Requirement

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

Note that requirements.txt contains necessary version requirements of certain dependencies, Python version is 3.6 and CUDA version is 10.1.

## Data and model

Before executing commands in the following sections, the data (preprocessed files) and/or trained model need to be downloaded and extracted. They are available as a single tar.gz file at link https://www.dropbox.com/s/kicxfpmg6o8pxoy/rotowire_orig.tar.gz?dl=0 (suited for global user) or https://pan.baidu.com/s/1ncUeE-1Gol3Squ_fwaF-dA (retrieval code: sede ). Please move the extracted folder rotowire_orig into this repo's folder.

## Training

The following command will train the model.

```
# pretrain command

BASE=/path/to/rotowire_orig
IDENTIFIER=f_pretrain_model
GPUID=0

OMP_NUM_THREADS=4 python -u train.py -data $BASE/preprocess/roto-two-stage-mlp-ent-app-pretrain-data-orig-two-cat-no-self-pretrain-stage -save_model $BASE/gen_model/$IDENTIFIER/roto -pretrain_emb -pre_hinge_thre 0.3 -pre_num_layers 2 -pre_d_model 600 -pre_heads 3 -pre_d_ff 1024 -pre_dropout 0.3 -dropout 0.3 -hier_meta $BASE/hier_meta.json -encoder_type1 mean -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 2 -dec_layers2 2 -batch_size 500 -feat_merge mlp -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -seed 1234 -start_checkpoint_at 1 -epochs 150 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 1000 -learning_rate 2 -report_every 100 -copy_attn -truncated_decoder 100 -gpuid $GPUID -attn_hidden 64 -reuse_copy_attn -valid_batch_size 10 -save_best

# stage 1 du command

BASE=/path/to/rotowire_orig
IDENTIFIER=sep_ncp_cc_du
EPOCHS=25
GPUID=0
STAGE=1
VALDATAPATH=$BASE/preprocess/roto-value-pretrain-orig
PRETRAIN_MODEL=/path/to/f_pretrain_model/file.pt

OMP_NUM_THREADS=4 python -u train.py -data $BASE/preprocess/roto-two-stage-mlp-ent-app-pretrain-data-orig-two-cat-no-self-pretrain-stage -save_model $BASE/gen_model/$IDENTIFIER/roto -sep_train -stage1_train -use_pretrain -val_use_pretrain -pretrain_model_path $PRETRAIN_MODEL -fix_use_pretrain -val_pretrain_data $VALDATAPATH -hier_meta $BASE/hier_meta.json -encoder_type1 mean -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 2 -dec_layers2 2 -batch_size 5 -feat_merge mlp -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -seed 1234 -start_checkpoint_at 4 -epochs $EPOCHS -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -report_every 100 -copy_attn -truncated_decoder 100 -gpuid $GPUID -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size 5

# stage 1 duv command (ran after stage 1 du command)

BASE=/path/to/rotowire_orig
IDENTIFIER=sep_ncp_cc_du
GPUID=0
STAGE=1
REIN_IDENTIFIER=duv_model
REIN_MODELNAME=roto_stage1_acc_73.0190_ppl_3.8344_e25.pt
EPOCHS=50
VALDATAPATH=$BASE/preprocess/roto-value-pretrain-orig
PRETRAIN_MODEL=/path/to/f_pretrain_model/file.pt

OMP_NUM_THREADS=4 python -u train.py -no_repetition -data $BASE/preprocess/roto-two-stage-mlp-ent-app-pretrain-data-orig-two-cat-no-self-pretrain-stage -save_model $BASE/gen_model/$IDENTIFIER/$REIN_IDENTIFIER/roto -sep_train -stage1_train -use_pretrain -val_use_pretrain -pretrain_model_path $PRETRAIN_MODEL -fix_use_pretrain -val_pretrain_data $VALDATAPATH -hier_meta $BASE/hier_meta.json -train_from1 $BASE/gen_model/$IDENTIFIER/$REIN_MODELNAME -reinforce -r_join_loss -r_rein_weight 0.3 -r_rein_not_weight_decay 0.97 -r_max_length 70 -r_pos_rwd 1.0 -r_neg_rwd -1.0 -r_order_rwd 1.0 -r_recall_beta 0.2 -r_gamma 0.9 -rwd_weight1 0.25 -rwd_weight2 0.15 -rwd_weight3 0.2 -rwd_weight4 0.1 -rwd_weight5 0.3 -encoder_type1 mean -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 2 -dec_layers2 2 -batch_size 5 -feat_merge mlp -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -seed 1234 -start_checkpoint_at 4 -epochs $EPOCHS -optim adagrad -learning_rate 0.07 -adagrad_accumulator_init 0.1 -report_every 100 -copy_attn -truncated_decoder 100 -gpuid $GPUID -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size 5

# stage 2 command

BASE=/path/to/rotowire_orig
IDENTIFIER=sep_ncp_cc_du
EPOCHS=25
GPUID=0
STAGE=2
VALDATAPATH=$BASE/preprocess/roto-value-pretrain-orig
PRETRAIN_MODEL=/path/to/f_pretrain_model/file.pt

OMP_NUM_THREADS=4 python -u train.py -data $BASE/preprocess/roto-two-stage-mlp-ent-app-pretrain-data-orig-two-cat-no-self-pretrain-stage -save_model $BASE/gen_model/$IDENTIFIER/roto -sep_train -stage2_train -use_pretrain -val_use_pretrain -pretrain_model_path $PRETRAIN_MODEL -fix_use_pretrain -val_pretrain_data $VALDATAPATH -hier_meta $BASE/hier_meta.json -encoder_type1 mean -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 2 -dec_layers2 2 -batch_size 5 -feat_merge mlp -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -seed 1234 -start_checkpoint_at 4 -epochs $EPOCHS -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -report_every 100 -copy_attn -truncated_decoder 100 -gpuid $GPUID -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size 5 


```

## Translate

The following command will generate on test set given trained model.

```
BASE=/path/to/rotowire_orig
MODEL1PATH=$BASE/gen_model/duv_model/roto_stage1_acc_72.9012_ppl_7.4952_e43.pt
MODEL2PATH=$BASE/gen_model/duv_model/roto_stage2_acc_58.5961_ppl_7.5971_e23.pt
GPUID=0

# stage 1

python translate.py -model $MODEL1PATH -src1 $BASE/inf_src_test.txt -src1_pretrain $BASE/pretrain_usage/inf_src_test.pretrain.pickle -output $BASE/gen/stage1_results.txt -batch_size 10 -max_length 80 -gpu $GPUID -min_length 35 -stage1 -src1_hist $BASE/hist_full/inf_src_test_hist_3.txt

python2 scripts/create_content_plan_from_index.py $BASE/inf_src_test.txt $BASE/gen/stage1_results.txt $BASE/gen/stage1_results_transform.txt $BASE/gen/stage1_results_inter.txt


# stage 2

python translate.py -model $MODEL1PATH -model2 $MODEL2PATH -src1 $BASE/inf_src_test.txt -tgt1 $BASE/gen/stage1_results.txt -src2 $BASE/gen/stage1_results_inter.txt -output $BASE/gen/stage2_results.txt -batch_size 10 -max_length 850 -gpu $GPUID -min_length 150 -src1_pretrain $BASE/pretrain_usage/inf_src_test.pretrain.pickle -src1_hist $BASE/hist_full/inf_src_test_hist_3.txt

```

## Evaluation

The following command will produce BLEU metric of model's generation on test set. The evaluation script can be obtained from link https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl

```
perl ref/multi-bleu.perl ref/test.txt < ./paper_model/model_test.txt
```

As for obtaining extractive evaluation metrics, please refer to https://github.com/ratishsp/data2text-1 for details.