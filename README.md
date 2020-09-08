Code developed for the Uralic language identification shared task at the VarDial 2020 Evaluation Campaign.

# Usage

Make train/dev/test split:

```bash
mkdir test
mkdir test/data
python make_train_dev_split.py --sampling_alpha 1.0 --weight_relevant 1.0 20000 100000 <path_to_ULI_training_data> test/data/split
```

Make vocab:

```bash
python get_vocab.py test/data/split/Training test/vocab.tsv
```

Pre-train model:

```bash
cd bert
CUDA_VISIBLE_DEVICES="0" python pretrain_BERT.py --bert_model_or_config_file bert_config.json --dir_train_data ../test/data/split/Training --path_vocab ../test/vocab.tsv --output_dir ../test/Pretrained_model --sampling_alpha 1.0 --weight_relevant 1.0 --seq_len 128 --min_freq 2 --max_train_steps 1000000 --num_train_steps_per_epoch 2000 --num_warmup_steps 10000 --learning_rate 1e-4 --seed 91500 --train_batch_size 32 --avgpool_for_spc --equal_betas --correct_bias
```

Fine-tune model:

```bash
CUDA_VISIBLE_DEVICES="0" python run_classifier.py --dir_pretrained_model ../test/Pretrained_model --dir_train ../test/data/split/Training --path_dev ../test/data/split/Test/dev-labeled.tsv --do_train --eval_during_training --max_train_steps 10000000 --num_train_steps_per_epoch 5000 --grad_accum_steps 1 --correct_bias --equal_betas --seed 91500 --seq_len 128 --learning_rate 3e-5 --train_batch_size 128 --no_mlm --sampling_alpha 0.75 --weight_relevant 2.0 --score_to_optimize track1 --dir_output ../test/Finetuned_model_track1
```

Get model predictions on test set:

```bash
CUDA_VISIBLE_DEVICES="0" python run_classifier.py --dir_pretrained_model ../test/Finetuned_model --path_test ../test/data/split/Test/test.txt --do_pred --dir_output ../test/Predictions 
```

Evaluate predictions:

```bash
cd ..
python scorer.py test/Predictions/pred.txt test/data/split/Test/test-gold-labels.txt
```
