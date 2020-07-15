""" Generate commands to run a grid search for the fine-tuning step. """

import os, argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_pretrained_model", type=str)
    parser.add_argument("max_train_steps", type=int)
    args = parser.parse_args()

    settings = {}
    settings["train_batch_size"] = ["16", "32"]
    settings["seq_len"] = ["128", "256"]
    settings["learning_rate"] = ["2e-5", "3e-5", "5e-5"]
    settings["grad_accum_steps"] = ["1", "2"]
    
    # Enumerate combinations
    combinations = [[]]
    for k in ["train_batch_size", "seq_len", "learning_rate", "grad_accum_steps"]:
        new_combinations = []
        for c in combinations:
            for v in settings[k]:
                new_combinations.append(c + [v])
        combinations = new_combinations[:]
    print("Nb combinations: %d" % len(combinations))
    
    # Check combinations
    filtered = []
    for (bs, sl, lr, gs) in combinations:
        if bs == "32" and sl == "256":
            continue
        filtered.append((bs, sl, lr, gs))
    combinations = filtered
    print("Nb combinations remaining after filtering: %d" % len(combinations))

    # Write commands (assume working directory is vardial/bert, and both the pretrained model and the finetuned model are saved here)
    for (bs, sl, lr, gs) in combinations:
        # Encode settings in output directory path
        dir_output = "Finetune_bs=%s_sl=%s_lr=%s_gs=%s" % (bs, sl, lr, gs)        

        # Build command
        cmd = "CUDA_VISIBLE_DEVICES=\"0\" python run_classifier.py"
        cmd += " --dir_pretrained_model %s" % os.path.relpath(args.dir_pretrained_model, "./bert")
        cmd += " --dir_data %s" % ("../data/finetuning_data_cut256" if sl == "256" else "../data/finetuning_data_cut128")
        cmd += " --do_train --eval_during_training"
        cmd += " --max_train_steps %d" % args.max_train_steps
        cmd += " --train_batch_size %s" % bs
        cmd += " --seq_len %s" % sl
        cmd += " --grad_accum_steps %s" % gs
        cmd += " --learning_rate %s" % lr
        cmd += " --dir_output %s" % dir_output
        print(cmd)
    
if __name__ == "__main__":
    main()
