""" Generate commands to run a grid search for the fine-tuning step. """

import os, argparse
from copy import deepcopy

PATH_JOB_TEMPLATE = "jobs/template.job"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_pretrained_model", type=str)
    parser.add_argument("dir_train_data", type=str)
    parser.add_argument("dir_output_jobs", type=str)
    args = parser.parse_args()
    if os.path.exists(args.dir_output_jobs) and len(os.listdir(args.dir_output_jobs)):
        msg = "dir_output_jobs (%s) is not empty" % args.dir_output_jobs
        raise RuntimeError(msg)
    if not os.path.exists(args.dir_output_jobs):
        os.makedirs(args.dir_output_jobs)

    # Fixed params
    max_train_steps = "75000"
    train_batch_size = "16"
    seq_len = "256"
    sampling_distro = "dampfreq"
    score_to_optimize = "track1"
    
    # Tuned hparams
    settings = {}
    settings["grad_accum_steps"] = ["1", "2"]
    settings["learning_rate"] = ["1e-5", "2e-5", "3e-5"]
    hname_to_short_name = {"grad_accum_steps":"ac",
                           "learning_rate":"lr"}
    
    # Enumerate combinations of tuned hparam values
    configs = [{}]
    for k in settings.keys():
        augmented_configs = []
        for c in configs:
            for v in settings[k]:
                augmented = deepcopy(c)
                augmented[k] = v
                augmented_configs.append(augmented)
        configs = deepcopy(augmented_configs)
    print("\nNb configs: %d" % len(configs))
    
    # Get template for job files
    with open(PATH_JOB_TEMPLATE) as f:
        job_template = f.read().strip()
    
    # Write job files (assume working directory is vardial/bert, and both the pretrained model and the finetuned model are saved here)
    for i, config in enumerate(configs):
        # Encode settings in output directory path
        dir_output = "FT_exp3"
        for k in sorted(config.keys()):
            dir_output += "_%s=%s" % (hname_to_short_name[k], config[k])
        
        # Build command         
        cmd = "CUDA_VISIBLE_DEVICES=\"0\" python run_classifier.py"
        cmd += " --dir_pretrained_model %s" % os.path.relpath(args.dir_pretrained_model, "./bert")
        cmd += " --dir_data %s" % os.path.relpath(args.dir_train_data, "./bert")
        cmd += " --dir_output %s" % dir_output        
        cmd += " --do_train --eval_during_training"
        cmd += " --score_to_optimize %s" % score_to_optimize
        cmd += " --max_train_steps %s" % max_train_steps
        cmd += " --sampling_distro %s" % sampling_distro
        cmd += " --train_batch_size %s" % train_batch_size
        cmd += " --seq_len %s" % seq_len
        cmd += " --avgpool"

        # Add tuned hparams
        for k in sorted(config.keys()):
            cmd += " --%s %s" % (k, config[k])

        # Write
        path = os.path.join(args.dir_output_jobs, "ft.%02d.job" % (i+1))
        with open(path, 'w') as f:
            f.write(job_template)
            f.write("\n")
            f.write(cmd)
    print("%d job files written in %s.\n" % (len(configs), args.dir_output_jobs))
    
if __name__ == "__main__":
    main()
