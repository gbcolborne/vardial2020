""" Train or evaluate classifier. """

import argparse

def main():
    parser = argparse.ArgumentParser()
    # Model and data are required
    parser.add_argument("--dir_pretrained_model",
                        type=str,
                        required=True,
                        help="Dir containing pre-trained model")
    parser.add_argument("--dir_data",
                        type=str,
                        required=True,
                        help=("Dir containing data (train.tsv, valid.tsv and/or test.tsv) "
                              "Training and validation data must be in 2-column TSV format. "
                              "Test data may containg one or 2 columns."))
    # Execution modes
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Run training")
    parser.add_argument("--eval_during_training",
                        action="store_true",
                        help="Run evaluation on dev set during training")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Evaluate model on dev set")
    parser.add_argument("--do_pred",
                        action="store_true",
                        help="Run prediction on test set")
    args = parser.parse_args()
    
    # Check args
    assert args.do_train or args.do_eval or args.do_pred
    if args.eval_during_training:
        assert args.do_train
    if args.do_train:
        path_train_data = os.path.join(args.dir_data, "train.tsv")
        assert os.path.exists(path_train_data)
    if args.do_eval or args.eval_during_training:
        path_dev_data = os.path.join(args.dir_data, "valid.tsv")        
        assert os.path.exists(path_dev_data)
    if args.do_pred:
        path_test_data = os.path.join(args.dir_data, "test.tsv")        
        assert os.path.exists(path_test_data)
        
        
if __name__ == "__main__":
    main()
