import argparse
import sys

def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    #utils.parse_optimizer(parser)

    enc_parser.add_argument('--conv_type', type=str,
                        help='type of convolution')
    enc_parser.add_argument('--method_type', type=str,
                        help='type of embedding')
    enc_parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    enc_parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')

    enc_parser.add_argument('--feature_num', type=int,
                        help='Training node_feature size')
    enc_parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    enc_parser.add_argument('--skip', type=str,
                        help='"all" or "last"')
    enc_parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    enc_parser.add_argument('--margin', type=float,
                        help='margin for loss')
    enc_parser.add_argument('--dataset', type=str,
                        help='Dataset')
    enc_parser.add_argument('--test_set', type=str,
                        help='test set filename')
    enc_parser.add_argument('--eval_interval', type=int,
                        help='how often to eval during training')
    enc_parser.add_argument('--val_size', type=int,
                        help='validation set size')
    enc_parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                        help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                        help='whether to use node anchoring in training')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
        help='tag to identify the run')

    enc_parser.set_defaults(conv_type='GIN',
                        method_type='order',
                        dataset='',
                        n_layers=4,
                        feature_num=100,
                        batch_size=64,
                        hidden_dim=256,
                        skip="learnable",
                        dropout=0.0,
                        n_batches=100000,
                        # n_batches=1000000,
                        opt='adam',   # opt_enc_parser
                        opt_scheduler='none',
                        opt_restart=100,
                        weight_decay=0.0,
                        lr=5e-5,
                        margin=0.1,
                        test_set='',
                        eval_interval=100,
                        # eval_interval=100,
                        n_workers=4,
                        model_path="ckpt/model.pt",
                        tag='',
                        val_size=4096,
                        node_anchored=True)
