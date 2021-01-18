from u_net_pytorch.training import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str,
                        default="D:/pycharm_project/U_V_Net_tumour_paper/u_net/dataset/data_cv_clip/train")
    parser.add_argument('--label_dir', type=str,
                        default="D:/pycharm_project/U_V_Net_tumour_paper/u_net/dataset/data_cv_clip/label")
    # parser.add_argument('--validset_dir', type=str, default='./data/valid')
    # parser.add_argument('--result_dir', type=str, default='./Result')
    # parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=50)
    # 线程
    parser.add_argument('--num_works', type=int, default=1)
    # parser.add_argument('--num_works_test', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_steps', type=int, default=200, help='number of epochs to update learning rate')
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
    return parser.parse_args()


def main(cfg):
    trainset = DataFeeder(train_dir=cfg.train_dir, label_dir=cfg.label_dir)
    # train_loader = DataLoader(dataset=trainset, num_workers=cfg.num_works,
    #                           batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(dataset=trainset, num_workers=cfg.num_works,
                              batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    train(train_loader, cfg)


if __name__ == '__main__':
    configures = parse_args()
    main(configures)



