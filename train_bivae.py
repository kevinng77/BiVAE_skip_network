import cornac
import torch
from models.Modify_bivaecf import BiVAECF
import pandas as pd
from collections import defaultdict
import numpy as np
from utils import cornac_utils
from config import config


def parameter_tuning(df_train,df_probe):
    data = cornac.eval_methods.base_method.BaseMethod(
        rating_threshold=2.0).from_splits(df_train.values,df_probe.values)
    B = [128]
    K = [64]
    D = [[32, 32], [32], [64], [64, 64]]
    E = [500, 850]
    A = ['tanh']
    lr = [1e-1, 1e-2, 1e-3, 1e-4]
    from models.Modify_bivaecf import BiVAECF
    from models import ModifyBivae
    metrics = [cornac.metrics.NDCG(k=50), cornac.metrics.Recall(k=50)]
    count = 0
    for epochs in E:
        for learning_rate in lr:
            for d2 in D:
                count += 1
                print("==" * 50)
                print(f"d{d2}_epoch{epochs}_lr{learning_rate}")
                model = BiVAECF(
                    k=64,
                    encoder_structure=[128],
                    decoder_structure=d2,
                    act_fn='tanh',
                    n_epochs=epochs,
                    batch_size=128,
                    learning_rate=learning_rate,
                    likelihood='pois',
                    beta_kl=1,
                    name=f"{d2}_{epochs}_{learning_rate}",
                    seed=7,
                    use_gpu=torch.cuda.is_available(),
                    verbose=True
                ).fit(data.train_set)
                ND, REC = cornac.eval_methods.base_method.ranking_eval(
                    model, metrics, data.train_set, data.test_set, verbose=0)[0]
                log = f"{model.name}\tNCDG {ND:.4f}\tRECALL {REC:.4f}\n"
                print(f"({count}/16)\t", log, "finished")
                with open("my_record.txt","a")as fp:
                    fp.write(log)


if __name__ == "__main__":
    print(config.raw_data_path)
    print(config.dev_data_path)
    df_train, train_data = cornac_utils.gen_cornac_dataset(config.raw_data_path)
    df_probe, dev_data = cornac_utils.gen_cornac_dataset(config.dev_data_path)
    # config.threshold = 50 default
    data_all = np.concatenate([df_train.values, df_probe.values])

    metrics = [cornac.metrics.NDCG(k=config.threshold),
               cornac.metrics.Recall(k=config.threshold)]
    data = cornac.eval_methods.base_method.BaseMethod(
        rating_threshold=config.true_threshold
    ).from_splits(data_all,df_probe.values)  # use data_all is owning to leaderboard submission.

    print(f"trainning: k:{config.k}\tlr{config.lr}\tepoch{config.epochs}\t"
          f"batch_size{config.batch_size}\t"
          f"encoder:{config.encoder_structure}\t" 
          f"decoder:{config.decoder_structure}" )
    model = BiVAECF(
        k=config.k,
        encoder_structure=config.encoder_structure,
        decoder_structure=config.decoder_structure,
        act_fn='tanh',
        n_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.lr,
        likelihood='pois',
        name=f"BiVAE_skipnet",
        seed=config.SEED,
        use_gpu=torch.cuda.is_available(),
        verbose=config.VERBOSE,
        true_threshold = config.true_threshold
    ).fit(data.train_set)
    # ND, REC = cornac.eval_methods.base_method.ranking_eval(
    #     model, metrics, data.train_set, data.test_set, verbose=0)[0]
    # log = f"k:{config.k}_lr{config.lr}_epoch{config.epochs}\t" \
    #       f"encoder:{config.encoder_structure}\t" \
    #       f"decoder:{config.decoder_structure}\t" \
    #       f"NCDG {ND:.4f}\tRECALL {REC:.4f}\n"
    # with open("my_record.txt", "a")as fp:
    #     fp.write(log)
    file = f"./checkout/recommend/{model.name}_{config.true_threshold}"
    model.gen_recommend(file)
    # print('recommend file saved to',file)
    # log_dir = f"./checkout/model"
    # state = {'model': model.bivae.state_dict()}
    # torch.save(state,log_dir)
    # checkpoint = torch.load(log_dir)
    # model.bivae.load_state_dict(checkpoint['model'])
"""
    ////////////////////////////////////////////////////////////////////
    //                          _ooOoo_                               //
    //                         o8888888o                              //
    //                         88" . "88                              //
    //                         (| ^_^ |)                              //
    //                         O\  =  /O                              //
    //                      ____/`---'\____                           //
    //                    .'  \\|     |//  `.                         //
    //                   /  \\|||  :  |||//  \                        //
    //                  /  _||||| -:- |||||-  \                       //
    //                  |   | \\\  -  /// |   |                       //
    //                  | \_|  ''\---/''  |   |                       //
    //                  \  .-\__  `-`  ___/-. /                       //
    //                ___`. .'  /--.--\  `. . ___                     //
    //              ."" '<  `.___\_<|>_/___.'  >'"".                  //
    //            | | :  `- \`.;`\ _ /`;.`/ - ` : | |                 //
    //            \  \ `-.   \_ __\ /__ _/   .-` /  /                 //
    //      ========`-.____`-.___\_____/___.-`____.-'========         //
    //                           `=---='                              //
    //      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        //
    //             佛祖保佑       永无BUG     光速炼丹                 //
    ////////////////////////////////////////////////////////////////////
"""
