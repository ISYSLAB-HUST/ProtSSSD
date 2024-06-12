"""
"""
import os
import argparse
import torch
import esm
from ProtSSSD.models import SSpredictor
from ProtSSSD.utils import ProtMetrics
import logging
from torch.utils.data import DataLoader
from ProtSSSD.data import JSONDataset, casp_collate_fn
from torch.utils.checkpoint import checkpoint
from minlora import add_lora,merge_lora, LoRAParametrization
import numpy as np
from functools import partial
from einops import rearrange
import os
import random


#####################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('predict.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)
#####################################


def get_params():
    parser = argparse.ArgumentParser('ProtSSSD predict mode')

    parser.add_argument("data_dir", type=str,
                        default="./secondary_structure", help='')
    
    parser.add_argument("saved_model", type=str,
                        default="./saved_model", help='')
    
    parser.add_argument('-b', '--batch-size', type=int,
                        default=1, help='minibatch size (default: 1)')

    args, _ = parser.parse_known_args()
    return args

def data_iter(data_path, batch_size):
    casp15_path = os.path.join(data_path, "casp_15.json")
    test_data_paths = {'casp15': casp15_path}
    test_data_dict = {key: JSONDataset(path) for key, path in test_data_paths.items()}
    test_data_loaders = {key: DataLoader(value, batch_size, collate_fn=casp_collate_fn) 
                         for key, value in test_data_dict.items()}

    return test_data_loaders


def eval(pretrain_model, model, batch_converter, loader, device, process):
    metrics = ProtMetrics(ss3=3, ss8=8, device=device, disorder_infer=False)
    pretrain_model.eval()
    model.eval()
    with torch.no_grad():
        for (i, item) in enumerate(loader):
            reprs = dict()
            _, _, batch_tokens = batch_converter(item["seq"])
            # print(batch_tokens)
            exclude_keys = ["seq"]
            data_dict = {key: value.to(device) if key not in exclude_keys else value for key, value in item.items()} 
            batch_tokens = batch_tokens.to(device)
            if item["valid_mask"].sum() > 0:
                with torch.autocast(device_type="cuda"):
                    # results = pretrain_model(batch_tokens, repr_layers=[33], need_head_weights=True, return_contacts=True)
                    results = checkpoint(pretrain_model, batch_tokens, use_reentrant=False, 
                                         repr_layers=[33], need_head_weights=True, return_contacts=False)

                    reprs["single_repr"] = results["representations"][33][:,1:-1]
                    # contacts = results["contacts"]
                    attentions = rearrange(results["attentions"][:,-1,:,1:-1, 1:-1], 'b h i j -> b i j h')
                    # reprs["pair_repr"] = torch.cat((contacts.unsqueeze(-1), attentions), dim=-1)
                    reprs["pair_repr"] = attentions
                    predict = model(reprs, mask=None, num_recycle=3)
                step_info = metrics.update_step(predict, data_dict)
            else:
                print([record[0] for record in item["seq"]])
        total_info = metrics.compute()
        metrics.reset()
        logger.info("{}: SS3:{:.5f}, SS8:{:.5f}, phi_mae:{:.5f}, psi_mae:{:.5f}, rsa_pcc:{:.5f}".format(
                    process, total_info['ss3'], total_info['ss8'], total_info['phi_mae'], 
                    total_info['psi_mae'],total_info['rsa_pcc']))


def main(args):

    ###############
    saved_model = args["saved_model"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ###############

    ##############
    # setup_seed(args['seed'])
    ##############
    lora_config = {
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=3),
        },
    }

    logger.info("init model")
    # load params
    model_checkpoint = torch.load(saved_model, map_location='cpu')
    pretrain_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    add_lora(pretrain_model, lora_config=lora_config)
    _ = pretrain_model.load_state_dict(model_checkpoint['lora_state_dict'], strict=False)
    merge_lora(pretrain_model)
    pretrain_model = pretrain_model.to(device)
    # submodel
    model = SSpredictor(dim=1280, num_layers=3, n_hidden=64, pair_dim=20, dropout=0)
    model.load_state_dict(model_checkpoint['state_dict'])
    model = model.to(device)
    # -------------
    logger.info("load dataset....")
    test_loader = data_iter(args["data_dir"], args['batch_size'])


    for k, v in test_loader.items():
        eval(pretrain_model, model, batch_converter, v, device, k)

if __name__ == "__main__":
    try:
        # hyper_param = json.load(open("./best_hyper_paramter.json", 'r'))
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
