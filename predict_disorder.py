"""
"""
import os
import argparse
import torch
import esm
from ProtSSSD.models import SSpredictor
from ProtSSSD.utils import ProtMetrics
import logging
from torch.utils.data import DataLoader, Dataset
# from ProtSSSD.data import JSONDataset, casp_collate_fn

from torch.utils.checkpoint import checkpoint
from minlora import add_lora,merge_lora, LoRAParametrization
import numpy as np
from functools import partial
from einops import rearrange
import os
import random
import json

from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import AUROC


#####################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('predict_disorder.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)
#####################################


def get_params():
    parser = argparse.ArgumentParser('ProtSSSD disorder predict mode')

    parser.add_argument("data_dir", type=str, help='disorder dir')
    
    parser.add_argument("saved_model", type=str, help='')
    
    parser.add_argument('-b', '--batch-size', type=int,
                        default=1, help='minibatch size (default: 1)')
    
    parser.add_argument('-o', '--output', type=str, help='output')


    args, _ = parser.parse_known_args()
    return args

class FastaDaset(Dataset):
    def __init__(self, fasta_name,) -> None:
        super().__init__()
        self.data = SeqIO.to_dict(SeqIO.parse(fasta_name, format='fasta'))
        self.id_list = [i for i in self.data.keys()]
        self.str2label = {'1':0, '0':1, '-':-1}

    def __getitem__(self, index):
        record  = self.data[self.id_list[index]]
        seq_str = str(record.seq)
        protein_seq = seq_str[:int(len(seq_str)/2)]
        label_seq = list(seq_str[int(len(seq_str)/2):])
        label_seq = [self.str2label[i] for i in label_seq]

        return {'seq':protein_seq,'id':record.id, 'disorder':label_seq}
    
    def __len__(self):
        return len(self.id_list)
    
    def collate_fn(self, batch_record):
        tokens_batch = [(item["id"], item["seq"]) for item in batch_record]
        disorder_batch = pad_sequence([torch.tensor(item["disorder"], dtype=torch.int) for item in batch_record], batch_first=True)
        valid_mask = disorder_batch != -1

        return {"seq":tokens_batch, "disorder":disorder_batch, "valid_mask":valid_mask}

def data_iter(data_path, batch_size):

    pdb_path = os.path.join(data_path, "disorder_pdb.fasta")
    test_data_paths = {'pdb':pdb_path}
    test_data_dict = {key: FastaDaset(path) for key, path in test_data_paths.items()}
    test_data_loaders = {key: DataLoader(value, batch_size, collate_fn=value.collate_fn) 
                         for key, value in test_data_dict.items()}

    return test_data_loaders


def eval(pretrain_model, model, batch_converter, loader, device, process, output):
    # metrics = ProtMetrics(ss3=3, ss8=8, device=device, disorder_infer=False)
    auc_metrics = AUROC(task="binary").to(device)
    pretrain_model.eval()
    model.eval()
    output_dict = {}
    with torch.no_grad():
        for (i, item) in enumerate(loader):
            reprs = dict()
            _, _, batch_tokens = batch_converter(item["seq"])
            # print(batch_tokens)
            exclude_keys = ["seq"]
            data_dict = {key: value.to(device) if key not in exclude_keys else value for key, value in item.items()}
            valid_mask = data_dict['valid_mask']
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
                    predict = model(reprs, mask=None, num_recycle=0)
                step_info = auc_metrics(predict['disorder'][valid_mask], data_dict["disorder"][valid_mask],)
                output_dict[item["seq"][0][0]] = predict['disorder'].cpu().tolist()
        roc_auc_score = auc_metrics.compute()
        auc_metrics.reset()
        logger.info("{}: disorder AUC:{:.5f}".format(process, roc_auc_score))
        json.dump(output_dict, open(output, 'w'))



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
        eval(pretrain_model, model, batch_converter, v, device, k, args["output"])

if __name__ == "__main__":
    try:
        # hyper_param = json.load(open("./best_hyper_paramter.json", 'r'))
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
