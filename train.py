from rtdl_num_embeddings import LinearReLUEmbeddings
from rtdl_num_embeddings import PeriodicEmbeddings
from rtdl_num_embeddings import PiecewiseLinearEncoding
from rtdl_num_embeddings import PiecewiseLinearEmbeddings
from rtdl_num_embeddings import compute_bins
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import warnings
from model import FeatureProcessor, train, LSModel, LSBlock, LSAttention, MutationDataset


if __name__ == "__main__":
    train_df = pd.read_csv("./data/mis_clinvar_train.csv")
    test_df = pd.read_csv("./data/mis_clinvar_test.csv")
    train_df['True_Label'] = train_df['True_Label'].replace(-1, 0)
    test_df['True_Label'] = test_df['True_Label'].replace(-1, 0)
    columns_to_fill = [
        'Interpro_domain', 'DOMAINS',  'Ensembl_proteinid'
    ]
    for col in columns_to_fill:
        test_df[col] = test_df[col].fillna("unknown")
        train_df[col] = train_df[col].fillna("unknown")
    columns_to_fill = ["GDI", "GDI-Phred", "LoFtool_score", "RVIS_EVS","RVIS_percentile_EVS", "ExAC_pLI", "gnomAD_pLI", "SORVA_LOF_MAF0.005_HetOrHom", "SORVA_LOForMissense_MAF0.005_HetOrHom","gnomAD_pRec", "gnomAD_pNull", "Gene_indispensability_score","1000Gp3_AF", "ExAC_AF", "ExAC_Adj_AF", "gnomAD_genomes_AF",  "ALFA_Other_AF", "ALFA_Total_AF",
    "GERP++_NR", "GERP++_RS_rankscore", "GERP_91_mammals_rankscore","phyloP100way_vertebrate_rankscore", "phyloP470way_mammalian_rankscore", "phyloP17way_primate_rankscore", "phastCons100way_vertebrate_rankscore",
    "phastCons470way_mammalian_rankscore", "phastCons17way_primate_rankscore", "SiPhy_29way_logOdds_rankscore", "bStatistic_converted_rankscore","SIFT4G_converted_rankscore", "Polyphen2_HDIV_rankscore", "Polyphen2_HVAR_rankscore", "MutationTaster_converted_rankscore",
    "MutationAssessor_rankscore", "fathmm-XF_coding_rankscore", "PROVEAN_converted_rankscore", "VEST4_rankscore", "MetaSVM_rankscore", "MetaLR_rankscore", "M-CAP_rankscore","REVEL_rankscore","MutPred_rankscore", "DEOGEN2_rankscore",
    "Eigen-phred_coding","DANN_rankscore",      
    ]
    mode_values = train_df[columns_to_fill].mode().iloc[0]
    train_df[columns_to_fill] = train_df[columns_to_fill].fillna(mode_values)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    processor = FeatureProcessor(train_df)
    train_set = MutationDataset(train_df, processor)
    processor = FeatureProcessor(val_df)
    val_set = MutationDataset(val_df, processor)
    processor = FeatureProcessor(test_set)
    test_set = MutationDataset(test_df, processor)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size)
    model = train(train_loader, val_loader)
