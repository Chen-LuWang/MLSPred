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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def predict(model, test_loader):
    model.eval()
    preds = []
    total_correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for protein,gene_text, gene,freq, cons, fun_s, labels in test_loader:
            inputs = [t.to(Config.device) for t in [protein,gene_text, gene,freq, cons, fun_s]]
            labels = labels.to(Config.device)

            outputs = model(*inputs)
            preds.append(torch.softmax(outputs, dim=1).cpu())

            total_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.append(outputs.softmax(1)[:, 1].cpu().numpy())  # 假设是二分类问题
            all_labels.append(labels.cpu().numpy())
    print("len(test_loader.dataset):", len(test_loader.dataset))
    acc = total_correct / len(test_loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    print(f"| test Acc: {acc:.4f} | test AUROC: {auroc:.4f} | test AUPRC: {auprc:.4f}")
    return all_preds

if __name__ == "__main__":
    model_save_path = "./model/all_feat_model.pth"
    model = CrossFormerModel(
        protein_dim=Config.protein_embed_dim,
        gene_text_dim  =Config.protein_embed_dim,
        gene_dim=len(Config.gene_features),
        freq_dim=len(Config.freq_features),
        conservation_dim=len(Config.conservation_features),  
        function_dim =len(Config.function_features)
    ).to(Config.device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  
    
    datasets = [
        (r"C:\Users\A\PycharmProjects\my_paper\miss_predict\mis_PPARG_test.csv",
         r"C:\Users\A\PycharmProjects\my_paper\miss_predict\all_feat_model_score\scores-mis_PPARG_test.csv"),

        (r"C:\Users\A\PycharmProjects\my_paper\miss_predict\mis_TP53_test.csv",
         r"C:\Users\A\PycharmProjects\my_paper\miss_predict\all_feat_model_score\scores-mis_TP53_test.csv"),

        (r"C:\Users\A\PycharmProjects\my_paper\miss_predict\mis_cancer_DIShcover.csv",
         r"C:\Users\A\PycharmProjects\my_paper\miss_predict\all_feat_model_score\scores-mis_cancer_DIShcover.csv"),

        (r"C:\Users\A\PycharmProjects\my_paper\miss_predict\mis_lofgof_gnomad.csv",
         r"C:\Users\A\PycharmProjects\my_paper\miss_predict\all_feat_model_score\scores-mis_lofgof_gnomad.csv"),

        (r"C:\Users\A\PycharmProjects\my_paper\miss_predict\mis_swissport.csv",
         r"C:\Users\A\PycharmProjects\my_paper\miss_predict\all_feat_model_score\scores-mis_swissport.csv")

    ] 

    for input_csv, output_csv in datasets:
        test_df = pd.read_csv(input_csv)

        test_df['True_Label'] = test_df['True_Label'].replace(-1, 0)

        columns_to_fill = [
            'Interpro_domain', 'DOMAINS', 'GO_biological_process', 'GO_cellular_component',
            'GO_molecular_function', 'Pathway(ConsensusPathDB)', 'Pathway(KEGG)_id',
            'Pathway(KEGG)_full', 'Ensembl_proteinid'
        ]
        for col in columns_to_fill:
            test_df[col] = test_df[col].fillna("unknown")

        processor = FeatureProcessor(test_df)
        test_set = MutationDataset(test_df, processor)
        test_loader = DataLoader(test_set, batch_size=Config.batch_size)

        pd.DataFrame(predictions, columns=["Prob_1"]).to_csv(output_csv, index=False)
        print(f"save to {output_csv}")

