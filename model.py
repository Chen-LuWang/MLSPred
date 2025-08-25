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

class Config:
    batch_size = 128
    lr = 2*1e-4
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model = 'monologg/biobert_v1.1_pubmed'
    max_token_len = 48
    protein_embed_dim = 768
    num_classes = 2
    gene_features = [
        "GDI", "GDI-Phred", "LoFtool_score", "RVIS_EVS","RVIS_percentile_EVS", "ExAC_pLI", "gnomAD_pLI", "SORVA_LOF_MAF0.005_HetOrHom", "SORVA_LOForMissense_MAF0.005_HetOrHom","gnomAD_pRec", "gnomAD_pNull", "Gene_indispensability_score"
    ]
    freq_features = [
        "1000Gp3_AF", "ExAC_AF", "ExAC_Adj_AF", "gnomAD_genomes_AF",  "ALFA_Other_AF", "ALFA_Total_AF"
    ]
    conservation_features = [
        "GERP++_NR", "GERP++_RS_rankscore", "GERP_91_mammals_rankscore","phyloP100way_vertebrate_rankscore", "phyloP470way_mammalian_rankscore", "phyloP17way_primate_rankscore", "phastCons100way_vertebrate_rankscore",
        "phastCons470way_mammalian_rankscore","phastCons17way_primate_rankscore", "SiPhy_29way_logOdds_rankscore", "bStatistic_converted_rankscore",
    ]
    function_features = [
        "SIFT4G_converted_rankscore", "Polyphen2_HDIV_rankscore", "Polyphen2_HVAR_rankscore", "MutationTaster_converted_rankscore", "MutationAssessor_rankscore", "fathmm-XF_coding_rankscore", "PROVEAN_converted_rankscore", "VEST4_rankscore",
        "MetaSVM_rankscore", "MetaLR_rankscore", "M-CAP_rankscore","REVEL_rankscore","MutPred_rankscore", "DEOGEN2_rankscore", "Eigen-phred_coding","DANN_rankscore",
    ]

class FeatureProcessor:
    def __init__(self, train_df):
        self._init_text_encoder()
        self._fit_numeric_transformers(train_df)

    def _init_text_encoder(self):
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_model)
        self.bert = BertModel.from_pretrained(Config.bert_model).to(Config.device)
        self.bert.eval()

    def _fit_numeric_transformers(self, df):
        self.gene_qt = QuantileTransformer(output_distribution='normal')
        self.gene_qt.fit(df[Config.gene_features].fillna(-1))
        self.gene_CRISPR = {'E': 1, 'N': -1, np.nan: -1}
        self.gene_CRISPR2 = {'E': 1,'S': 0,'N': -1, np.nan: -1}

    def process_row(self, row, is_train=True):
        protein = self._process_protein(row)
        # gene_text = self._process_gene(row)
        gene = row[Config.gene_features].copy()
        gene = self.gene_qt.transform(gene.fillna(-1).values.reshape(1, -1))
        freq = row[Config.freq_features].copy()
        median_values = row[[
            "GDI", "GDI-Phred", "LoFtool_score", "RVIS_EVS", "RVIS_percentile_EVS", "ExAC_pLI", "gnomAD_pLI",  "SORVA_LOF_MAF0.005_HetOrHom", "SORVA_LOForMissense_MAF0.005_HetOrHom", "gnomAD_pRec", "gnomAD_pNull",
            "Gene_indispensability_score",     "1000Gp3_AF", "ExAC_AF", "ExAC_Adj_AF", "gnomAD_genomes_AF",  "ALFA_Other_AF", "ALFA_Total_AF"
            "GERP++_NR","GERP++_RS_rankscore", "GERP_91_mammals_rankscore", "phyloP100way_vertebrate_rankscore", "phyloP470way_mammalian_rankscore", "phyloP17way_primate_rankscore", "phastCons100way_vertebrate_rankscore",
            "phastCons470way_mammalian_rankscore", "phastCons17way_primate_rankscore", "SiPhy_29way_logOdds_rankscore", "bStatistic_converted_rankscore",
            "SIFT4G_converted_rankscore", "Polyphen2_HDIV_rankscore", "Polyphen2_HVAR_rankscore","MutationTaster_converted_rankscore", "MutationAssessor_rankscore", "fathmm-XF_coding_rankscore", "PROVEAN_converted_rankscore","VEST4_rankscore",
            "MetaSVM_rankscore", "MetaLR_rankscore", "M-CAP_rankscore", "REVEL_rankscore", "MutPred_rankscore","DEOGEN2_rankscore","Eigen-phred_coding", "DANN_rankscore",
        ]].median()
        conservation = row[["GERP++_NR", "GERP++_RS_rankscore", "GERP_91_mammals_rankscore","phyloP100way_vertebrate_rankscore",
        "phyloP470way_mammalian_rankscore", "phyloP17way_primate_rankscore", "phastCons100way_vertebrate_rankscore", "phastCons470way_mammalian_rankscore","phastCons17way_primate_rankscore", "SiPhy_29way_logOdds_rankscore", "bStatistic_converted_rankscore",
        ]].fillna(median_values).values
        function_scores = row[[
            "SIFT4G_converted_rankscore", "Polyphen2_HDIV_rankscore", "Polyphen2_HVAR_rankscore", "MutationTaster_converted_rankscore",
            "MutationAssessor_rankscore", "fathmm-XF_coding_rankscore", "PROVEAN_converted_rankscore", "VEST4_rankscore", "MetaSVM_rankscore", "MetaLR_rankscore", "M-CAP_rankscore", "REVEL_rankscore", "MutPred_rankscore", "DEOGEN2_rankscore",
            "Eigen-phred_coding", "DANN_rankscore",
        ]].fillna(median_values).values
        return {
            'protein': torch.FloatTensor(protein),
            # 'gene_text': torch.FloatTensor(gene_text),
            'gene': torch.FloatTensor(gene.squeeze()),
            'freq':torch.FloatTensor(freq),
            'conservation': torch.FloatTensor(conservation.squeeze()),
            'function_scores': torch.FloatTensor(function_scores.squeeze()),
            'label': torch.tensor(row['True_Label']) if is_train else None
        }

    def _process_protein(self, row):
        def encode_interpro(text):
            texts = ['Interpro_domain'] + text.split(';')[:15] if isinstance(text, str) else []

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=Config.max_token_len,
                return_tensors='pt'
            ).to(Config.device)

            with torch.no_grad():
                outputs = self.bert(**inputs)


            return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()


        def encode_Ensembl_proteinid(text):
            texts = ['Ensembl_proteinid'] + text.split(';')[:15] if isinstance(text, str) else []
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=Config.max_token_len,
                return_tensors='pt'
            ).to(Config.device)
            with torch.no_grad():
                outputs = self.bert(**inputs)
            return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()

        def encode_domains(text):
            texts = ['DOMAINS'] + text.split(',')[:15] if isinstance(text, str) else []
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=Config.max_token_len,
                return_tensors='pt'
            ).to(Config.device)
            with torch.no_grad():
                outputs = self.bert(**inputs)
            return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
        domains = encode_domains(row['DOMAINS'])
        interpro = encode_interpro(row['Interpro_domain'])
        Ensembl_proteinid = encode_Ensembl_proteinid(row['Ensembl_proteinid'])
        domains = np.expand_dims(domains, axis=0)
        interpro = np.expand_dims(interpro, axis=0)  )
        Ensembl_proteinid = np.expand_dims(Ensembl_proteinid, axis=0)
        protein_embedding = np.concatenate([domains, interpro, Ensembl_proteinid], axis=0)
        return protein_embedding



    # def _process_gene(self, row):
    #     def encode_GO_biological_process(text):
    #         texts = ['GO_biological_process'] + text.split(';')[:25] if isinstance(text, str) else []
    #
    #         inputs = self.tokenizer(
    #             texts,
    #             padding=True,
    #             truncation=True,
    #             max_length=Config.max_token_len,
    #             return_tensors='pt'
    #         ).to(Config.device)
    #
    #         with torch.no_grad():
    #             outputs = self.bert(**inputs)
    #             # print(outputs.last_hidden_state.shape)
    #
    #         return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
    #
    #     def encode_GO_cellular_component(text):
    #         texts = ['GO_cellular_component'] + text.split(';')[:30] if isinstance(text, str) else []
    #         inputs = self.tokenizer(
    #             texts,
    #             padding=True,
    #             truncation=True,
    #             max_length=Config.max_token_len,
    #             return_tensors='pt'
    #         ).to(Config.device)
    #
    #         with torch.no_grad():
    #             outputs = self.bert(**inputs)
    #
    #         return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
    #
    #     def encode_GO_molecular_function(text):
    #         texts = ['GO_molecular_function'] + text.split(';')[:30] if isinstance(text, str) else []
    #         inputs = self.tokenizer(
    #             texts,
    #             padding=True,
    #             truncation=True,
    #             max_length=Config.max_token_len,
    #             return_tensors='pt'
    #         ).to(Config.device)
    #         with torch.no_grad():
    #             outputs = self.bert(**inputs)
    #         return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
    #
    #     def encode_Pathway_ConsensusPathDB(text):
    #         texts = ['Pathway(ConsensusPathDB)'] + text.split(';')[:30] if isinstance(text, str) else []
    #         inputs = self.tokenizer(
    #             texts,
    #             padding=True,
    #             truncation=True,
    #             max_length=Config.max_token_len,
    #             return_tensors='pt'
    #         ).to(Config.device)
    #         with torch.no_grad():
    #             outputs = self.bert(**inputs)
    #         return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
    #
    #     def encode_Pathway_KEGG_id(text):
    #         texts = ['Pathway(KEGG)_id'] + text.split(';')[:15] if isinstance(text, str) else []
    #         inputs = self.tokenizer(
    #             texts,
    #             padding=True,
    #             truncation=True,
    #             max_length=Config.max_token_len,
    #             return_tensors='pt'
    #         ).to(Config.device)
    #         with torch.no_grad():
    #             outputs = self.bert(**inputs)
    #         return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
    #
    #     def encode_Pathway_KEGG_full(text):
    #         texts = ['Pathway(KEGG)_full'] + text.split(';')[:15] if isinstance(text, str) else []
    #         inputs = self.tokenizer(
    #             texts,
    #             padding=True,
    #             truncation=True,
    #             max_length=Config.max_token_len,
    #             return_tensors='pt'
    #         ).to(Config.device)
    #         with torch.no_grad():
    #             outputs = self.bert(**inputs)
    #         return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
    #     GO_biological_process = encode_GO_biological_process(row['GO_biological_process'])
    #     GO_cellular_component = encode_GO_cellular_component(row['GO_cellular_component'])
    #     GO_molecular_function = encode_GO_molecular_function(row['GO_molecular_function'])
    #     Pathway_ConsensusPathDB = encode_Pathway_ConsensusPathDB(row['Pathway(ConsensusPathDB)'])
    #     Pathway_KEGG_id = encode_Pathway_KEGG_id(row['Pathway(KEGG)_id'])
    #     Pathway_KEGG_full = encode_Pathway_KEGG_full(row['Pathway(KEGG)_full'])
    #     GO_biological_process = np.expand_dims(GO_biological_process, axis=0)
    #     GO_cellular_component = np.expand_dims(GO_cellular_component, axis=0)
    #     GO_molecular_function = np.expand_dims(GO_molecular_function, axis=0)
    #     Pathway_ConsensusPathDB = np.expand_dims(Pathway_ConsensusPathDB, axis=0)
    #     Pathway_KEGG_id = np.expand_dims(Pathway_KEGG_id, axis=0)
    #     Pathway_KEGG_full = np.expand_dims(Pathway_KEGG_full, axis=0)
    #     gene_embedding = np.concatenate([GO_biological_process, GO_cellular_component, GO_molecular_function,Pathway_ConsensusPathDB,Pathway_KEGG_id,Pathway_KEGG_full], axis=0)
    #     return gene_embedding

# ==================== 数据集类 ====================
class MutationDataset(Dataset):
    def __init__(self, df, processor, is_train=True):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        data = self.processor.process_row(row, self.is_train)
        if self.is_train:
            return data['protein'],data['gene'],data['freq'], data['conservation'], data['function_scores'], data['label']
            # return data['protein'], data['gene_text'],data['gene'],data['freq'], data['conservation'], data['function_scores'], data['label']
        else:
            return data['protein'],data['gene'],data['freq'], data['conservation'], data['function_scores']
            # return data['protein'], data['gene_text'],data['gene'],data['freq'], data['conservation'], data['function_scores']


class LSAttention(nn.Module):
    def __init__(self, dim, num_heads, scales=[3, 5, 15]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True)
            for _ in range(len(scales))
        ])
        self.proj = nn.Linear(dim * len(scales), dim)

    def forward(self, x):
        B, L, C = x.shape
        outputs = []
        for scale, attn in zip(self.scales, self.attentions):
            if L % scale != 0:
                pad_size = scale - (L % scale)
                x_padded = F.pad(x, (0, 0, 0, pad_size))
                L_padded = L + pad_size
            else:
                x_padded = x
                L_padded = L


            x_reshaped = x_padded.reshape(B, L_padded // scale, scale, C)   #
            x_reshaped = x_reshaped.reshape(-1, scale, C)  

            out, _ = attn(x_reshaped, x_reshaped, x_reshaped)


            out = out.reshape(B, L_padded // scale, scale, C)
            out = out.reshape(B, L_padded, C)[:, :L, :]
            outputs.append(out)

        return self.proj(torch.cat(outputs, dim=-1))


class CrossscaleConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.crossscale = nn.Conv2d(dim, dim, kernel_size=kernel,
                            padding=kernel//2, groups=dim)
    def forward(self, x):
        return self.crossscale(x)



class CrossScaleEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, scales=[3, 5, 7], norm_layer=nn.LayerNorm):
        super().__init__()
        self.scales = scales
        self.projs = nn.ModuleList([
            nn.Sequential(
                CrossscaleConv(in_chans, k),
                nn.Conv2d(in_chans, embed_dim // len(scales), 1)
            )
            for k in scales
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        B, C, H, W = x.shape
        xs = [proj(x) for proj in self.projs]
        x = torch.cat(xs, dim=1)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class LSBlock(nn.Module):
    def __init__(self, dim, num_heads, scales=[3, 5, 15]):
        super().__init__()
        self.cse = CrossScaleEmbed(dim, dim, scales=[3, 5, 7])
        self.LS_attn = LSAttention(dim, num_heads, scales)
        # self.norm1 = nn.LayerNorm(dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(dim, dim * 4),
        #     nn.GELU(),
        #     nn.Linear(dim * 4, dim)
        # )
        # self.norm2 = nn.LayerNorm(dim)


    def forward(self, x):
        x = self.cse(x)
        x = x + self.LS_attn(self.norm1(x))

        return x


class LSModel(nn.Module):
    def __init__(self, protein_dim, gene_text_dim, gene_dim,freq_dim,conservation_dim,function_dim):
        super().__init__()

        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )

        # self.gene_text_encoder = nn.Sequential(
        #     nn.Linear(gene_text_dim, 512),
        #     nn.GELU(),
        #     nn.LayerNorm(512)
        # )

        self.gene_encoder = nn.Sequential(
            PeriodicEmbeddings(gene_dim, 32, lite=False),
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (1, 512))
        )

        self.freq_encoder = nn.Sequential(
            PeriodicEmbeddings(freq_dim, 28, lite=False),
            nn.Flatten(start_dim=1),
            nn.Linear(freq_dim*28, 512),
            nn.Unflatten(1, (1, 512))
        )

        self.conservation_encoder = nn.Sequential(
            PeriodicEmbeddings(conservation_dim, 24, lite=False),
            nn.Flatten(start_dim=1),
            nn.Linear(conservation_dim*24, 512),
            nn.Unflatten(1, (1, 512))
        )

        self.function_encoder = nn.Sequential(
            PeriodicEmbeddings(function_dim, 32, lite=False),
            nn.Flatten(start_dim=1),
            nn.Unflatten(1, (3, 512))
        )

        self.LS_former = nn.Sequential(
            LSBlock(dim=512, num_heads=8),
            LSBlock(dim=512, num_heads=8),
            LSBlock(dim=512, num_heads=8)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, Config.num_classes)
        )

    def forward(self, protein, gene_text,gene,freq, conservation,function_score):
        protein_feat = self.protein_encoder(protein)
        # gene_text_feat = self.gene_text_encoder(gene_text)
        gene_feat = self.gene_encoder(gene)
        freq_feat = self.freq_encoder(freq)
        conservation_feat = self.conservation_encoder(conservation)
        function_feat = self.function_encoder(function_score)

        combined = torch.cat([protein_feat,gene_feat, freq_feat,conservation_feat,function_feat], dim=1)
        # combined = torch.cat([protein_feat, gene_text_feat,gene_feat, freq_feat,conservation_feat,function_feat], dim=1)

        features = self.LS_former(combined)
        features = features.mean(dim=1)
        return self.classifier(features)

def train(train_loader, val_loader):
    model = LSFormerModel(
        protein_dim=Config.protein_embed_dim,
        # gene_text_dim  =Config.protein_embed_dim,
        gene_dim=len(Config.gene_features),
        freq_dim=len(Config.freq_features),
        conservation_dim=len(Config.conservation_features),
        function_dim =len(Config.function_features)
    ).to(Config.device)
    class_weights = torch.tensor([1.0, 3.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(Config.device))
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    best_val_loss = float('inf')
    patience = 6
    patience_counter = 0
    best_model_state = None
    model_save_path = r"./model/model.pth"

    for epoch in range(Config.epochs):
        model.train()
        for protein,gene_text, gene,freq, cons, fun_s, labels in train_loader:
            inputs = [t.to(Config.device) for t in [protein,gene_text, gene,freq, cons, fun_s]]
            labels = labels.to(Config.device)
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total_correct = 0
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for protein,gene_text, gene,freq, cons, fun_s, labels in val_loader:
                inputs = [t.to(Config.device) for t in [protein,gene_text, gene,freq, cons, fun_s]]
                labels = labels.to(Config.device)
                outputs = model(*inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                all_preds.append(outputs.softmax(1)[:, 1].cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        acc = total_correct / len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        print(
            f"Epoch {epoch + 1}/{Config.epochs} | Val Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Val AUROC: {auroc:.4f} | Val AUPRC: {auprc:.4f}")
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best validation loss: {best_val_loss:.4f}")
                break
    model.load_state_dict(best_model_state)

    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    torch.save(best_model_state, model_save_path)
    print(f"Best model saved to {model_save_path}")

    return model
