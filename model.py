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



# ==================== 配置参数 ====================
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
        "GDI", "GDI-Phred", "LoFtool_score", "RVIS_EVS","RVIS_percentile_EVS", "ExAC_pLI", "gnomAD_pLI",
        "SORVA_LOF_MAF0.005_HetOrHom", "SORVA_LOForMissense_MAF0.005_HetOrHom","gnomAD_pRec", "gnomAD_pNull", "Gene_indispensability_score"
    ]
    freq_features = [
        "1000Gp3_AF", "ExAC_AF", "ExAC_Adj_AF", "gnomAD_genomes_AF",  "ALFA_Other_AF", "ALFA_Total_AF"
    ]
    conservation_features = [
        "GERP++_NR", "GERP++_RS_rankscore", "GERP_91_mammals_rankscore","phyloP100way_vertebrate_rankscore",
        "phyloP470way_mammalian_rankscore", "phyloP17way_primate_rankscore", "phastCons100way_vertebrate_rankscore",
        "phastCons470way_mammalian_rankscore","phastCons17way_primate_rankscore", "SiPhy_29way_logOdds_rankscore", "bStatistic_converted_rankscore",
    ]
    function_features = [
        "SIFT4G_converted_rankscore", "Polyphen2_HDIV_rankscore", "Polyphen2_HVAR_rankscore", "MutationTaster_converted_rankscore",
        "MutationAssessor_rankscore", "fathmm-XF_coding_rankscore", "PROVEAN_converted_rankscore", "VEST4_rankscore",
        "MetaSVM_rankscore", "MetaLR_rankscore", "M-CAP_rankscore","REVEL_rankscore","MutPred_rankscore", "DEOGEN2_rankscore",
        "Eigen-phred_coding","DANN_rankscore",
    ]


# ==================== 数据处理器 ====================
class FeatureProcessor:
    def __init__(self, train_df):
        self._init_text_encoder()
        self._fit_numeric_transformers(train_df)

    def _init_text_encoder(self):
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_model)
        self.bert = BertModel.from_pretrained(Config.bert_model).to(Config.device)
        self.bert.eval()

    def _fit_numeric_transformers(self, df):
        # df['Essential_gene_CRISPR'] = df['Essential_gene_CRISPR'].replace({'E': '1', 'N': '-1'})
        # df['Essential_gene_CRISPR2'] = df['Essential_gene_CRISPR2'].replace({'E': 1,'S': 0,'N': -1, np.nan: -1})

        self.gene_qt = QuantileTransformer(output_distribution='normal')
        self.gene_qt.fit(df[Config.gene_features].fillna(-1))

        self.gene_CRISPR = {'E': 1, 'N': -1, np.nan: -1}
        self.gene_CRISPR2 = {'E': 1,'S': 0,'N': -1, np.nan: -1}

    def process_row(self, row, is_train=True):
        # 蛋白质特征
        protein = self._process_protein(row)
        gene_text = self._process_gene(row)
        # 基因+频率特征合并
        gene = row[Config.gene_features].copy()
        gene = self.gene_qt.transform(gene.fillna(-1).values.reshape(1, -1))
        freq = row[Config.freq_features].copy()

        # 计算中位数
        median_values = row[[
            "GDI", "GDI-Phred", "LoFtool_score", "RVIS_EVS", "RVIS_percentile_EVS", "ExAC_pLI", "gnomAD_pLI",
            "SORVA_LOF_MAF0.005_HetOrHom", "SORVA_LOForMissense_MAF0.005_HetOrHom", "gnomAD_pRec", "gnomAD_pNull",
            "Gene_indispensability_score",     "1000Gp3_AF", "ExAC_AF", "ExAC_Adj_AF", "gnomAD_genomes_AF",  "ALFA_Other_AF", "ALFA_Total_AF"
            "GERP++_NR","GERP++_RS_rankscore", "GERP_91_mammals_rankscore", "phyloP100way_vertebrate_rankscore",
            "phyloP470way_mammalian_rankscore", "phyloP17way_primate_rankscore", "phastCons100way_vertebrate_rankscore",
            "phastCons470way_mammalian_rankscore", "phastCons17way_primate_rankscore", "SiPhy_29way_logOdds_rankscore",
            "bStatistic_converted_rankscore",
            "SIFT4G_converted_rankscore", "Polyphen2_HDIV_rankscore", "Polyphen2_HVAR_rankscore","MutationTaster_converted_rankscore",
            "MutationAssessor_rankscore", "fathmm-XF_coding_rankscore", "PROVEAN_converted_rankscore","VEST4_rankscore",
            "MetaSVM_rankscore", "MetaLR_rankscore", "M-CAP_rankscore", "REVEL_rankscore", "MutPred_rankscore","DEOGEN2_rankscore","Eigen-phred_coding", "DANN_rankscore",
        ]].median()

        # 保守性特征
        conservation = row[[
            "GERP++_NR", "GERP++_RS_rankscore", "GERP_91_mammals_rankscore","phyloP100way_vertebrate_rankscore",
        "phyloP470way_mammalian_rankscore", "phyloP17way_primate_rankscore", "phastCons100way_vertebrate_rankscore",
        "phastCons470way_mammalian_rankscore","phastCons17way_primate_rankscore", "SiPhy_29way_logOdds_rankscore", "bStatistic_converted_rankscore",
        ]].fillna(median_values).values  # 20
        # 功能预测分数
        function_scores = row[[
            "SIFT4G_converted_rankscore", "Polyphen2_HDIV_rankscore", "Polyphen2_HVAR_rankscore",
            "MutationTaster_converted_rankscore",
            "MutationAssessor_rankscore", "fathmm-XF_coding_rankscore", "PROVEAN_converted_rankscore",
            "VEST4_rankscore",
            "MetaSVM_rankscore", "MetaLR_rankscore", "M-CAP_rankscore", "REVEL_rankscore", "MutPred_rankscore",
            "DEOGEN2_rankscore",
            "Eigen-phred_coding", "DANN_rankscore",
            # ... 其他保守性特征（根据实际CSV列名补充）
        ]].fillna(median_values).values  # 48个
        return {
            'protein': torch.FloatTensor(protein),
            'gene_text': torch.FloatTensor(gene_text),
            'gene': torch.FloatTensor(gene.squeeze()),
            'freq':torch.FloatTensor(freq),
            'conservation': torch.FloatTensor(conservation.squeeze()),
            'function_scores': torch.FloatTensor(function_scores.squeeze()),
            'label': torch.tensor(row['True_Label']) if is_train else None
        }

    def _process_protein(self, row):
        def encode_interpro(text):
            texts = ['Interpro_domain'] + text.split(';')[:15] if isinstance(text, str) else []
            # print("interpro_domain",texts)
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=Config.max_token_len,
                return_tensors='pt'
            ).to(Config.device)

            # 查看输入张量的形状
            # print("input_ids shape:", inputs["input_ids"].shape)
            # print("attention_mask shape:", inputs["attention_mask"].shape)
            with torch.no_grad():
                outputs = self.bert(**inputs)
                # print(outputs.last_hidden_state.shape)

            return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()


        def encode_Ensembl_proteinid(text):
            texts = ['Ensembl_proteinid'] + text.split(';')[:15] if isinstance(text, str) else []
            # print("interpro_domain",texts)
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
            # print("domain",texts)
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

        # 假设 domains, interpro, Ensembl_proteinid 是 numpy.ndarray，形状为 (768,)
        domains = np.expand_dims(domains, axis=0)  # 扩展维度，形状变为 (1, 768)
        interpro = np.expand_dims(interpro, axis=0)  # 扩展维度，形状变为 (1, 768)
        Ensembl_proteinid = np.expand_dims(Ensembl_proteinid, axis=0)  # 扩展维度，形状变为 (1, 768)

        # 使用 np.concatenate() 沿着第0维（行方向）拼接
        protein_embedding = np.concatenate([domains, interpro, Ensembl_proteinid], axis=0)  # 形状变为 (3, 768)

        # print(protein_embedding.shape)  # 输出: (3, 768)
        return protein_embedding
        # return (domains + interpro) / 2



    def _process_gene(self, row):
        def encode_GO_biological_process(text):
            texts = ['GO_biological_process'] + text.split(';')[:25] if isinstance(text, str) else []
            # print("interpro_domain",texts)
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=Config.max_token_len,
                return_tensors='pt'
            ).to(Config.device)

            with torch.no_grad():
                outputs = self.bert(**inputs)
                # print(outputs.last_hidden_state.shape)

            return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()

        def encode_GO_cellular_component(text):
            texts = ['GO_cellular_component'] + text.split(';')[:30] if isinstance(text, str) else []
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

        def encode_GO_molecular_function(text):
            texts = ['GO_molecular_function'] + text.split(';')[:30] if isinstance(text, str) else []
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

        def encode_Pathway_ConsensusPathDB(text):
            texts = ['Pathway(ConsensusPathDB)'] + text.split(';')[:30] if isinstance(text, str) else []
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

        def encode_Pathway_KEGG_id(text):
            texts = ['Pathway(KEGG)_id'] + text.split(';')[:15] if isinstance(text, str) else []
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

        def encode_Pathway_KEGG_full(text):
            texts = ['Pathway(KEGG)_full'] + text.split(';')[:15] if isinstance(text, str) else []
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


        GO_biological_process = encode_GO_biological_process(row['GO_biological_process'])
        GO_cellular_component = encode_GO_cellular_component(row['GO_cellular_component'])
        GO_molecular_function = encode_GO_molecular_function(row['GO_molecular_function'])
        Pathway_ConsensusPathDB = encode_Pathway_ConsensusPathDB(row['Pathway(ConsensusPathDB)'])
        Pathway_KEGG_id = encode_Pathway_KEGG_id(row['Pathway(KEGG)_id'])
        Pathway_KEGG_full = encode_Pathway_KEGG_full(row['Pathway(KEGG)_full'])


        # 假设 domains, interpro, Ensembl_proteinid 是 numpy.ndarray，形状为 (768,)
        GO_biological_process = np.expand_dims(GO_biological_process, axis=0)  # 扩展维度，形状变为 (1, 768)
        GO_cellular_component = np.expand_dims(GO_cellular_component, axis=0)  # 扩展维度，形状变为 (1, 768)
        GO_molecular_function = np.expand_dims(GO_molecular_function, axis=0)  # 扩展维度，形状变为 (1, 768)
        Pathway_ConsensusPathDB = np.expand_dims(Pathway_ConsensusPathDB, axis=0)  # 扩展维度，形状变为 (1, 768)
        Pathway_KEGG_id = np.expand_dims(Pathway_KEGG_id, axis=0)  # 扩展维度，形状变为 (1, 768)
        Pathway_KEGG_full = np.expand_dims(Pathway_KEGG_full, axis=0)  # 扩展维度，形状变为 (1, 768)


        # 使用 np.concatenate() 沿着第0维（行方向）拼接
        gene_embedding = np.concatenate([GO_biological_process, GO_cellular_component, GO_molecular_function,Pathway_ConsensusPathDB,Pathway_KEGG_id,Pathway_KEGG_full], axis=0)  # 形状变为 (3, 768)

        # print(protein_embedding.shape)  # 输出: (3, 768)
        return gene_embedding

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
            return data['protein'], data['gene_text'],data['gene'],data['freq'], data['conservation'], data['function_scores'], data['label']
        else:
            return data['protein'], data['gene_text'],data['gene'],data['freq'], data['conservation'], data['function_scores']


# ==================== CrossFormer 模块 ====================
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, scales=[3, 5, 15]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True)
            for _ in range(len(scales))
        ])
        self.proj = nn.Linear(dim * len(scales), dim)

    def forward(self, x):
        B, L, C = x.shape           #B=batch_size,L=16,C=特征维度
        outputs = []
        for scale, attn in zip(self.scales, self.attentions):
            if L % scale != 0:
                pad_size = scale - (L % scale)
                x_padded = F.pad(x, (0, 0, 0, pad_size))  # 在序列维度填充
                L_padded = L + pad_size
            else:
                x_padded = x
                L_padded = L

            # 分割特征
            x_reshaped = x_padded.reshape(B, L_padded // scale, scale, C)   # [B, blocks, scale, C]
            x_reshaped = x_reshaped.reshape(-1, scale, C)  # [B * (L // scale), scale, C]

            # 多尺度注意力
            out, _ = attn(x_reshaped, x_reshaped, x_reshaped)  # [B * (L // scale), scale, C]

            # 恢复原始形状并截断填充部分
            out = out.reshape(B, L_padded // scale, scale, C)  # [B, L_padded // scale, scale, C]
            out = out.reshape(B, L_padded, C)[:, :L, :]  # [B, L, C]
            outputs.append(out)

        # 特征融合
        return self.proj(torch.cat(outputs, dim=-1))


class CrossFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, scales=[3, 5, 15]):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads, scales)        #dim-512,num_heads=8
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # 跨尺度注意力
        x = x + self.cross_attn(self.norm1(x))
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== 模型架构 ====================
class CrossFormerModel(nn.Module):
    def __init__(self, protein_dim, gene_text_dim, gene_dim,freq_dim,conservation_dim,function_dim):
        super().__init__()
        # 蛋白质特征编码
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )
        # 基因特征编码
        self.gene_text_encoder = nn.Sequential(
            nn.Linear(gene_text_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )

        # 基因数字特征编码
        self.gene_encoder = nn.Sequential(
            PeriodicEmbeddings(gene_dim, 32, lite=False),  # 输出形状为 [batch_size, 48, aaa]
            nn.Flatten(start_dim=1),  # 将形状变为 [batch_size, 48 * aaa]
            nn.Unflatten(1, (1, 512))  # 将形状变为 [batch_size, bbb, 256]
        )
        # self.gene_encoder = nn.Sequential(
        #     nn.Linear(gene_dim, 128),
        #     nn.GELU(),
        #     nn.LayerNorm(128)
        # )

        # 频率特征编码
        self.freq_encoder = nn.Sequential(
            PeriodicEmbeddings(freq_dim, 28, lite=False),  # 输出形状为 [batch_size, 48, aaa]
            nn.Flatten(start_dim=1),  # 将形状变为 [batch_size, 48 * aaa]
            nn.Linear(freq_dim*28, 512),
            nn.Unflatten(1, (1, 512))  # 将形状变为 [batch_size, bbb, 256]
        )
        # self.freq_encoder = nn.Sequential(
        #     nn.Linear(freq_dim, 128),
        #     nn.GELU(),
        #     nn.LayerNorm(128)
        # )

        # # 保守性特征编码
        # self.conservation_encoder = nn.Sequential(
        #     nn.Linear(conservation_dim, 64),
        #     nn.GELU(),
        #     nn.LayerNorm(64)
        # )
        # 保守性特征编码
        self.conservation_encoder = nn.Sequential(
            PeriodicEmbeddings(conservation_dim, 24, lite=False),  # 输出形状为 [batch_size, 48, aaa]
            nn.Flatten(start_dim=1),  # 将形状变为 [batch_size, 48 * aaa]
            nn.Linear(conservation_dim*24, 512),
            nn.Unflatten(1, (1, 512))  # 将形状变为 [batch_size, bbb, 256]
        )
        # self.conservation_encoder = nn.Sequential(
        #     PeriodicEmbeddings(conservation_dim, 8, lite=False),#conservation_dim=20
        #     nn.Flatten(),
        #     nn.Linear(160, 64),
        #     nn.GELU(),
        #     nn.LayerNorm(64)
        # )

        # 功能分数特征编码
        self.function_encoder = nn.Sequential(
            PeriodicEmbeddings(function_dim, 32, lite=False),  # 输出形状为 [batch_size, 48, aaa]
            nn.Flatten(start_dim=1),  # 将形状变为 [batch_size, 48 * aaa]
            nn.Unflatten(1, (3, 512))  # 将形状变为 [batch_size, bbb, 256]
        )
        # self.function_encoder = nn.Sequential(
        #     PeriodicEmbeddings(function_dim, 4, lite=False),    #function_dim=48
        #     nn.Flatten(),
        #     nn.Linear(192, 64),
        #     nn.GELU(),
        #     nn.LayerNorm(64)
        # )



        # CrossFormer 模块
        self.cross_former = nn.Sequential(
            CrossFormerBlock(dim=512, num_heads=8),
            CrossFormerBlock(dim=512, num_heads=8)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, Config.num_classes)
        )

        # 分类头
        # self.classifier = nn.Sequential(
        #     PeriodicEmbeddings(512, 6, lite=False),
        #     nn.Flatten(),
        #     nn.Linear(3072, 256),
        #     nn.GELU(),
        #     nn.Linear(256, Config.num_classes)
        # )



    def forward(self, protein, gene_text,gene,freq, conservation,function_score):
        # 特征编码
        #print(protein.shape,gene_text.shape,gene.shape,freq.shape,conservation.shape,function_score.shape)


        protein_feat = self.protein_encoder(protein)  # [B, 256]
        # print("只因你太美",protein_feat.shape)
        gene_text_feat = self.gene_text_encoder(gene_text)  # [B, 256]

        gene_feat = self.gene_encoder(gene)  # [B, 128]
        freq_feat = self.freq_encoder(freq)  # [B, 128]
        conservation_feat = self.conservation_encoder(conservation)  # [B, 128]
        function_feat = self.function_encoder(function_score)   # [B, 128]


        #print(protein_feat.shape, gene_text_feat.shape, gene_feat.shape, freq_feat.shape, conservation_feat.shape, function_feat.shape)


        # 特征拼接
        combined = torch.cat([protein_feat, gene_text_feat,gene_feat, freq_feat,conservation_feat,function_feat], dim=1)  # [B, 512]
        #print(combined.shape)

        # 调整形状为 [B, L, C]，其中 L > 1
        #combined = combined.unsqueeze(1).repeat(1, 16, 1)  # [B, 16, 512]
        # print("输出内容combined.shape:",combined.shape)
        # print("输出内容combined:",combined)

        # CrossFormer 特征交互
        features = self.cross_former(combined)  # [B, 15, 512]

        # 全局平均池化
        features = features.mean(dim=1)  # [B, 512]

        # 分类
        return self.classifier(features)


# ==================== 训练流程 ====================
def train(train_loader, val_loader):
    model = CrossFormerModel(
        protein_dim=Config.protein_embed_dim,
        gene_text_dim  =Config.protein_embed_dim,
        gene_dim=len(Config.gene_features),
        freq_dim=len(Config.freq_features),
        conservation_dim=len(Config.conservation_features),  # 20  # 根据实际特征数量调整
        function_dim =len(Config.function_features)
    ).to(Config.device)

    class_weights = torch.tensor([1.0, 3.0])  # 假设正样本权重为3
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(Config.device))
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)

    best_val_loss = float('inf')  # 记录最佳验证集损失
    patience = 6  # 设置耐心值
    patience_counter = 0  # 当前未提升的轮数计数器
    best_model_state = None  # 保存最佳模型状态
    # 模型保存路径
    model_save_path = r"C:\Users\A\PycharmProjects\my_paper\miss_predict\model\all_feat_model.pth"

    for epoch in range(Config.epochs):
        model.train()
        for protein,gene_text, gene,freq, cons, fun_s, labels in train_loader:
            inputs = [t.to(Config.device) for t in [protein,gene_text, gene,freq, cons, fun_s]]

            # print("Input shape:", gene_freq.shape)
            labels = labels.to(Config.device)  # CrossEntropyLoss损失函数的
            print("真的开始训练了，准备运行outputs = model(*inputs)")

            optimizer.zero_grad()
            outputs = model(*inputs)

            loss = criterion(outputs, labels)  # CrossEntropyLoss损失函数的

            loss.backward()
            optimizer.step()

        # 验证
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
                # print("没什么问题")
                total_loss += loss.item() * labels.size(0)

                total_correct += (outputs.argmax(1) == labels).sum().item()
                all_preds.append(outputs.softmax(1)[:, 1].cpu().numpy())  # 假设是二分类问题
                all_labels.append(labels.cpu().numpy())

        # 计算平均 Loss
        avg_loss = total_loss / len(val_loader.dataset)
        # 计算 ACC
        acc = total_correct / len(val_loader.dataset)
        # 计算 AUROC 和 AUPRC
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        auroc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)

        # 打印结果
        print(
            f"Epoch {epoch + 1}/{Config.epochs} | Val Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Val AUROC: {auroc:.4f} | Val AUPRC: {auprc:.4f}")
        # Early Stopping 逻辑
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_model_state = model.state_dict()  # 保存最佳模型状态
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best validation loss: {best_val_loss:.4f}")
                break

    # 加载最佳模型状态
    model.load_state_dict(best_model_state)

    # 保存最佳模型到磁盘
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    torch.save(best_model_state, model_save_path)
    print(f"Best model saved to {model_save_path}")

    return model




# ==================== 预测函数 ====================
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

    # 计算 ACC
    acc = total_correct / len(test_loader.dataset)
    # 计算 AUROC 和 AUPRC
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    # 打印结果
    print(f"| test Acc: {acc:.4f} | test AUROC: {auroc:.4f} | test AUPRC: {auprc:.4f}")

    # return torch.cat(preds, dim=0).numpy()
    return all_preds


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 数据加载与处理
    train_df = pd.read_csv(r"C:\Users\A\PycharmProjects\my_paper\miss_predict\mis_clinvar_train.csv")
    test_df = pd.read_csv(r"C:\Users\A\PycharmProjects\my_paper\miss_predict\mis_clinvar_test.csv")
    scores_csv = r"C:\Users\A\PycharmProjects\my_paper\miss_predict\all_feat_model_score\scores-mis_clinvar_test.csv"
    #mis_TP53_test  mis_cancer_DIShcover    mis_clinvar_test    mis_clinvar_train   mis_lofgof_gnomad   mis_PPARG_test  mis_swissport

    train_df['True_Label'] = train_df['True_Label'].replace(-1, 0)
    test_df['True_Label'] = test_df['True_Label'].replace(-1, 0)

    # train_df['Interpro_domain'] = train_df['Interpro_domain'].fillna("unknown")
    # test_df['Interpro_domain'] = test_df['Interpro_domain'].fillna("unknown")
    #
    # train_df['DOMAINS'] = train_df['DOMAINS'].fillna("unknown")
    # test_df['DOMAINS'] = test_df['DOMAINS'].fillna("unknown")
    columns_to_fill = [
        'Interpro_domain', 'DOMAINS', 'GO_biological_process', 'GO_cellular_component',
        'GO_molecular_function', 'Pathway(ConsensusPathDB)', 'Pathway(KEGG)_id',
        'Pathway(KEGG)_full', 'Ensembl_proteinid'
    ]
    for col in columns_to_fill:
        test_df[col] = test_df[col].fillna("unknown")
        train_df[col] = train_df[col].fillna("unknown")



    # 指定需要填补的列
    columns_to_fill = ["GenoCanyon_rankscore", "integrated_fitCons_rankscore",
                       "integrated_confidence_value", "GM12878_fitCons_rankscore",
                       "GM12878_confidence_value", "H1-hESC_fitCons_rankscore",
                       "HUVEC_fitCons_rankscore", "HUVEC_confidence_value",
                       "LINSIGHT_rankscore", "GERP++_NR", "GERP++_RS_rankscore",
                       "GERP_91_mammals_rankscore", "phyloP100way_vertebrate_rankscore",
                       "phyloP470way_mammalian_rankscore", "phyloP17way_primate_rankscore",
                       "phastCons100way_vertebrate_rankscore", "phastCons470way_mammalian_rankscore",
                       "phastCons17way_primate_rankscore",
                       "SiPhy_29way_logOdds_rankscore", "bStatistic_converted_rankscore",
                       "SIFT4G_converted_rankscore", "Polyphen2_HDIV_rankscore",
                       "Polyphen2_HVAR_rankscore", "MutationTaster_converted_rankscore",
                       "MutationAssessor_rankscore", "FATHMM_converted_rankscore",
                       "PROVEAN_converted_rankscore", "VEST4_rankscore",
                       "MetaSVM_rankscore", "MetaLR_rankscore", "Reliability_index",
                       "MetaRNN_rankscore", "M-CAP_rankscore", "REVEL_rankscore",
                       "MutPred_rankscore", "MVP_rankscore", "gMVP_rankscore",
                       "MPC_rankscore", "PrimateAI_rankscore", "DEOGEN2_rankscore",
                       "BayesDel_addAF_rankscore", "BayesDel_noAF_rankscore",
                       "ClinPred_rankscore", "LIST-S2_rankscore", "VARITY_R_rankscore",
                       "VARITY_ER_rankscore", "VARITY_R_LOO_rankscore",
                       "VARITY_ER_LOO_rankscore", "ESM1b_rankscore", "EVE_rankscore",
                       "AlphaMissense_rankscore", "PHACTboost_rankscore",
                       "MutFormer_rankscore", "MutScore_rankscore", "CADD_raw",
                       "CADD_raw_rankscore", "CADD_phred", "CADD_raw_hg19",
                       "CADD_raw_rankscore_hg19", "CADD_phred_hg19", "DANN_score",
                       "DANN_rankscore", "fathmm-MKL_coding_rankscore",
                       "fathmm-XF_coding_rankscore", "Eigen-raw_coding_rankscore",
                       "Eigen-phred_coding", "Eigen-PC-raw_coding_rankscore",
                       "Eigen-PC-phred_coding"]

    # 计算这些列的众数（取第一个众数）
    mode_values = train_df[columns_to_fill].mode().iloc[0]
    ## 计算这些列的众数（取第一个众数），如果某列全部缺失则返回 0
    # mode_values = train_df[columns_to_fill].apply(lambda col: col.mode().iloc[0] if col.notna().any() else 0)
    # 用众数填补空缺值
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

    # 训练与预测
    model = train(train_loader, val_loader)
    predictions = predict(model, test_loader)

    pd.DataFrame(predictions, columns=["Prob_1"]).to_csv(scores_csv, index=False)
    print(f"预测结果已保存到 {scores_csv}")

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

    ]  # mis_cancer_DIShcover   mis_lofgof_gnomad  mis_swissport   mis_TP53_test   mis_PPARG_test

    # 遍历每个数据集
    for input_csv, output_csv in datasets:
        # 读取数据
        test_df = pd.read_csv(input_csv)


        # 数据预处理
        test_df['True_Label'] = test_df['True_Label'].replace(-1, 0)
        # test_df['Interpro_domain'] = test_df['Interpro_domain'].fillna("unknown")
        # test_df['DOMAINS'] = test_df['DOMAINS'].fillna("unknown")
        # test_df['GO_biological_process'] = test_df['GO_biological_process'].fillna("unknown")
        # test_df['GO_cellular_component'] = test_df['GO_cellular_component'].fillna("unknown")
        # test_df['GO_molecular_function'] = test_df['GO_molecular_function'].fillna("unknown")
        # test_df['Pathway(ConsensusPathDB)'] = test_df['Pathway(ConsensusPathDB)'].fillna("unknown")
        # test_df['Pathway(KEGG)_id'] = test_df['Pathway(KEGG)_id'].fillna("unknown")
        # test_df['Pathway(KEGG)_full'] = test_df['Pathway(KEGG)_full'].fillna("unknown")
        # test_df['Ensembl_proteinid'] = test_df['Ensembl_proteinid'].fillna("unknown")
        columns_to_fill = [
            'Interpro_domain', 'DOMAINS', 'GO_biological_process', 'GO_cellular_component',
            'GO_molecular_function', 'Pathway(ConsensusPathDB)', 'Pathway(KEGG)_id',
            'Pathway(KEGG)_full', 'Ensembl_proteinid'
        ]
        for col in columns_to_fill:
            test_df[col] = test_df[col].fillna("unknown")



        # 特征处理器
        processor = FeatureProcessor(test_df)

        # 创建数据集和数据加载器
        test_set = MutationDataset(test_df, processor)
        test_loader = DataLoader(test_set, batch_size=Config.batch_size)

        # 进行预测
        predictions = predict(model, test_loader)

        # 保存预测结果
        pd.DataFrame(predictions, columns=["Prob_1"]).to_csv(output_csv, index=False)
        print(f"预测结果已保存到 {output_csv}")

