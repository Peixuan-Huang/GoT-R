import torch
import torch.nn as nn
from random import sample
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=128, r=16384, m=0.999, T=0.1, device_ids=[0,1],mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        self.max_len = 64
        self.device_ids = device_ids
        # create the encoders
        self.model_name = "./distilbert/distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.encoder_q = DistilBertModel.from_pretrained(self.model_name)
        self.encoder_k = DistilBertModel.from_pretrained(self.model_name)

        if torch.cuda.device_count() > 1:
            device_ids = self.device_ids  # 10卡机
            self.encoder_q = torch.nn.DataParallel(self.encoder_q, device_ids=device_ids)  # 指定要用到的设备
            self.encoder_q = self.encoder_q.cuda(device=device_ids[0])
            self.encoder_k = torch.nn.DataParallel(self.encoder_k, device_ids=device_ids)  # 指定要用到的设备
            self.encoder_k = self.encoder_k.cuda(device=device_ids[0])


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(device=self.device_ids[0])

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_shuffle(self,x):
        """
        Batch shuffle, for making use of BatchNorm.
        This version does not support DistributedDataParallel (DDP) model.
        """
        # Gather all tensors into one
        batch_size_this = x.shape[0]
        x_gather = x.clone()  # 在没有分布式情况下直接使用原始张量
        batch_size_all = x_gather.shape[0]

        # Random shuffle index
        idx_shuffle = torch.randperm(batch_size_all)

        # Index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # Shuffled index for this batch
        idx_this = idx_shuffle[:batch_size_this]  # 取前 batch_size_this 个元素

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    @torch.no_grad()
    def _batch_unshuffle(self,x, idx_unshuffle):
        """
        Undo batch shuffle.
        This version does not support DistributedDataParallel (DDP) model.
        """
        # 直接使用输入张量
        x_gather = x.clone()  # 在没有分布式情况下直接使用原始张量
        # batch_size_all = x_gather.shape[0]

        # 使用提供的 idx_unshuffle 还原顺序
        restored_x = x_gather[idx_unshuffle]

        return restored_x

    def forward(self, im_q, atomic_relations=None, im_k=None, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        if is_eval:
            question_input = self.tokenizer(im_q, return_tensors="pt", padding='max_length', max_length=self.max_len,
                                       truncation=True).to(device=self.device_ids[0])
            k = self.encoder_k(**question_input).last_hidden_state.mean(dim=1)
            k = F.normalize(k, dim=1)
            # k = self.encoder_k(im_q)
            # k = nn.functional.normalize(k, dim=1)
            return k
        
        # compute query features
        # q = self.encoder_q(im_q)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)
        question_input = self.tokenizer(im_q, return_tensors="pt", padding='max_length', max_length=self.max_len,
                                        truncation=True).to(device=self.device_ids[0])
        q = self.encoder_q(**question_input).last_hidden_state.mean(dim=1)
        q = F.normalize(q, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            pos_sample_id = im_k
            pos_sample = list(atomic_relations[pos_sample_id])
            positive_input = self.tokenizer(pos_sample, return_tensors="pt", padding='max_length',
                                       max_length=self.max_len, truncation=True).to(device=self.device_ids[0])
            positive_embedding_last = self.encoder_k(**positive_input).last_hidden_state.mean(dim=1)
            positive_embedding_last = F.normalize(positive_embedding_last, dim=1)
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle(im_k)
            
            # k = self.encoder_k(im_k)  # keys: NxC
            # k = nn.functional.normalize(k, dim=1)
            pos_sample_id = im_k
            pos_sample = list(atomic_relations[pos_sample_id])
            question_input = self.tokenizer(pos_sample, return_tensors="pt", padding='max_length', max_length=self.max_len,
                                       truncation=True).to(device=self.device_ids[0])
            k = self.encoder_k(**question_input).last_hidden_state.mean(dim=1)
            k = F.normalize(k, dim=1)
            
            # undo shuffle
            k = self._batch_unshuffle(k, idx_unshuffle)
        
        if not cluster_result:
            # 随机负样本对比学习
            pos_sample_id = im_k
            pos_sample = list(atomic_relations[pos_sample_id])
            positive_input = self.tokenizer(pos_sample, return_tensors="pt", padding='max_length',
                                       max_length=self.max_len, truncation=True).to(device=self.device_ids[0])
            positive_embedding = self.encoder_k(**positive_input).last_hidden_state.mean(dim=1)
            positive_embedding = F.normalize(positive_embedding, dim=1)
            
            # 随机负样本
            negative_embedding = self.queue.clone().detach().T # torch.Size([5, 768])
            filtered_negative = []
            
            for pos in positive_embedding_last:
                # 找到与当前正样本不相等的负样本
                mask = ~torch.all(negative_embedding.to(device=self.device_ids[0]) == pos.to(device=self.device_ids[0]).unsqueeze(0), dim=1).any(dim=0)
                valid_negatives = negative_embedding[mask]
                # print(valid_negatives.shape)
                if valid_negatives.shape[1] < self.r:
                    valid_negatives = valid_negatives.expand(1, self.r, -1)
                
                # 将有效的负样本添加到列表中
                filtered_negative.append(valid_negatives)

            # 将列表转换为张量
            negative_embedding = torch.stack(filtered_negative).squeeze(1).to(device=self.device_ids[0])
            
                    
            
            # compute logits
            # 计算锚点与正样本的相似度
            pos_sim = F.cosine_similarity(q, positive_embedding, dim=1)  # 形状 [batch_size, 1]

            # 计算锚点与所有负样本的相似度
            neg_sims = F.cosine_similarity(q.unsqueeze(1),negative_embedding)
            # print(neg_sims.shape) #[16,3]
            # neg_sims = F.cosine_similarity(q.unsqueeze(1),negative_embedding) # 形状 [batch_size, 3]

            # 将正样本相似度和负样本相似度合并
            logits = torch.cat((pos_sim.unsqueeze(1), neg_sims), dim=1)  # 形状 [batch_size, 4]
            logits /= self.T
            labels = torch.zeros(q.size(0), dtype=torch.long).to(device=self.device_ids[0])  # 形状 [batch_size]
            
            # dequeue and enqueue
            self._dequeue_and_enqueue(k)
            # return q,positive_embedding,negative_embedding, None, None
            return logits, labels, None, None
        
       
        
        # prototypical contrast
        elif cluster_result is not None:
            # 硬负样本对比学习
            pos_sample_id = im_k
            pos_sample = list(atomic_relations[pos_sample_id])
            positive_input = self.tokenizer(pos_sample, return_tensors="pt", padding='max_length',
                                       max_length=self.max_len, truncation=True).to(device=self.device_ids[0])
            positive_embedding = self.encoder_k(**positive_input).last_hidden_state.mean(dim=1)
            positive_embedding = F.normalize(positive_embedding,dim=1)

            # # print(np.array(cluster_result['neg_samples_index'])[0].shape) ###############################
            neg_sample_id = np.array(cluster_result['neg_samples_index'])[0][index] 
            
            # print(neg_sample_id)
            # print(neg_sample_id.shape)
            neg_samples = list(atomic_relations[neg_sample_id])
            # print(neg_samples)

                

            negative_input = [self.tokenizer(neg_question.reshape(-1,).tolist(), return_tensors="pt", padding='max_length', max_length=self.max_len,
                                        truncation=True).to(device=self.device_ids[0]) for neg_question in
                              neg_samples]
            negative_embedding = [F.normalize(self.encoder_k(**neg_input).last_hidden_state.mean(dim=1),dim=1) for neg_input in negative_input]
            
            negative_embedding = torch.stack(negative_embedding).squeeze(1).to(device=self.device_ids[0])
            
            

            # 原型对比学习
            proto_labels = []
            proto_logits = []
            for n, (im2cluster,prototypes,density,neg_proto_indexs,alllabel) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'],cluster_result['neg_cluster_index'],cluster_result['alllabel'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index] # 样本所属的簇的id
                # print(pos_proto_id) # tensor([76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76],device='cuda:0')
                pos_proto_id_new = [alllabel.index(pid) for pid in pos_proto_id.cpu().numpy().tolist()]
                pos_prototypes = prototypes[pos_proto_id_new].to(device=self.device_ids[0])  # 样本所述的簇的质心即id
                # print(pos_prototypes)
                
                # sample negative prototypes
                neg_proto_id_new = neg_proto_indexs[pos_proto_id_new]
                neg_proto_id = [alllabel[idx] for indexs in neg_proto_id_new for idx in indexs]
                neg_prototypes =prototypes[neg_proto_id_new].to(device=self.device_ids[0])
                
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).to(device=self.device_ids[0])],dim=0)].reshape(q.size(0),-1).to(device=self.device_ids[0])
                
            return q,positive_embedding,negative_embedding,pos_prototypes,neg_prototypes,temp_proto
            # return logits, proto_logits
        # else:
        #     return logits, labels, None, None


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
