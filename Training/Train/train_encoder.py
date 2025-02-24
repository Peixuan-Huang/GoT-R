import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import jsonlines
import faiss
import numpy as np
from easydict import EasyDict
import random
import math
import time
import pcl.builder
import torch.nn.functional as F
from libKMCUDA import kmeans_cuda



class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.max_len = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
                    "question": sample["abstract_question"],
                    # "positive_question": sample["pos_relation"],
                    "pos_relation_indexs": sample["pos_relation_indexs"]
               }, idx

    def load_data(self, data_path):
        data = []
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                data.append(obj)
        data = random.sample(data,len(data)//args.train_batch_size*args.train_batch_size)
        return data
    
class DevDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.max_len = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
                    "question": sample["abstract_question"],
                    # "positive_question": sample["pos_relation"],
                    "pos_relation_indexs": sample["pos_relation_indexs"]
               }, idx

    def load_data(self, data_path):
        data = []
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                data.append(obj)
        return data

class TestDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.max_len = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
                    "question": sample["abstract_question"],
                    # "positive_question": sample["pos_relation"],
                    "pos_relation_indexs": sample["pos_relation_indexs"]
               }, idx

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r') as reader:
            data = json.load(reader)
        return data



# 设置随机种子
# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_features(train_loader, model, args):
    model.eval()
    # features = torch.zeros((len(train_loader.dataset)//args.train_batch_size)*args.train_batch_size, args.low_dim).to(device=args.device_ids[0])
    features = torch.zeros(len(train_loader.dataset), args.low_dim).to(device=args.device_ids[0])
    # print(features.shape)
    pos_relation_indexs = []
    for i, (batch, index) in enumerate(tqdm(train_loader,desc="calculating embeddings")):
        with torch.no_grad():
            question_embedding = model(batch["question"], is_eval=True)
            # print(question_embedding.shape) [16,768]
            features[index] = question_embedding
            pos_relation_indexs.extend(batch["pos_relation_indexs"]) # 每个样本所属的标签

    # # 或者你可以使用以下方式更清晰地表达：
    # features = features[features.any(dim=1)]

    # # 结果的形状
    # print(features.shape)
    
    return features.cpu(), np.array(pos_relation_indexs)



# 计算质心
def compute_centroid(embeddings):
    return torch.mean(embeddings, dim=0)
    # return np.mean(embeddings.cpu().numpy(), axis=0)


class CosineSimilarityIndex(faiss.IndexFlat):
    def __init__(self, d):
        super().__init__(d)
        
    def search(self, x, k):
        # 计算点积
        D, I = super().search(x, k)

        # 计算模长
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        print(I)
        print(k)
        norms = self.reconstruct_n(I, k)  # 重建聚类中心
        norms = np.linalg.norm(norms, axis=1, keepdims=True)

        # 余弦相似度 = 点积 / (||A|| * ||B||)
        similarity = D / (x_norm * norms.T)
        return 1 - similarity, I  # 返回距离（1 - 相似度）

def run_kmeans(x, pos_relation_indexs, atomic_relations, im2cluster, density,model, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': [], 'neg_samples_index':[],"neg_cluster_index":[],'alllabel':[]}

    model.eval()
    with torch.no_grad():
        atomic_relations_embedding = model(atomic_relations.tolist(), is_eval=True)
    
    d = x.shape[1]
    for seed, num_cluster in enumerate(args.num_cluster):
        k = int(num_cluster)
        if im2cluster==[]:
            centroids, assignments, avg_distance = kmeans_cuda(x, k, metric="cos", verbosity=1, seed=args.seed, average_distance=True)
            # print(centroids)
            # print(len(centroids)) # 3446
            # print(assignments)
            # print(len(assignments))
            # print(avg_distance) # 0.2061482071876526
            
            im2cluster = assignments # [3379 2510 3083 ... 2933 2190 1752] 23888
            # print("im2cluster的最小值",min(im2cluster),max(im2cluster))
            
            # 计算每个样本到其对应聚类中心的余弦相似度
            D = F.cosine_similarity(x, torch.tensor(centroids[assignments]),dim=1).unsqueeze(1).cpu().numpy()
            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im, i in enumerate(im2cluster):
                Dcluster[i].append(1-D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i, dist in enumerate(Dcluster):
                if len(dist) > 1:
                    d = np.asarray(dist).mean() / np.log(len(dist) + args.alpha) # alpha = 10
                    # d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10) # alpha = 10
                    density[i] = d

            # if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i, dist in enumerate(Dcluster):
                if len(dist) <= 1:
                    density[i] = dmax

            density = density.clip(np.percentile(density, 10),
                                np.percentile(density, 90))  # clamp extreme values for stability
            density = args.temperature * density / density.mean()  # scale the mean to temperature
        
        
        
        ##############################################
        # get cluster centroids 根据关系重新计算质心
        im2cluster_tensor = torch.tensor(im2cluster).to(device=args.device_ids[0])
        pos_relation_indexs_tensor = torch.tensor(pos_relation_indexs).to(device=args.device_ids[0])
        # atomic_relations_tensor = torch.tensor(atomic_relations).to(device=args.device_ids[0])
        
        # 用于保存每个簇的质心
        pos_relation_centroids = []

        # 获取唯一的簇标签
        alllabel = sorted(list(set(im2cluster)), reverse=False)

        # 对每个簇计算质心
        for cluster_id in tqdm(alllabel, desc="calculating centroids"):
            cluster_indices = (im2cluster_tensor == cluster_id).nonzero(as_tuple=True)[0]  # 获取当前簇的索引
            if len(cluster_indices) == 0:
                continue  # 如果当前簇没有样本，跳过

            # 关系索引
            cluster_relations_indices = pos_relation_indexs_tensor[cluster_indices]
            
            # 获取关系
            cluster_relations = atomic_relations[cluster_relations_indices.cpu().numpy()]

            # 计算关系的 embedding
            model.eval()
            with torch.no_grad():
                cluster_relations_embedding = model(cluster_relations.tolist(), is_eval=True)

            # 计算质心
            centroid = compute_centroid(cluster_relations_embedding)

            pos_relation_centroids.append(centroid)

        # 将质心转换为 NumPy 数组
        centroids = torch.stack(pos_relation_centroids).cpu().numpy()
        
        
        

        #################################################
        # Step 3: 为每个簇选择 top-k 最近簇中的样本作为负样本
        # 计算所有簇的余弦相似度
        pos_relation_centroids = torch.stack(pos_relation_centroids).to(device=args.device_ids[0])  # 移动到 GPU
        # 设定 batch_size
        batch_size = 100  # 可以根据你的 GPU 内存大小调整
        num_clusters = pos_relation_centroids.size(0)
        negative_samples = []

        # 分块计算余弦相似度
        for i in tqdm(range(0, num_clusters, batch_size), desc="calculating similarities"):
            end = min(i + batch_size, num_clusters)
            batch_centroids = pos_relation_centroids[i:end]

            # 计算与所有质心的余弦相似度
            similarities = F.cosine_similarity(batch_centroids.unsqueeze(1), pos_relation_centroids.unsqueeze(0), dim=-1)

            for j in range(batch_centroids.size(0)):
                # 选择 top-k 最近的簇
                top_k_clusters = torch.topk(similarities[j], args.topk_neg_cluster + 1, largest=True).indices[1:]  # 排除自身
                negative_samples.append(top_k_clusters.cpu().numpy())

        # 将结果转换为 NumPy 数组
        negative_samples = np.array(negative_samples)  # 每个簇对应的最近簇
        
        # print(negative_samples.shape) # 

        # 硬负样本采样
        neg_samples_index = []
        for i,sample_cluster_label in enumerate(tqdm(im2cluster,desc="finding negative samples")):
            neg_clusters = negative_samples[alllabel.index(sample_cluster_label)] # 每个样本需要取的簇
            neg_samples_idx = []
            for neg_cluster_idx in neg_clusters:
                # print(alllabel[neg_cluster_idx])
                neg_sample_indices = np.where(np.array(im2cluster) == alllabel[neg_cluster_idx])[0]
                neg_sample_index = np.random.choice(neg_sample_indices, size=1)
                neg_sample_id = pos_relation_indexs[neg_sample_index]
                trytime = 0
                while neg_sample_id==pos_relation_indexs[i] and trytime<3:
                    neg_sample_index = np.random.choice(neg_sample_indices, size=1) # 随机抽取一个负样本
                    neg_sample_id = pos_relation_indexs[neg_sample_index]
                    trytime+=1
                neg_samples_idx.append(neg_sample_id) # 取出样本索引
            neg_samples_index.append(neg_samples_idx)
        neg_samples_index = np.array(neg_samples_index) # 每个样本对应的最近簇里的样本
        # print(neg_samples_index.shape) # (12383, 3, 1)
        neg_cluster_index = np.array(negative_samples) # # 每个簇对应的最近簇
        # print(neg_cluster_index.shape) # (8133, 3)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(device=args.device_ids[0])
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).to(device=args.device_ids[0])
        density = torch.Tensor(density).to(device=args.device_ids[0])

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)
        results['neg_samples_index'].append(neg_samples_index)
        results['neg_cluster_index'].append(neg_cluster_index)
        results['alllabel'].append(alllabel)

    return results


# 计算 top-k 最近簇
def find_top_k_clusters(cluster_centroids, target_centroid, top_k=5):
    # similarities = cosine_similarity([target_centroid], cluster_centroids)[0]
    # # 排序并选择 top-k 最近的簇
    # top_k_indices = np.argsort(similarities)[::-1][1:top_k + 1]
    # return top_k_indices
    cluster_centroids_gpu = torch.tensor(np.array(cluster_centroids)).to(device=args.device_ids[0])
    target_centroid_gpu = torch.tensor(np.array(target_centroid)).to(device=args.device_ids[0]) #.unsqueeze(0)
    
    # 计算余弦相似度
    similarities = F.cosine_similarity(target_centroid_gpu, cluster_centroids_gpu)
    
    # 排序并选择 top-k 最近的簇
    top_k_indices = torch.argsort(similarities, descending=True)[1:top_k + 1]
    return top_k_indices.cpu().numpy()  # 从GPU移回CPU

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, atomic_relations, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    loop = tqdm(enumerate(train_loader), total =len(train_loader))
    end = time.time()
    for i, (batch, index) in loop: #enumerate(train_loader)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        q,positive_embedding,negative_embedding,pos_prototypes,neg_prototypes,temp_proto = model(im_q=batch["question"], atomic_relations=atomic_relations,im_k=batch["pos_relation_indexs"],cluster_result=cluster_result, index=index)
        # InfoNCE loss
        # loss = criterion(output, target)
        loss = criterion(q, positive_embedding,negative_embedding,pos_prototypes,neg_prototypes,temp_proto)
        # print("loss0")
        # print(loss) #tensor(1.4352, device='cuda:0', grad_fn=<NllLossBackward0>


        losses.update(loss.item(), len(batch['question']))
        # acc = accuracy(output, target)[0]
        # acc_inst.update(acc[0], batch['question'].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #更新信息
        loop.set_description(f'Epoch [{epoch+1}/{args.num_epochs}]')
        loop.set_postfix(loss=loss)
        
        # if i % args.print_freq == 0:
        #     progress.display(i)

    return loss


def validate(dev_loader, model, atomic_relations, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')

    progress = ProgressMeter(
        len(dev_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    # switch to validate mode
    model.eval()

    loop = tqdm(enumerate(dev_loader), total =len(dev_loader))
    with torch.no_grad():
        end = time.time()
        atomic_relations_embeddings = model(im_q=list(atomic_relations), is_eval=True)
        # acc = 0
        K = [1,3,5,10]
        hits = [0 for i in range(len(K))]
        for i, (batch, index) in loop: # enumerate(dev_loader)

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            question_embeddings = model(im_q=batch["question"], is_eval=True)
            # print(question_embeddings.shape)
            # print(atomic_relations_embeddings.shape)


            similarity = F.cosine_similarity(question_embeddings.unsqueeze(1), atomic_relations_embeddings.unsqueeze(0), dim=2)

            # print(similarity.shape) # torch.Size([5, 1986])
            # 获取 top-k 个结果
            pos_relation_indexs = batch["pos_relation_indexs"].unsqueeze(1)  # 使形状匹配 torch.Size([5, 1])
            for kidx,k in enumerate(K): # 你希望获取的数量
                _, topk_indices = torch.topk(similarity, k, dim=1,largest=True)
                # print(topk_indices.shape) # torch.Size([5, 3])
                # print(batch["pos_relation_indexs"].shape)
                # print(pos_relation_indexs.shape)
                matches = topk_indices == pos_relation_indexs.to(device=args.device_ids[0])  # 进行比较
                hits[kidx] += matches.sum().item()
                # acc += matches.sum().item()  # 统计匹配的个数并累加

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #更新信息
            loop.set_description(f'Epoch [{epoch+1}/{args.num_epochs}]')
            loop.set_postfix(hits=hits)
            # if i % args.print_freq == 0:
            #     progress.display(i)

    return hits


# SimCLR 的对比损失函数
class OurLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(OurLoss, self).__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
    
    def InfoNCE(self,anchor, positive, negatives,temperature):
        batch_size = anchor.size(0)
        
        # 将正样本和负样本拼接在一起作为对比样本
        # print("anchor:",anchor.shape)
        # print("pos:",positive.shape)
        # print("neg:",negatives.shape)
        positives_and_negatives = torch.cat((positive.unsqueeze(1) , negatives), dim=1) 
        # print("pos+neg:",positives_and_negatives.shape)

        # 计算 anchor 和其他样本的相似度
        similarities = self.cosine_sim(anchor.unsqueeze(1), positives_and_negatives) / temperature
        # print("sim:",similarities.shape)
        
        # print(similarities.shape) # torch.Size([16, 17])
        
        # 对相似度进行 softmax
        logits = torch.exp(similarities)
        # print("exp:",logits.shape)
        logits = logits.sum(dim=1, keepdim=True)
        # print("sum:",logits.shape)
        # logits = logits / logits_sum  # softmax

        # 正样本的相似度得分
        if type(temperature)==float:
            pos_sim = torch.exp(self.cosine_sim(anchor, positive) / temperature)
        else:
            # print(temperature.shape)
            pos_sim = torch.exp(self.cosine_sim(anchor, positive) / temperature[:,0])
        # print("pos_sim:",pos_sim.shape) #torch.Size([16])
        # print(logits.shape) # torch.Size([16, 1])

        # 计算最终的 InfoNCE 损失
        loss = -torch.log(pos_sim / logits.squeeze())
        # print("loss:",loss.shape)
        return loss
        

    def forward(self, anchor, positive, negatives,pos_prototypes,neg_prototypes,density):
        loss1 = self.InfoNCE(anchor, positive, negatives,self.temperature)
        loss2 = self.InfoNCE(anchor, pos_prototypes,neg_prototypes,density)
        # loss2 = self.InfoNCE(positive, pos_prototypes,neg_prototypes,density)
        return (loss1+loss2).mean()



def main(args):
    # create model
    print("=> creating model ...")
    model = pcl.builder.MoCo(args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.device_ids, args.mlp)
    # model_path = r"/root/autodl-tmp/EncoderModel/abstract_cluster_allr_16_pos_fixnewneg_1026/question_model_epoch23.pth"
    # model.load_state_dict(torch.load(model_path,map_location="cuda"))
    # print(model)

    # 加载数据
    print("=> loading dataset ...")
    data_path = args.train_data_path
    dataset = CustomDataset(data_path)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    data_path_dev = args.dev_data_path
    dataset_dev = CustomDataset(data_path_dev)
    dev_dataloader = DataLoader(dataset_dev, batch_size=args.dev_batch_size, shuffle=True)
    
    # data_path_test = args.test_data_path
    # dataset_test = TestDataset(data_path_test)
    # test_dataloader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=True)

    atomic_relation_path = args.atomic_relation_path
    with open(atomic_relation_path,'r') as f:
        atomic_relations = np.array([line.strip().replace(',',', ') for line in f.readlines()]) # 所有的关系的列表

    # define loss function (criterion) and optimizer
    criterion = OurLoss(temperature=args.temperature).to(device=args.device_ids[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 开始训练
    print("=> start training ...")
    im2cluster = []
    density = []
    for epoch in range(args.num_epochs):
        # # print(len(test_dataloader)) #312
        # # print(len(test_dataloader.dataset)) #1559
        # 首先聚类
        cluster_result = None
        if epoch >= args.warmup_epoch:  # 20
            print("start clustring ...")
            # compute momentum features for center-cropped images
            features,pos_relation_indexs = compute_features(train_dataloader, model,args)
            # print(features.shape) 
            # print("pos_relation_indexs:",pos_relation_indexs.shape) 

            if args.gpu == 0:
                # features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice
                # features = features.numpy()
                cluster_result = run_kmeans(features, pos_relation_indexs, atomic_relations, im2cluster,density,model,args)  # run kmeans clustering on master node
                im2cluster = cluster_result['im2cluster'][0]
                im2cluster = im2cluster.cpu().numpy().tolist()
                density = cluster_result['density'][0]
                density = density.cpu().numpy()
                # print(cluster_result)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        total_loss = train(train_dataloader, model, atomic_relations, criterion, optimizer, epoch, args, cluster_result)

        # validate
        # total_accuracy = validate(dev_dataloader, model, atomic_relations, epoch, args)
        # test_accuracy = validate(test_dataloader, model, atomic_relations, epoch, args)
        total_hits = validate(dev_dataloader, model, atomic_relations, epoch, args)
        # test_hits = validate(test_dataloader, model, atomic_relations, epoch, args)
        
        # print(f"Epoch {epoch + 1}/{args.num_epochs}, Acc_test: {test_accuracy / 1639}")
        # print(f"Epoch {epoch + 1}/{args.num_epochs}, Acc_dev: {total_accuracy / len(dev_dataloader.dataset)}")
        # print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / len(train_dataloader)}, Acc_dev: {total_accuracy / len(dev_dataloader.dataset)}")
        # print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / len(train_dataloader)}, Acc_dev: {total_accuracy / len(dev_dataloader.dataset)}, Acc_test: {test_accuracy / 1639}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / len(train_dataloader)}, Acc_dev: {[hit/len(dev_dataloader.dataset) for hit in total_hits]}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Acc_dev: {[hit/len(dev_dataloader.dataset) for hit in total_hits]}")
        

        torch.save(model.state_dict(),args.savedir+'question_model_epoch' + str(epoch) + '.pth')



if __name__=="__main__":
    # 加载参数
    args = {
        "alpha":10,   # 可调
        "print_freq":1,
        "num_epochs":50,
        "lr": 2e-6,# 2e-6,  # 可调
        "temperature": 0.5,#0.2, # 可调
        "num_cluster": [3446],
        "gpu": 0,
        "topk_neg_cluster": 16, # 可调
        "warmup_epoch": 0,
        "low_dim": 768,
        "seed": 2024,
        "pcl_r":16, #queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)
        "moco_m":0.999,   # 可调
        "mlp":False,
        "train_batch_size":16,
        "dev_batch_size":5,
        "test_batch_size":5,
        "cos":False,
        "schedule":[120, 160],   # 可调
        "device_ids":[0,1,2,3],
        "train_data_path":r"./dataset/WebQSP/dataset_train.jsonl",
        "dev_data_path": r"./dataset/WebQSP/dataset_dev.jsonl",
        # "test_data_path":r"./dataset/WebQSP/dataset_test.jsonl",
        "atomic_relation_path": r"./dataset/WebQSP/atomic_rtypes.txt",
        "savedir": r"./checkpoints/"
    }
    args = EasyDict(args)
    # 示例：设置随机种子
    set_seed(args.seed)
    main(args)

