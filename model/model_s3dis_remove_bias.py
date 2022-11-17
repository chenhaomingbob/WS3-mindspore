# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/06/23
    Description:
         v2. 往graph mode上改，但忽略了sp loss
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.ops import composite as C
from mindspore.ops import operations as op
from mindspore import Tensor, Parameter, ms_function
from mindspore import dtype as mstype

from .base_model_remove_bias import SharedMLP, LocalFeatureAggregation
import time
from time import strftime, localtime


class RandLANet_S3DIS(nn.Cell):
    def __init__(self, d_in, num_classes):
        super(RandLANet_S3DIS, self).__init__()

        self.fc_start = nn.Dense(d_in, 8, has_bias=False)
        self.bn_start = nn.SequentialCell([
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        ])

        # encoding layers
        self.encoder = nn.CellList([
            LocalFeatureAggregation(8, 16),
            LocalFeatureAggregation(32, 64),
            LocalFeatureAggregation(128, 128),
            LocalFeatureAggregation(256, 256),
            LocalFeatureAggregation(512, 512)
        ])

        self.mlp = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(0.2))

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.LeakyReLU(0.2)
        )
        self.decoder = nn.CellList([
            SharedMLP(1536, 512, **decoder_kwargs),
            SharedMLP(768, 256, **decoder_kwargs),
            SharedMLP(384, 128, **decoder_kwargs),
            SharedMLP(160, 32, **decoder_kwargs),
            SharedMLP(64, 32, **decoder_kwargs)
        ])

        self.fc_end_fc1 = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.fc_end_fc2 = SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.fc_end_drop = nn.Dropout()
        self.fc_end_fc3 = SharedMLP(32, num_classes)
        # # final semantic prediction
        # self.fc_end = nn.SequentialCell([
        #     SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(0.2)),
        #     SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(0.2)),
        #     nn.Dropout(),
        #     SharedMLP(32, num_classes)
        # ])

    def construct(self, xyz, feature, neighbor_idx, sub_idx, interp_idx):
        r"""
            construct method

            Parameters
            ----------
            xyz: list of ms.Tensor, shape (num_layer, B, N_layer, 3), each layer xyz
            feature: ms.Tensor, shape (B, N, d), input feature [xyz ; feature]
            neighbor_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer knn neighbor idx
            sub_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer pooling idx
            interp_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 1), each layer interp idx

            Returns
            -------
            ms.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        # print(feature.shape)
        # a_time = time.time()
        feature = self.fc_start(feature).swapaxes(-2, -1).expand_dims(-1)  # (B, N, 6) -> (B, 8, N, 1)
        feature = self.bn_start(feature)  # shape (B, 8, N, 1)
        # b_time = time.time()
        # print(f"[model_s3dis.py] fc_start & bn_start. Time :{b_time - a_time}")
        # <<<<<<<<<< ENCODER

        f_stack = []
        for i in range(5):
            # at iteration i, feature.shape = (B, d_layer, N_layer, 1)
            encoder_begin_time = time.time()
            f_encoder_i = self.encoder[i](xyz[i], feature,
                                          neighbor_idx[i])  # (B,40960,3)  (4, 8, 40960, 1) (4, 40960, 16)
            encoder_end_time = time.time()
            # print(f"[model_s3dis.py] encoder {i}. Time :{encoder_end_time - encoder_begin_time}s")
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            random_sample_time = time.time()
            # print(f"[model_s3dis.py] random_sample {i}. Time :{random_sample_time - encoder_end_time}s")
            feature = f_sampled_i
            if i == 0:
                f_stack.append(f_encoder_i)
            f_stack.append(f_sampled_i)
            # print(f"[model_s3dis.py] append {i}. Time :{time.time() - random_sample_time}s")
        # c_time = time.time()
        # print(f"[model_s3dis.py] encoder & random_sample. Time :{c_time - b_time}s")
        # # >>>>>>>>>> ENCODER

        feature = self.mlp(f_stack[-1])  # [B, d, N, 1]

        # <<<<<<<<<< DECODER

        f_decoder_list = []
        for j in range(5):
            f_interp_i = self.random_sample(feature, interp_idx[-j - 1])  # [B, d, n, 1]
            cat = P.Concat(1)
            f_decoder_i = self.decoder[j](cat((f_stack[-j - 2], f_interp_i)))
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # d_time = time.time()
        # print(f"[model_s3dis.py] random_sample & decoder. Time :{d_time - c_time}s")
        # >>>>>>>>>> DECODER
        f_layer_fc1 = self.fc_end_fc1(f_decoder_list[-1])
        f_layer_fc2 = self.fc_end_fc2(f_layer_fc1)
        f_layer_drop = self.fc_end_drop(f_layer_fc2)
        f_layer_fc3 = self.fc_end_fc3(f_layer_drop)

        # e_time = time.time()
        # print(f"[model_s3dis.py] fc_end. Time :{e_time - d_time}s")

        f_layer_fc2, f_layer_fc3 = f_layer_fc2.swapaxes(1, 3), f_layer_fc3.swapaxes(1, 3)
        f_layer_out = P.Concat(axis=-1)([f_layer_fc3, f_layer_fc2])
        f_out = f_layer_out.squeeze(1)  # (B,N_points,13+32)

        return f_out  # (B,N_points,45)

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, d, N, 1] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, d, N', 1] pooled features matrix
        """
        b, d = feature.shape[:2]
        n_ = pool_idx.shape[1]
        # [B, N', max_num] --> [B, d, N', max_num]
        # pool_idx = P.repeat_elements(pool_idx.expand_dims(1), feature.shape[1], 1)
        pool_idx = P.Tile()(pool_idx.expand_dims(1), (1, feature.shape[1], 1, 1))
        # [B, d, N', max_num] --> [B, d, N'*max_num]
        pool_idx = pool_idx.reshape((b, d, -1))
        pool_features = P.GatherD()(feature.squeeze(-1), -1, pool_idx.astype(mstype.int32))
        pool_features = pool_features.reshape((b, d, n_, -1))
        pool_features = P.ReduceMax(keep_dims=True)(pool_features, -1)  # [B, d, N', 1]
        return pool_features


class RandLA_S3DIS_WithLoss(nn.Cell):
    """RadnLA-net with loss"""

    def __init__(self, network, weights, num_classes, ignored_label_indexs, c_epoch, loss3_type, topk):
        super(RandLA_S3DIS_WithLoss, self).__init__()
        self.network = network
        self.weights = Tensor(weights, dtype=mstype.float16)
        # self.weights = Tensor(weights, dtype=mstype.float32)
        self.num_classes = num_classes
        self.ignored_label_inds = ignored_label_indexs
        self.c_epoch = c_epoch
        self.loss3_type = loss3_type
        self.topk = topk

        self.c_epoch_k = Tensor(self.c_epoch, dtype=mstype.float16)
        # self.c_epoch_k = Tensor(self.c_epoch, dtype=mstype.float32)
        #
        self.onehot = nn.OneHot(depth=num_classes, dtype=mstype.float16)
        # self.onehot = nn.OneHot(depth=num_classes, dtype=mstype.float32)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, feature, feature2, labels, input_inds, cloud_inds,
                  p0, p1, p2, p3, p4, n0, n1, n2, n3, n4, pl0,
                  pl1, pl2, pl3, pl4, u0, u1, u2, u3, u4):
        # data_begin_time = time.time()
        xyz = [p0, p1, p2, p3, p4]
        neighbor_idx = [n0, n1, n2, n3, n4]
        sub_idx = [pl0, pl1, pl2, pl3, pl4]
        interp_idx = [u0, u1, u2, u3, u4]

        # network_begin_time = time.time()
        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[model_s3dis.py] data_prepare 耗时: {network_begin_time - data_begin_time}s")
        logits_embed = self.network(xyz, feature, neighbor_idx, sub_idx, interp_idx)
        # network_end_time = time.time()
        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[model_s3dis.py] network 耗时: {network_end_time - network_begin_time}s")
        # pred_embed = logits  # (B, N, 45)
        xyzrgb = feature2  # (B, N, 6)
        labels = labels  # (B,N)

        # 当c_epoch_k==0时，只需要计算有GT标注点的loss值
        logits = logits_embed[..., :self.num_classes]  # (B,N,45) -> (B,N,13)
        pred_embed = logits_embed[..., self.num_classes:]  # (B,N,45) -> (B,N,32)
        logits = logits.reshape((-1, self.num_classes))
        pred_embed = pred_embed.reshape((-1, 32))
        # logit = logits.reshape((-1, self.num_classes))  # (B*N,13)
        labels = labels.reshape((-1,))  # (b,n) -> (b*n)
        xyzrgb = xyzrgb.reshape((-1, 6))  # (b,n,6) -> (b*n,6)

        # Boolean mask of points that should be ignored
        # (B*N,)
        ignore_mask = P.zeros_like(labels).astype(mstype.bool_)
        for ign_label in self.ignored_label_inds:
            ignore_mask = P.logical_or(ignore_mask, P.equal(labels, ign_label))  #
        # 0为无效,1为有效
        valid_mask = P.logical_not(ignore_mask)  # (B*N,)

        # (B*N,13)
        one_hot_labels = self.onehot(labels)  # (B*N,) -> (B*N,13)
        weights = self.weights * one_hot_labels * valid_mask.reshape(-1, 1)  # (B*N,13)
        # 输出Tensor各维度上的和，以达到对所有维度进行归约的目的。也可以对指定维度进行求和归约
        # (B*N, 13) -> (B*N,)
        weights = P.ReduceSum()(weights, 1)  #
        # (B*N,) and (B*N,13) ->
        unweighted_loss = self.loss_fn(logits, one_hot_labels)
        weighted_loss = unweighted_loss * weights
        weighted_loss = weighted_loss * valid_mask
        CE_loss = P.ReduceSum()(weighted_loss)
        num_valid_points = P.ReduceSum()(valid_mask.astype(weighted_loss.dtype))
        # num_valid_points = int(P.count_nonzero(valid_mask.astype(mstype.int32)))
        # CE_loss = P.ReduceSum()(weighted_loss).sum()
        CE_loss = CE_loss / num_valid_points
        ###

        if self.c_epoch_k == 0:
            loss = CE_loss
        else:
            # loss = CE_loss
            SP_loss = self.get_sp_loss_by_mask(pred_embed, logits, one_hot_labels, valid_mask,
                                               self.topk) * self.c_epoch_k
            loss = CE_loss + SP_loss

        # loss_end_time = time.time()
        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[model_s3dis.py] loss 耗时: {loss_end_time - network_end_time}s")
        #
        # print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
        #       f"[model_s3dis.py] RandLA_S3DIS_WithLoss 耗时: {loss_end_time - data_begin_time}s")
        return loss

    def get_sp_loss_by_mask(self, embed, logits, one_hot_label, valid_mask, topk):
        """

        Args:
            embed:
            logits:
            one_hot_label:
            valid_mask:
            topk:

        Returns:

        """
        invalid_mask = P.logical_not(valid_mask)  # (B*N,)
        num_invalid_points = int(P.count_nonzero(invalid_mask.astype(mstype.int32)))
        topk += num_invalid_points
        # 点类别的数量
        num_class = one_hot_label.shape[1]  # scalar: 13

        valid_one_hot_label = one_hot_label * valid_mask.reshape(-1, 1)  # (B*N,13)
        valid_embed = embed * valid_mask.reshape(-1, 1)  # (B*N,32)
        invalid_embed = embed * invalid_mask.reshape(-1, 1)  # (B*N,32)
        # => 将有效点的label矩阵进行转置:[M,class_num] -> [class_num,M] (13,B*N)
        valid_one_hot_label_T = P.transpose(valid_one_hot_label, (1, 0))
        # => 每一行那个类有那些点:[class_num,B*N] * [B*N,dim] -> [class_num,dim]
        sum_embed = P.matmul(valid_one_hot_label_T, valid_embed)
        # => 求class的平均嵌入:[class_num,dim]
        # mean_embed 为每个类别的embedding，如果这个类别没有样本，则embedding全为0
        mean_embed = sum_embed / (P.reduce_sum(valid_one_hot_label_T, axis=1).reshape(-1, 1) + 0.001)
        # => 求unlabelled points 与 class embedding的相似度
        # adj_matrix 欧式距离，距离越大说明越不相似  [N,M]
        adj_matrix = self.double_feature(invalid_embed, mean_embed)
        # adj_matrix = RandLA_S3DIS_WithLoss.double_feature(invalid_embed, mean_embed)

        # => 稀疏点，N个点中M分别找K和最相似的，把没有和任何M相似的去掉（说明这些点不容易分）
        neg_adj = -adj_matrix  # (B*N,13) 取负
        # 转置为了下一步 [N, M] -> [M,N] (M是有标签的点)
        neg_adj_t = P.transpose(neg_adj, (1, 0))  # (13,B*N)
        # 取最不相似的 top k个点
        _, nn_idx = P.TopK()(neg_adj_t, topk)
        s = P.shape(neg_adj_t)  # s (M,N)
        row_idx = P.tile(P.expand_dims(P.arange(s[0]), 1), (1, topk))
        ones_idx = P.Stack(axis=1)([row_idx.reshape([-1]), nn_idx.reshape([-1])])
        res = P.scatter_nd(ones_idx, P.ones(s[0] * topk, neg_adj_t.dtype), s)
        nn_idx_multi_hot = P.transpose(res, (1, 0))  # [N,M] multi_hot

        new_valid_mask = P.reduce_sum(nn_idx_multi_hot, axis=1) > 0  # (B*N,)
        new_valid_mask = new_valid_mask.reshape(-1, 1)  # (B*N,1)
        num_new_valid_mask = int(P.count_nonzero(new_valid_mask.astype(mstype.int32)))

        w_ij = P.exp(-1.0 * adj_matrix)  # (B*N,13)
        w_ij = w_ij * new_valid_mask  # (B*N,13)
        w_ij_label = nn_idx_multi_hot * new_valid_mask  # (B*N,13)
        w_ij = P.mul(w_ij, w_ij_label)  # (B*N,13)

        new_soft_label_hot = nn.Softmax(axis=-1)(w_ij)  # (B*N,13)
        top1 = new_soft_label_hot.argmax(axis=-1)
        soft_label_mask = self.onehot(top1)
        # soft_label_mask = P.OneHot()(top1, num_class,
        #                              Tensor(1.0, dtype=mstype.float16),
        #                              Tensor(0.0, dtype=mstype.float16))
        new_soft_label_hot = P.mul(new_soft_label_hot, soft_label_mask)

        logits = logits * new_valid_mask
        new_soft_label_hot = new_soft_label_hot * new_valid_mask
        loss = nn.SoftmaxCrossEntropyWithLogits()(logits, new_soft_label_hot)
        loss = loss.sum() / num_new_valid_mask

        return loss

    # @ms_function()
    # @staticmethod
    def double_feature(self, point_feature1, point_feature2):
        """
        Compute pairwise distance of a point cloud.
        Args:
        [N,C],[M,C]
            point_cloud: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        point2_transpose = P.transpose(point_feature2, (1, 0))  # [C, M]
        point_inner = P.matmul(point_feature1, point2_transpose)  # [N,M]
        point_inner = -2 * point_inner

        point1_square = P.ReduceSum(keep_dims=True)(P.square(point_feature1), axis=-1)
        point2_square = P.ReduceSum(keep_dims=True)(P.square(point_feature2), axis=-1)

        point2_square_transpose = P.transpose(point2_square, (1, 0))  # [1,M]
        adj_matrix = point1_square + point_inner + point2_square_transpose

        return adj_matrix


class TrainingWrapper(nn.Cell):
    """Training wrapper."""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network_logits = self.network.network
        self.network.set_grad()
        self.opt_weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, xyz, feature, neighbor_idx, sub_idx, interp_idx, labels):
        """Build a forward graph"""

        # loss calculate
        loss = self.network(xyz, feature, neighbor_idx, sub_idx, interp_idx, labels)
        logit = self.network_logits(xyz, feature, neighbor_idx, sub_idx, interp_idx)

        # opt update
        opt_weights = self.opt_weights
        sens = op.Fill()(op.DType()(loss), op.Shape()(loss), self.sens)
        grads = self.grad(self.network, opt_weights)(xyz, feature, neighbor_idx, sub_idx, interp_idx, labels, sens)
        res = P.depend(loss, self.optimizer(grads))
        return res, logit
