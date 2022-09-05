import os.path as osp
import time
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import Model
from scrfd import SCRFD
from utils import norm_crop

class PyFAT:
    def __init__(self, N=10):
        self.N = N # 生成图片数量
        self.num_iter = 41 # 迭代次数
        self.image_size = 112 # 图像大小

        # 判断是否在GPU上运行
        if torch.cuda.is_available():
            device = torch.device("cuda:3")
            self.is_cuda = True
        else:
            device = 'cpu'
            self.is_cuda = False
        self.device = device

    def get_mask(self, x_scale=1.0, y_scale=1.0):
        # 从固定的五官点制作对应的mask进行对抗
        left_eye = [38.2946, 51.6963]
        right_eye = [73.5318, 51.5014]
        nose = [56.0252, 82.7366]

        eye_base_len = int(0.09746 * self.image_size)
        nose_base_len = int(0.15492 * self.image_size)

        eye_expand_len = 4

        left_eye_xMin = int(left_eye[0] - eye_base_len * x_scale / 2) - eye_expand_len
        left_eye_xMax = int(left_eye[0] + eye_base_len * x_scale / 2) + eye_expand_len
        left_eye_yMin = int(left_eye[1] - eye_base_len * y_scale / 2)
        left_eye_yMax = int(left_eye[1] + eye_base_len * y_scale / 2)

        right_eye_xMin = int(right_eye[0] - eye_base_len * x_scale / 2) - eye_expand_len
        right_eye_xMax = int(right_eye[0] + eye_base_len * x_scale / 2) + eye_expand_len
        right_eye_yMin = int(right_eye[1] - eye_base_len * y_scale / 2)
        right_eye_yMax = int(right_eye[1] + eye_base_len * y_scale / 2)

        nose_xMin = int(nose[0] - nose_base_len * y_scale / 2)
        nose_xMax = int(nose[0] + nose_base_len * y_scale / 2)
        nose_yMin = int(nose[1] - nose_base_len * x_scale / 2)
        nose_yMax = int(nose[1] + nose_base_len * x_scale / 2)

        if nose_yMin < (left_eye_yMax - 1):
            nose_yMin = left_eye_yMax - 3
        if nose_yMin < (right_eye_yMax - 1):
            nose_yMin = right_eye_yMax - 3

        mask_np = np.ones((self.image_size, self.image_size, 3), dtype=np.float)
        mask_np[left_eye_yMin:left_eye_yMax, left_eye_xMin:left_eye_xMax, :] = 0
        mask_np[right_eye_yMin:right_eye_yMax, right_eye_xMin:right_eye_xMax, :] = 0
        mask_np[nose_yMin:nose_yMax, nose_xMin:nose_xMax, :] = 0

        mask_out = torch.Tensor(mask_np.transpose(2, 0, 1)).unsqueeze(0)
        mask_out = F.interpolate(mask_out, self.image_size).to(self.device)
        choose_area = torch.sum(mask_out != torch.ones((self.image_size, self.image_size), device=self.device)) / 3

        return mask_out, choose_area

    def load(self, assets_path):
        # 导入人脸检测模型进行检测
        detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
        ctx_id = -1 if not self.is_cuda else 0
        # 定义人脸检测模型相关参数
        detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))

        # 定义人脸识别模型并加载参数
        tf_nas_a = Model.TF_NAS_A(7, 7, 512, drop_ratio=0)
        tf_nas_a.load_state_dict(torch.load('./assets/tf-nas-a.pt', map_location='cpu'))
        tf_nas_a.eval().to(self.device)
        tf_nas_a_ex = Model.TF_NAS_A(7, 7, 512, drop_ratio=0)
        tf_nas_a_ex.load_state_dict(torch.load('./assets/tf-nas-a-ex.pth', map_location='cpu'))
        tf_nas_a_ex.eval().to(self.device)
        r50 = Model.iresnet50()
        r50.load_state_dict(torch.load('./assets/w600k_r50.pth', map_location='cpu'))
        r50.eval().to(self.device)
        repvgg = Model.create_RepVGG_A0(deploy=False)
        repvgg.load_state_dict(torch.load('./assets/repvgg.pt', map_location='cpu'))
        repvgg.eval().to(self.device)
        hrnet = Model.get_cls_net()
        hrnet.load_state_dict(torch.load('./assets/hrnet.pt', map_location='cpu'))
        hrnet.eval().to(self.device)
        ghost = Model.GhostNet()
        ghost.load_state_dict(torch.load('./assets/ghost.pt', map_location='cpu'))
        ghost.eval().to(self.device)
        # mobilenet1 = Model.FaceMobileNet(512)
        # mobilenet1.load_state_dict(torch.load('./assets/mobilenet_specific.pth', map_location='cpu'))
        # mobilenet1.eval().to(self.device)
        # 将定义的人脸检测器和模型，人脸掩码定义
        self.detector = detector
        self.model = nn.ModuleList([r50, tf_nas_a, repvgg, hrnet, ghost, tf_nas_a_ex])

        # 考虑是否需要对模型求梯度
        for model in self.model:
            for param in model.parameters():
                param.requires_grad = False

    def size(self):
        return self.N

    def generate(self, im_a, im_v, n):
        h, w, c = im_a.shape
        assert len(im_a.shape) == 3
        assert len(im_v.shape) == 3

        # 截取人脸区域进行预处理
        bboxes, kpss = self.detector.detect(im_a, max_num=1)
        if bboxes.shape[0] == 0:
            return im_a
        att_img, M = norm_crop(im_a, kpss[0], image_size=self.image_size)
        bboxes, kpss = self.detector.detect(im_v, max_num=1)
        if bboxes.shape[0] == 0:
            return im_a
        vic_img, _ = norm_crop(im_v, kpss[0], image_size=self.image_size)
        att_img = att_img[:, :, ::-1]
        vic_img = vic_img[:, :, ::-1]

        # 定义CosineSimilarity以及LayerNorm
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6).to(self.device)
        ln = nn.LayerNorm(512).to(self.device)

        # 获取被攻击者图像特征
        vic_img = torch.Tensor(vic_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        vic_img.div_(255).sub_(0.5).div_(0.5)
        vic_feats = []
        # 将被攻击者特征通过LN层后通过列表保存
        for model in self.model:
            temp = model.forward(vic_img)
            vic_feats.append(ln(temp))

        att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        att_img.div_(255).sub_(0.5).div_(0.5)
        att_img_ = att_img.clone()

        # 定义初始参数
        alpha = 0.035
        count = 0
        y_scale_start = 2.07 - n * 0.13
        x_scale_start = 2.07 - n * 0.13
        y_scale_item = y_scale_start
        x_scale_item = x_scale_start

        att_img_current = att_img.clone()
        att_img_current.requires_grad = True
        for model in self.model:
            model.zero_grad()

        start_time = time.time()
        current_face_mask, choose_area = self.get_mask(x_scale=x_scale_item, y_scale=y_scale_item)

        # I-FGSM
        for k in range(self.num_iter):
            # 先大后小的攻击步长
            if k > 8:
                alpha = 0.008

            count += 1
            adv_images = att_img_current.clone()

            # 将攻击者的图像输入到特征提取并保存到列表
            adv_feats = []
            for model in self.model:
                temp = model.forward(adv_images)
                adv_feats.append(ln(temp))

            # 将被攻击者的特征和攻击者的特征拼接求余弦相似度
            similarity_score = []
            for _, model in enumerate(self.model):
                similarity_score.append(
                    f'{cosine_similarity(adv_feats[_], vic_feats[_]).detach().cpu().numpy()[0]:.3f}')
            adv_feats_all = torch.cat(adv_feats, dim=1)
            vic_feats_all = torch.cat(vic_feats, dim=1)
            loss = 1 - cosine_similarity(adv_feats_all, vic_feats_all) \
                   + torch.sum(torch.square(adv_feats_all - vic_feats_all)) \
                   + torch.sum(torch.abs(adv_feats_all - vic_feats_all)) \

            loss.backward(retain_graph=True)
            # 计算梯度
            grad = att_img_current.grad.data.clone()
            # 对梯度进行绝对值和平均
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            # 通过取模得到梯度最大值的坐标
            sum_grad = grad

            # 攻击方式 通过sign梯度乘以预制定的mask
            att_img_current.data = att_img_current.data - torch.sign(sum_grad) * \
                                   alpha * (1 - current_face_mask)

            # 利用clamp控制修改图像大小
            att_img_current.data = torch.clamp(att_img_current.data, -1.0, 1.0)
            att_img_current = att_img_current.data.requires_grad_(True)

            diff_ = att_img_current - att_img_
            diff = diff_.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
            diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
            diff_bgr = diff[:, :, ::-1]
            best_adv_img = im_a + diff_bgr

            if k % 1 == 0:
                print(f'图片序号：{n} '
                      f'训练与攻击次数：{k:04d} '
                      f'缩放倍率：{x_scale_item:.2f} '
                      f'模型相似度：{similarity_score} '
                      f'选择区域面积：{choose_area} '
                      f'得分：{1 - (choose_area / self.image_size ** 2)}')

            if time.time() - start_time > 80:
                return best_adv_img

        return best_adv_img
