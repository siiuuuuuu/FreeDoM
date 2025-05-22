import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

from .clip.base_clip import CLIPEncoder
from .face_parsing.model import FaceParseTool
from .anime2sketch.model import FaceSketchTool 
from .landmark.model import FaceLandMarkTool
from .arcface.model import IDLoss


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def clip_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, prompt=None, stop=100, domain="face"):
    clip_encoder = CLIPEncoder().cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if domain == "face":
            repeat = 1
        elif domain == "imagenet":
            if 800 >= i >= 500:
                repeat = 10
            else:
                repeat = 1
        
        for idx in range(repeat):
        
            xt.requires_grad = True
            
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # get guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * 0.02
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            xt_next -= rho * norm_grad
            
            x0_t = x0_t.detach()
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]

#主要
def parse_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    """
    使用 DDIM 算法进行图像去噪，同时结合面部解析引导。

    参数:
    x (torch.Tensor): 输入的噪声图像。
    seq (list): 时间步序列。
    model (torch.nn.Module): 去噪模型。
    b (torch.Tensor): 扩散过程中的 beta 值。
    cls_fn (callable, optional): 分类器函数。默认为 None。
    rho_scale (float, optional): 引导梯度的缩放因子。默认为 1.0。
    stop (int, optional): 停止引导的时间步。默认为 100。
    ref_path (str, optional): 参考图像的路径。默认为 None。

    返回:
    list: 最后一个时间步的图像。
    list: 最后一个时间步预测的原始图像。
    """
    # 初始化面部解析工具，并将其移动到 CUDA 设备上
    parser = FaceParseTool(ref_path=ref_path).cuda()

    # 设置迭代变量
    # 获取输入图像的批量大小
    n = x.size(0)
    # 生成下一个时间步的序列
    seq_next = [-1] + list(seq[:-1])
    # 用于存储每个时间步预测的原始图像
    x0_preds = []
    # 用于存储每个时间步的图像
    xs = [x]

    # 遍历时间步序列
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        # 创建当前时间步的张量
        t = (torch.ones(n) * i).to(x.device)
        # 创建下一个时间步的张量
        next_t = (torch.ones(n) * j).to(x.device)
        # 计算当前时间步的 alpha 值
        at = compute_alpha(b, t.long())
        # 计算下一个时间步的 alpha 值
        at_next = compute_alpha(b, next_t.long())
        # 获取当前时间步的图像，并移动到 CUDA 设备上
        xt = xs[-1].to('cuda')
        
        # 开启梯度计算
        xt.requires_grad = True
        
        # 根据是否提供分类器函数来调用模型
        if cls_fn is None:
            # 不使用分类器，直接调用模型
            et = model(xt, t)
        else:
            # 打印使用分类器的信息
            print("use class_num")
            # 指定类别编号
            class_num = 281
            # 创建类别张量
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
            # 调用模型并传入类别信息
            et = model(xt, t, classes)
            # 截取前三个通道
            et = et[:, :3]
            # 减去分类器的输出
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        # 如果模型输出的通道数为 6，则截取前三个通道
        if et.size(1) == 6:
            et = et[:, :3]

        # 估计当前时间步的原始图像
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        # 计算面部解析的残差
        residual = parser.get_residual(x0_t)
        # 计算残差的范数
        norm = torch.linalg.norm(residual)
        # 计算范数关于当前图像的梯度
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
        
        # 设置 eta 参数
        eta = 0.5
        # 计算系数 c1
        c1 = (1 - at_next).sqrt() * eta
        # 计算系数 c2
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        # 计算下一个时间步的图像
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # 计算引导梯度的缩放因子
        rho = at.sqrt() * rho_scale
        # 如果当前时间步大于停止时间步，则应用引导梯度
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        # 分离张量，避免梯度传播
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        # 将预测的原始图像添加到列表中
        x0_preds.append(x0_t.to('cpu'))
        # 将下一个时间步的图像添加到列表中
        xs.append(xt_next.to('cpu'))

    # 返回最后一个时间步的图像和预测的原始图像
    return [xs[-1]], [x0_preds[-1]]


def sketch_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2sketch = FaceSketchTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            # print("use class_num")
            class_num = 7
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2sketch.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def landmark_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2landmark = FaceLandMarkTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2landmark.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
        
        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def arcface_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    idloss = IDLoss(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = idloss.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
        
        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

