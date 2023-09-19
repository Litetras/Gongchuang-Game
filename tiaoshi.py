import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    #class_codes = {'bottles': 0}#________________________________________________________________CJQ____add#############
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  #默认为保存图片，不保存图片可以在执行程序中输入 --nosave
    webcam = source.isnumeric()#摄像头检测的一个判断，是你在–source后的输入，当检测对象是本地的一个摄像头判定你的检测对象是摄像头检测。

    # Directories用于保存检测结果的文件夹
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  #通过降低少许检测精度提高检测速度的判断 half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model 模型的载入，其中attempt_load()是对权重文件的一个载入，主要是通过torch.load()这个函数，值得注意的是，这个可以是多个模型的混和载入（需要注意每个模型的输入和输入大小），也可以是对一个完整检测模型的载入。
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier 判断是否进行第二步检测分类，通过resnet101对通过yolov5识别检测后的结果在进行一次预测
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader用于读取图片或视频
    vid_path, vid_writer = None, None
    if webcam:#if里是如果检测目标是摄像头或者网络视频源，check_imshow()检测运行环境是否是在docker下（docker环境下不能打开界面显示，会报错），然后是视频流或者摄像头的一个载入。如果不是视频流或者摄像头，载入检测对象为图片、视频或者文件夹内的文件，通过迭代的方式一个一个的进行检测。
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors 载入模型内部的class names，以及当前检测每一个name对应的矩形框的颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference 开始执行检测推理部分的代码
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()#t0是为了计算整个推理过程的时间的开始计时，即完成文件夹内图像的推理时间
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()#t1是为了计算每一张图像的推理时间的开始计时，即完成一张图像的推理时间
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier 是否对预测结果进行二次预测分类
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections 预测结果的处理
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            #s += '%gx%g ' % img.shape[2:]  # print string____________________________________________________________________________zyp   delete########
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):#如果det中有相应的检测目标，将det[:, 0:4]，即在img（原图（im0s）resize到targetSize后的图像）上xyxy对于的像数值坐标还原到原图（im0s）的像数值坐标，为了后续方便在原图上画出目标物的矩形框。然后是确定det[:, -1]中的所存在同类的classID一共有多少个，并有几个类别，是在后续输出检测完成后打印检测结果的相关信息。
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}"  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file检测结果写入，如果在程序执行时输入了 --save-txt,则将类别 xywh 置信度 写入保存在save_dir / ‘labels’/图片名中，如果去掉置信度和前后的空格就是通过labelimg标注后的yolo标签。
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')________________________________________________________________zyp delete#############
            print(f'{s}')##_____________________________________________________________________________________zyp add###############

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)#______________________________________________________________________zyp 注释掉这行就能取消实时显示图像#############
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video如果检测对象是本地视频则通过cv2获取fps，w，h，检测对象是网络视频默认fps=30。然后通过cv2.VideoWriter(args).write（im0）写入每一帧检测结果，保存视频。
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:#打印最总的检测结果。
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")#____________________________________________________________________________zyp delete############

    #print(f'Done. ({time.time() - t0:.3f}s)')#____________________________________________________________________________zyp delete############


if __name__ == '__main__':
    #设置目标检测的配置参数：
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')#权重文件手动改
    parser.add_argument('--source', type=str, default='0', help='source')  # default='data/images'代表直接检测文件夹里的图片, default改为0则默认用摄像头
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')#尺寸是640*640
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))#执行检测程序前，查看requirements.txt里的相关安装包是否安装在执行检测的python环境下
    #执行检测部分：
    with torch.no_grad():
        detect()
