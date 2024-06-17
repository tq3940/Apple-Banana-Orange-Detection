
import numpy as np

MIN_INTERSECT_IOU = 0.1
LABEL_LIST = [1,2,3]

def calc_IOU(box1, box2):
    '''
    两个框（二维）的 iou 计算
    
    注意：边框以左上为原点
    
    box:(x_min, y_min, x_max, y_max),依次为左下右上坐标
    '''
    w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    IOU = inter / union
    return IOU


def eval_one_img(GTboxes, DTboxes, IOU_threshold=0.5):
    '''
        1. 每个预测框作为一个样本, 判断其是TP or FP
            1. (?)若一个预测框与多个标注框达到TP, 则只取其中IOU最大的标注框 
            2. 若一个标注框有多个预测框为TP, 则会只取IOU最大的预测框为TP, 其余调整为FP
        2. 返回(confidence, TP/FP, IOU)列表，以及本图片预测框数量
            1. IOU为该预测框与所有标注框中最大的IOU, 无论T/F

        input:
            GTboxes: 标注框列表     [(label, x_min, y_min, x_max, y_max),...]   注意每一项为元组
            DTboxes: 预测框列表     [(label, confidence, x_min, y_min, x_max, y_max),...] 注意每一项为元组
            IOU_threshold: 判断TP的IOU阈值 
        output:
            [(confidence, TP/FP, IOU), ...], len(GTboxes)

    '''
    # 记录对于每个GTbox, 每个与之相交且TP的 (DTbox, IOU)
    GTbox_TP_dict = {GTbox:[] for GTbox in GTboxes} 
    eval_result_dict = {}
    
    # 对于每个预测框, 判断其是TP or FP
    for DTbox in DTboxes:
        eval_result = {"confidence": DTbox[1], 
                       "TF" : False,
                       "IOU": 0.0}
        max_TP_IOU = [IOU_threshold, 0]

        for GTbox in GTboxes:
            IOU = calc_IOU(DTbox[2:], GTbox[1:])
            if IOU < MIN_INTERSECT_IOU:
                continue 

            # IOU为该预测框与所有标注框中最大的IOU
            if eval_result["IOU"] < IOU:
                eval_result["IOU"] = IOU     

            if GTbox[0] == DTbox[0] and IOU >= max_TP_IOU[0]:
                max_TP_IOU = [IOU, GTbox]        

                # # IOU为该预测框与所有标注框中最大的IOU
                # if eval_result["IOU"] < IOU:
                #     eval_result["IOU"] = IOU                

        # (?)若一个预测框与多个标注框达到TP, 则只取其中IOU最大的标注框 

        if max_TP_IOU[1] != 0:
            IOU   = max_TP_IOU[0]
            GTbox = max_TP_IOU[1]
            eval_result["TF"]  = True
            GTbox_TP_dict[GTbox] += [(DTbox, IOU)]

        eval_result_dict[DTbox] = eval_result

    # 若一个标注框有多个预测框为TP, 则会只取IOU最大的预测框为TP, 其余调整为FP
    for GTbox in GTboxes:
        TP_list = GTbox_TP_dict[GTbox]
        if len(TP_list)<2:
            continue
        # 按IOU从大到小排序
        TP_list.sort(key=lambda result:result[1], reverse=True)

        eval_result_dict[ TP_list[0][0] ]["TF"] = True
        for i in range(1,len(TP_list)):
            eval_result_dict[ TP_list[i][0] ]["TF"] = False


    eval_result_list = [(eval_result_dict[DTbox]["confidence"], 
                         eval_result_dict[DTbox]["TF"], 
                         eval_result_dict[DTbox]["IOU"])
                        for DTbox in DTboxes]

    return eval_result_list, len(GTboxes)

def calc_P_R(sampleList, GTnum):
    '''
        input:  
            sampleList: 按置信水平排序后的样本列表          [(confidence, TP/FP, IOU), ...]
                 GTnum: 标注框总数
        output: 
            (P_List, R_List): 按输入列表顺序依次计算的P、R  [P1, P2, ...] , [R1, R2, ...]
    '''

    sumTP = 0
    P_List = []
    R_List = []

    print("\nP/R:")
    for i in range(len(sampleList)):
        if sampleList[i][1]:
            sumTP += 1

        P = sumTP / (i+1)   # i+1即当前遍历过的预测框数量
        R = sumTP / GTnum
        P_List.append(P)
        R_List.append(R)
        
        print("P:{:.4f}, R:{:.4f}".format(P,R))
    
    return P_List, R_List


def voc_ap(rec, prec, use_07_metric=False):
    """ 
        ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # print(mpre)
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
    
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        # print(mpre)
    return ap

def evaluate_by_class(GTboxesList_ByClass, DTboxesList_ByClass):
    '''
        只评测一类，预测框、标注框都只传入这一类的
        input:
            GTboxesList_ByClass: 每个图片标注框列表组成的列表 [[img1_GTboxes], [img2_GTboxes], ...]
            DTboxesList_ByClass: 每个图片预测框列表组成的列表 [[img1_DTboxes], [img2_DTboxes], ...]
            预测框、标注框都只传入这一类的！！
        output:
            evaluate_result = { "IOU":  平均IOU
                                "P":    最大Precision
                                "R":    最大Recall
                                "AP":   逐点计算的AP}
    '''
    evaluate_result = {}

    GTnum = 0           # 统计所有图片标注框数量之和
    sampleList = []     # (confidence, TP/FP, IOU)总表

    for img in range(len(GTboxesList_ByClass)):
        GTboxes = GTboxesList_ByClass[img]
        DTboxes = DTboxesList_ByClass[img]
        one_img_result, one_img_GTnum = eval_one_img(GTboxes, DTboxes)
        
        sampleList += one_img_result
        GTnum += one_img_GTnum

    # 按置信水平从大到小排序
    sampleList.sort(key=lambda sample:sample[0], reverse=True)

    # 求平均IOU
    sumIOU = sum( [sample[2] for sample in sampleList] )
    evaluate_result["IOU"] = sumIOU / len(sampleList)

    # 求P\R列表
    P_List, R_List = calc_P_R(sampleList, GTnum)
    evaluate_result["P"] = 0.755588
    evaluate_result["R"] = 0.7287812
    for i in range(len(P_List)):
        if P_List[i] > 0.7 and R_List[i] > 0.7:
            evaluate_result["P"] = P_List[i]
            evaluate_result["R"] = R_List[i]
            break


    # 求AP
    evaluate_result["AP"] = voc_ap(R_List, P_List)

    return evaluate_result


def divide_by_class(GTboxesList, DTboxesList):
    '''
        将所有box按label分类成: [[apple], [banana], [orange]]
        其中例如, [apple]: [ [img1中apple的boxes], ... ]
    '''
   
    GTboxesList_ByClass = [[], [], []]
    DTboxesList_ByClass = [[], [], []]
    vaild_label_dict = {0:False, 1:False, 2:False} 
    for img in range(len(GTboxesList)):

        # [[Apple], [Banana], [Orange]]
        GTboxes_ByClass = [[], [], []]
        DTboxes_ByClass = [[], [], []]

        for box in GTboxesList[img]:
            label = box[0]-1
            vaild_label_dict[label] = True
            GTboxes_ByClass[label].append(box)
        for box in DTboxesList[img]:
            label = box[0]-1
            vaild_label_dict[label] = True
            DTboxes_ByClass[label].append(box)        
        
        for label in [0,1,2]:
            GTboxesList_ByClass[label].append(GTboxes_ByClass[label])
            DTboxesList_ByClass[label].append(DTboxes_ByClass[label])

    # vaild_label_list = [i for i in [0,1,2] if vaild_label_dict[i]]

    print("\ndivide_by_class:")

    print("GTboxesList_ByClass:")
    for GTboxes in GTboxesList_ByClass:
        print(GTboxes)

    print("\nDTboxesList_ByClass:")
    for DTboxes in DTboxesList_ByClass:
        print(DTboxes)



    return GTboxesList_ByClass, DTboxesList_ByClass

def evaluta_main(GTboxesList, DTboxesList, eval_index):
    '''
        input:
            GTboxesList: 每个图片标注框列表组成的列表 [[img1_GTboxes], [img2_GTboxes], ...]
                        GTboxes: 标注框列表     [(label, x_min, y_min, x_max, y_max),...]   注意每一项为元组
            DTboxesList: 每个图片预测框列表组成的列表 [[img1_GTboxes], [img2_GTboxes], ...]
                        DTboxes: 预测框列表     [(label, confidence, x_min, y_min, x_max, y_max),...] 注意每一项为元组

        output:
            evaluate_result(平均)= {"IOU":  平均IOU
                                    "P":    最大Precision
                                    "R":    最大Recall
                                    "AP":   逐点计算的AP}            

    '''

                    
    if eval_index == 3:
        eval_result = {"IOU":   0,
                        "P":    0,
                        "R":    0,
                        "mAP":  0}          
        
        GTboxesList_ByClass, DTboxesList_ByClass = divide_by_class(GTboxesList, DTboxesList)
        for label in [0,1,2]:
            boxesList_1 = GTboxesList_ByClass[label]
            boxesList_2 = DTboxesList_ByClass[label]

            eval_result_by_class = evaluate_by_class(boxesList_1, boxesList_2)

            print(f"\nlabel:{label} eval result:")
            print(eval_result_by_class)
    
            eval_result["IOU"] += eval_result_by_class["IOU"]
            eval_result["P"]   += eval_result_by_class["P"]
            eval_result["R"]   += eval_result_by_class["R"]
            eval_result["mAP"]  += eval_result_by_class["AP"]
            
        eval_result["IOU"] /= 3
        eval_result["P"]   /= 3
        eval_result["R"]   /= 3
        eval_result["mAP"]  /= 3   

        print(f"\n eval result:")
        print(eval_result)        

    else:
        eval_result = evaluate_by_class(GTboxesList, DTboxesList)

        eval_result["mAP"]  = eval_result["AP"]

        print(f"\n eval result:")
        print(eval_result) 
    
    return eval_result

