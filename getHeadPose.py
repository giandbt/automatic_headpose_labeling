import os
import json
import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import shutil

from sys import path

path.insert(0, os.path.join(os.path.dirname(__file__), 'HopeNet', 'code'))
import datasets, hopenet, utils

path.insert(0, os.path.join(os.path.dirname(__file__), 'FSA', 'demo'))
from FSANET_model import *


def run_head_pose_hopenet(input_folder, save_dir):
    '''
    Run the head pose on the images in the input folder.
    :param: input_folder
            string for the root folder with the images we want to label
    :param: faces_dir
            string for the root folder with the faces derived from original images
    :param: save_dir
            string for the root folder to save all the output images (original images + head pose axis)
    :return:
    '''

    # ensures we have a fresh folder
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    gpu_id = 0
    cudnn.enabled = True

    snapshot_path = os.path.join(os.path.dirname(__file__), 'deep-head-pose', 'models', 'hopenet_alpha2.pkl')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.Scale(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu_id)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu_id)

    images = [os.path.join(input_folder, image) for image in os.listdir(input_folder) if 'jpg' or 'png' in image]

    head_pose_results = {}

    for image_counter, image in enumerate(images):
        print('Detecting Hopenet Headpose for Image %i out of %i' % (image_counter+1, len(images)))

        img_id = os.path.basename(image)[:-4]

        head_pose_results[img_id] = {}

        img = Image.open(image)

        # Transform
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(gpu_id)

        yaw, pitch, roll = model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)

        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        # convert to floats
        yaw_predicted = float(yaw_predicted.cpu().numpy())
        pitch_predicted = float(pitch_predicted.cpu().numpy())
        roll_predicted = float(roll_predicted.cpu().numpy())

        head_pose_results[img_id] = {'yaw': yaw_predicted, 'pitch': pitch_predicted,
                                         'roll': roll_predicted}
    # save result out to json
    r = json.dumps(head_pose_results, indent=4)
    with open(os.path.join(save_dir, 'head_pose_prediction_hopenet.json'), 'w') as f:
        f.write(r)

def run_head_pose_FSA(input_folder, save_dir):
    annotations = os.path.join(save_dir, 'head_pose_prediction_hopenet.json')
    padding_perc = 0.4
    
    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())
    
    images = [os.path.join(input_folder, image) for image in os.listdir(input_folder) if 'jpg' or 'png' in image]
    avail_imgs = annon_dict.keys()
    
    # load model and weights
    # Parameters
    img_size = 64
    
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7 * 3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
    
    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    num_primcaps = 8 * 8 * 3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
    
    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    print('Loading models ...')
    
    weight_file1 = './FSA-Net/pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')
    
    weight_file2 = './FSA-Net/pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')
    
    weight_file3 = './FSA-Net/pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')
    
    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)
    
    for idx, image in enumerate(images):
        print('Detecting FSA Headpose for Image:', idx+1, 'out of', len(images))
        img = Image.open(image)
        
        # gets images Id
        img_id = os.path.basename(image)[:-4]
        
        # ensures the image is in the dictionary key
        if not img_id in avail_imgs:
            continue
            
        img = img.resize((img_size, img_size))
        face = np.expand_dims(np.array(img), axis=0)
        p_result = model.predict(face)
        
        yaw = p_result[0][0]
        pitch = p_result[0][1]
        roll = p_result[0][2]
        
        annon_dict[img_id]['roll'] = float(roll)
        annon_dict[img_id]['pitch'] = float(pitch)
        annon_dict[img_id]['yaw'] = float(yaw)
            
    # save result out to json
    r = json.dumps(annon_dict, indent=4)
    
    with open(os.path.join(save_dir, 'head_pose_prediction_FSA.json'), 'w') as f:
        f.write(r)
        
def essemble_head_pose(save_dir):

    annotations_hopenet = os.path.join(save_dir, 'head_pose_prediction_hopenet.json')
    
    with open(annotations_hopenet, 'r') as f:
        hopenet_dict = json.loads(f.read())
    
    annotations_FSA = os.path.join(save_dir, 'head_pose_prediction_FSA.json')
    
    with open(annotations_FSA, 'r') as f:
        FSA_dict = json.loads(f.read())
    
    for img_id in FSA_dict.keys():
        average_yaw = (hopenet_dict[img_id]['yaw'] +
                       FSA_dict[img_id]['yaw']) / 2
        average_pitch = (hopenet_dict[img_id]['pitch'] +
                         FSA_dict[img_id]['pitch']) / 2
        average_roll = (hopenet_dict[img_id]['roll'] +
                        FSA_dict[img_id]['roll']) / 2
        
        FSA_dict[img_id]['yaw'] = average_yaw
        FSA_dict[img_id]['pitch'] = average_pitch
        FSA_dict[img_id]['roll'] = average_roll
    
    # save result out to json
    r = json.dumps(FSA_dict, indent=4)
    
    with open(os.path.join(save_dir, 'head_pose_prediction_ensemble.json'), 'w') as f:
        f.write(r)
        
def get_head_pose_images(input_folder, save_dir):
    annotations = os.path.join(save_dir, 'head_pose_prediction_ensemble.json')
    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())

    images = [os.path.join(input_folder, image) for image in os.listdir(input_folder) if 'jpg' in image]
    for image_counter, image in enumerate(images):
        print('Saving Headpose for Image %i out of %i' % (image_counter + 1, len(images)))
        img_id = os.path.basename(image)[:-4]
        cv2_frame = cv2.imread(image)
        h = cv2_frame.shape[0]
        w = cv2_frame.shape[1]
        
        cv2_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        
        yaw_predicted = annon_dict[img_id]['yaw']
        pitch_predicted = annon_dict[img_id]['pitch']
        roll_predicted = annon_dict[img_id]['roll']

        # create image of result
        cv2_frame = utils.draw_axis(cv2_frame, yaw_predicted, pitch_predicted, roll_predicted,
                                    tdx=(w) / 2, tdy=(h) / 2)

        Image.fromarray(cv2_frame).save('%s/%s_%s' % (save_dir, img_id, 'head_pose_prediction.jpg'))
        
def get_head_pose(input_folder, save_dir):
    run_head_pose_hopenet(input_folder, save_dir)
    run_head_pose_FSA(input_folder, save_dir)
    essemble_head_pose(save_dir)
    #get_head_pose_images(input_folder, save_dir)
    
    
if __name__ == '__main__':
    image_folder = '/home/giancarlo/Documents/HeadPose-test/data_labeling/test'
    head_pose_dir = '/home/giancarlo/Documents/HeadPose-test/results_labeling/test'
    get_head_pose(image_folder,head_pose_dir)
