import argparse
from PIL import Image
import torch
from torchvision import transforms
import os
import shutil
import numpy as np
from torch.optim import Adam
import random 
from torch.nn import ReLU
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn import BatchNorm2d
from torch.nn import Parameter
from torch.nn import Linear
from torch.nn import DataParallel
from torch.nn import Sequential
from torch.nn import ConvTranspose2d
from torch.nn import GRUCell
from torch.nn import Module
from torch.nn import Flatten
from torch.nn.init import normal_
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
from torch.nn import Conv2d
from torch.nn import Conv1d
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from tqdm import tqdm
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)
import torch.nn as nn
import einops
from einops import rearrange


class Block(Module):
    def __init__(self, channels):
        super(Block, self).__init__()
        self.downsample = Sequential(OrderedDict([
            ('conv1' , Conv2d(channels, channels, kernel_size = 5 , stride = 1 , padding = 2)),
            ('bn1'   , BatchNorm2d(channels)),
            ('relu1' , ReLU()),
            ('conv2' , Conv2d(channels, channels, kernel_size = 5 , stride = 1 , padding = 2)),
            ('bn2'   , BatchNorm2d(channels))
        ]))
        
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(x + self.downsample(x))

class BlockUp(Module):
    def __init__(self , in_channels , out_channels):
        super(BlockUp, self).__init__()
        
        self.downsample = Sequential(OrderedDict([
            ('conv1' , Conv2d(in_channels , out_channels , kernel_size = 5 , stride = 1 , padding = 2)),
            ('bn1'   , BatchNorm2d(out_channels)),
            ('relu1' , ReLU()),
            ('conv2' , Conv2d(out_channels , out_channels , kernel_size = 5 , stride = 1 , padding = 2)),
            ('bn2'   , BatchNorm2d(out_channels)),
        ]))
        self.skip = Sequential(OrderedDict([
            ('conv1' , Conv2d(in_channels , out_channels, kernel_size = 5 , stride = 1 , padding = 2)),
            ('bn1'   , BatchNorm2d(out_channels))
        ]))
        self.relu = ReLU()



    def forward(self, x):
        return self.relu(self.skip(x) + self.downsample(x))
    
    
class ResNet18(Module):
    def __init__(self):

        super(ResNet18, self).__init__()
        layer0 = Sequential(OrderedDict([
         ('conv' , Conv2d(3 , 16 , kernel_size = 5 , stride = 1 , padding = 2)),
         ('bn'   , BatchNorm2d(16)),
         ('relu' , ReLU())
        ]))

        self.resnet = Sequential(OrderedDict([
            ('layer0' , layer0),
            ('block1' , Block(16)),
            ('block2' , Block(16)),
            ('block3' , BlockUp(16 , 32)),
            ('block4' , Block(32)),
            ('block5' , Block(32)),
            ('block6' , BlockUp(32 , 64)),
            ('block7' , Block(64)),
            ('block8' , Block(64))
        ]))


    def forward(self, x):
        return self.resnet(x)
    
class ObjectDiscovery(Module):
    def __init__(self , encoder_resolution ,  decoder_resolution , T   , K , D_slots):
        super(ObjectDiscovery , self).__init__()
        print(f'Initialized ObjectDiscovery!')
        self.layers = Sequential(OrderedDict([
          ("encoder" , ImageEncoder(resolution = encoder_resolution , T = T,
                                    K = K , D_slots = D_slots )),
          ("decoder" , SlotAttentionDecoder(resolution = decoder_resolution , K = K , D_slots = D_slots))
        ]))
    def forward(self , image):
        return self.layers(image)

class PositionEncoder(Module):
    def __init__(self, output_dim , resolution):
        super(PositionEncoder , self).__init__()
        self.linear =  Linear(in_features = 4 , out_features = output_dim)
        # above is equivalent to a linear layer
        self.grid = Parameter(data = PositionEncoder.build_grid(resolution) , requires_grad = False)
        print("Grid shape:" , self.grid.shape)

    @staticmethod
    def build_grid(resolution):
        ranges = [np.linspace(start = 0.0 , stop = 1.0 , num = dimension) for dimension in resolution] # dim = (2 , 128)
        grid = np.meshgrid(*ranges , sparse = False, indexing = "ij") # dim = (128 , 128)
        # row[i] of grid[0] has all elements i / 127 and col[j] of grid[1] has all elements j / 127
        grid = np.stack(grid , axis = -1) # dim = (64 , 64 , 2) to match conv later
        grid = np.expand_dims(grid, axis = 0) # dim = (1 , 64 , 64 , 2) for batch dimension
        grid = grid.astype(np.float32) # PyTorch throws an error later otherwise
        return torch.tensor(np.concatenate([grid , 1.0 - grid] , axis = 3)) # (1 , 64 , 64 , 4)


    def forward(self, x): # x has shape (batch , 64 , 64 , D_inputs)
        return x + self.linear(self.grid)


class ImageEncoder(Module):
    def __init__(self , resolution , T  ,  K , D_slots):
        super(ImageEncoder , self).__init__()
        print(f'Initialized ImageEncoder! resolution: {resolution}')
        D_inputs = 64
        self.encoder_cnn = ResNet18()
        down_resolution = (128 , 128)
        positional_encoder = PositionEncoder(output_dim =  D_inputs , resolution = down_resolution)


        slot_attention = SlotAttention(T = T , K = K , D_slots = D_slots)
        self.pos_encode_feedforward_slotattn = Sequential(OrderedDict([
            ("pos_encoder" , positional_encoder), #  (batch , 64 , 64 , D_inputs)
            ("flatten" , Flatten(start_dim = 1 , end_dim = 2)), # (batch , 64 * 64 , D_inputs)
            ("layer_norm" ,  LayerNorm(normalized_shape = 64)), # (batch , 64 * 64 , D_inputs)
            ("linear1:" , Linear(in_features = 64 , out_features = 128)),
            ("relu1:"   , ReLU()),
            ("linear2:" , Linear(in_features = 128 , out_features = 128)),
            ("slot_attention" , slot_attention) #  (batch , K , D_slots)
        ]))
    # feature map has shape (channels , h , w)
    # N is h * w i.e. each pixel is a different feature "vector", size of this vector of D_inputs = # of channels
    # channels = D_inputs , h * w = N
    def forward(self , x): # x is aimge with shape (batch , 3 , 126 , 128)
        x = self.encoder_cnn(x).permute(dims = (0 , 2 , 3 , 1))  # (batch , 64 , 64 , D_inputs)
        return self.pos_encode_feedforward_slotattn(x)



class SlotAttention(Module):
    def __init__(self , T, K , D_slots  , epsilon = 1e-8):
        super(SlotAttention , self).__init__()
        print(f'Initialzing SlotAttention parameters: T: {T} , K: {K} , D_slots: {D_slots}')
        self.T = T
        self.K = K
        self.D_slots = D_slots
        self.epsilon = epsilon

        self.norm_inputs = LayerNorm(normalized_shape =  128)
        self.query_from_slots  = Sequential(OrderedDict([
            ("NormSlots" , LayerNorm(normalized_shape = D_slots)),
            ("SlotQuery" , Linear(in_features = D_slots  , out_features = D_slots , bias = False))
        ]))


#         self.slots_init = Parameter(data = normal_(torch.empty(1 , K , D_slots)) , requires_grad = True)

        self.init_latents = Parameter(normal_(torch.empty(1 , self.K , self.D_slots)))



        self.keys_from_inputs  = Linear(in_features = 128 , out_features = D_slots , bias = False)
        self.vals_from_inputs  = Linear(in_features = 128 , out_features = D_slots , bias = False)
        self.gru = GRUCell(input_size = D_slots , hidden_size = D_slots)
        self.norm_feed = Sequential(OrderedDict([
            ("layer_norm" ,  LayerNorm(normalized_shape = D_slots)),
            ("linear1" , Linear(in_features = D_slots , out_features = 256)),
            ("relu1"   , ReLU()),
            ("linear2" , Linear(in_features = 256 , out_features = D_slots))
        ]))


    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = self.norm_inputs(inputs)
        keys = self.keys_from_inputs(inputs)
        vals = self.vals_from_inputs(inputs)
        slots = self.init_latents.repeat(batch_size , 1 , 1)
        for _ in range(self.T):
            slots_prev = slots
            queries = self.query_from_slots(slots)
            attn_logits = (self.D_slots ** -0.5) * torch.einsum('bnc,bmc->bnm', keys , queries)
            attn = F.softmax(attn_logits , dim = -1)
            attn = attn + self.epsilon
            attn2 = attn
            attn = attn / torch.sum(attn, dim = 1, keepdim = True)
            updates = torch.einsum('bnm,bnc->bmc' , attn , vals)
            slots = self.gru(
                updates.view(batch_size * self.K , self.D_slots),
                slots_prev.view(batch_size * self.K , self.D_slots),
            )
            slots = slots.view(batch_size , self.K , self.D_slots)
            slots = slots + self.norm_feed(slots)

        return slots , attn2


class SlotAttentionDecoder(Module):
    def __init__(self , resolution , K , D_slots):
        super(SlotAttentionDecoder , self).__init__()
        print(f'Initialized SlotAttentionDecoder! resolution: {resolution}')
        print(f'Decoder parameters: K: {K} , D_slots: {D_slots}')
        self.resolution = resolution
        self.positional_encoder = PositionEncoder(output_dim = D_slots , resolution = resolution)

        self.K = K
        self.D_slots = D_slots

        self.decoder_cnn = Sequential(OrderedDict([
            ("conv1" , ConvTranspose2d(in_channels = D_slots , out_channels = 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1)),
            ('relu1' , ReLU()),
            ("conv2" , ConvTranspose2d(in_channels = 64 , out_channels = 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1)),
            ('relu2' , ReLU()),
            ("conv3" , ConvTranspose2d(in_channels = 64 , out_channels = 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1)),
            ('relu3' , ReLU()),
            ("conv4" , ConvTranspose2d(in_channels = 64 , out_channels = 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1)),
            ('relu4' , ReLU()),
            ("conv5" , ConvTranspose2d(in_channels = 64 , out_channels = 4 , kernel_size = 1 , stride = 1)),
        ]))


    def forward(self , slots):
        slots = slots[0]
        # slots comes form the ImageEncoder shape : (batch , K , D_slots)
        batch_size = slots.shape[0]
        x =  slots.view(batch_size * self.K , 1 , 1 , self.D_slots) # (batch * K , 1 , 1 , D_slots)
        x = x.repeat(1 , self.resolution[0] , self.resolution[1] , 1) # (batch * K , 8 , 8 , D_slots)
        x = self.positional_encoder(x).permute(0 , 3 , 1 , 2) # (batch * K , 8 , 8 , D_slots)
        x = self.decoder_cnn(x) # dim = (batch * K , 4 , 128 , 128)
        x = x.view(batch_size, self.K , 4 , 128 , 128)# dim = (batch , K , 4 , 128 , 128)
        recons , masks = x[: , : , : 3 , : , :]  , x[: , : , -1 : , : , :]
        masks = F.softmax(masks , dim = 1) # dim = (batch , K , 1 , 128, 128)
        reconstructed = torch.sum(recons * masks , dim = 1) # dim = (batch , 3 , 128, 128)
        return reconstructed , masks

#############################################################################################################




def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for COL775 Assignment 2")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where results will be stored")
    return parser.parse_args()


def make_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
        print(f"directory '{dir_name}' and its contents deleted!")

    print(f"directory '{dir_name}' created!")
    os.mkdir(dir_name)


def save_image(image , file):
    pil = Image.fromarray(image)
    pil.save(file)

def save_empty_masks(index , output_dir):
    for slot in range(7 , 11):
        null_mask = np.zeros(shape = (128 , 128) , dtype = np.uint8)
        save_image(null_mask , file = f'{output_dir}/{index}_{slot}.png')

def build_output_part1(input_dir , transform , model , output_dir):
    num_images = os.listdir(input_dir)
    with torch.no_grad():
        for idx in range(1 , len(num_images) + 1):
            image_path = f'{input_dir}/{idx}.png'
            image = transform(Image.open(image_path).convert('RGB')).to(device)
            reconstructed , masks = model(image.unsqueeze(0))
            reconstructed , masks = reconstructed.squeeze(0) , masks.squeeze(0).squeeze(1)
            reconstructed  = (255 * (0.5 * reconstructed.cpu().numpy().transpose(1, 2, 0) + 0.5)).astype(np.uint8)
            for slot in range(7):
                slot_mask = masks[slot]
                slot_mask  = (255 * (slot_mask.cpu().numpy())).astype(np.uint8)
                save_image(slot_mask , f'{output_dir}/{idx}_{slot}.png')
            save_empty_masks(idx , output_dir = output_dir)
            save_image(reconstructed , f'{output_dir}/{idx}.png')


# python3 infer.py --input_dir input_images --model_path model.pth --output_dir output

if __name__ == "__main__":
    args = parse_args()
    print("torch version:" , torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:" , device)
    model = None
    if torch.cuda.is_available():
        model = torch.load(args.model_path)
        
    else:
        model = torch.load(args.model_path , map_location = 'cpu')
    model.to(device)
    model.eval()
    make_directory(args.output_dir)

    image_transform = transforms.Compose([
        transforms.CenterCrop(size = (192 , 192)),
        transforms.Resize(size = (128 , 128) ,  interpolation = InterpolationMode.BILINEAR),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.5 ,  0.5 , 0.5) , (0.5, 0.5 , 0.5)),
    ])


    print("Now inferring....")
    build_output_part1(input_dir = args.input_dir , transform = image_transform , model = model,
                output_dir = args.output_dir)


    
