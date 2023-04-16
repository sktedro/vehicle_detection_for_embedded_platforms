import torch
import torchvision.transforms as transforms
from mmdet.apis import init_detector
from torch.utils.data import DataLoader
from torchvision.datasets import coco
from tqdm import tqdm

from mmyolo.utils import register_all_modules
register_all_modules()


transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((288, 512)), # (h, w)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def calibrate(model, data_loader):
    counter = 0
    with torch.no_grad():
        for image, _ in tqdm(data_loader):
            counter += 1
            model(image)
            if counter == 10:
                return

if __name__ == "__main__":
    # Load the
    model = init_detector("working_dir_yolov8_p_conf8_512x288_lr01/yolov8_p_512x288.py", "working_dir_yolov8_p_conf8_512x288_lr01/epoch_500.pth", device="cpu")
    model = torch.quantization.QuantWrapper(model)
    model = model.to("cpu") # To be sure

    # Set model to evaluation mode
    model.eval()

    # Set backend and quantization configuration
    backend = "fbgemm"
    # backend = "qnnpack"
    # backend = "x86"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    # TODO Fuse modules

    print("Preparing model")
    model = torch.quantization.prepare(model, inplace=False)

    print("Calibrating")
    dataset = coco.CocoDetection("/home/tedro/Downloads/datasets/", "/home/tedro/Downloads/datasets/val.json", transform) # TODO train subset?
    batch_size = 1 # With 32, collate says RuntimeError: each element in list of batch should be of equal size
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8) # TODO shuffle=True
    calibrate(model, data_loader)

    print("Converting")
    model = torch.quantization.convert(model, inplace=False)

    calibrate(model, data_loader)

    # Save the quantized model
    torch.save(model.state_dict(), "your_model_quantized.pth")

    print("All done")