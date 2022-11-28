# stdlib

# external
import torch
# from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import resnet18, ResNet18_Weights
from matplotlib import pyplot as plt

# local
import config
import rose_youtu_dataset



if __name__ == '__main__':
    print('hello world')

    ''' Model '''
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    preprocess = weights.transforms()

    ''' Dataset '''

    train_ds = rose_youtu_dataset.RoseYoutuLoader('train')

    ''' Training '''

    img, x, y = next(iter(train_ds))

    img_batch = preprocess(img)
    img_no_b = preprocess(img[0])

    model.eval()

    with torch.no_grad():
        out = model(img_batch)
        out_no_b = model(img_no_b)

    pred = out.squeeze(0).softmax(0)
    class_id = torch.argmax(pred).item()
    category_name = weights.meta["categories"][class_id]
    score = pred[class_id].item()

    # convert chw image to hwc
    img_hwc = img[0].permute(1, 2, 0)
    plt.imshow(img_hwc)
    plt.title(f'Prediction: {category_name}, score: {score:.2f}')
    plt.show()

