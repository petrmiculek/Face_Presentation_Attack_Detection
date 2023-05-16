"""
Source: https://raw.githubusercontent.com/pytorch/vision/main/references/classification/presets.py
"""
import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:
    def __init__(
            self,
            *,
            crop_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation=InterpolationMode.BILINEAR,
            hflip_prob=0.5,
            auto_augment_policy=None,
            ra_magnitude=9,
            augmix_severity=3,
            random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
            self,
            *,
            crop_size,
            resize_size=256,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation=InterpolationMode.BILINEAR,
    ):
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ''' Exploring augmentations '''

    train_loader = ...
    model, preprocess = ..., ...
    batch_size = 1

    aug_train_kwargs = {'crop_size': 224, }
    t_train = ClassificationPresetTrain(**aug_train_kwargs)

    t_eval = ClassificationPresetEval(**aug_train_kwargs)

    # call for each image in batch separately or for the whole batch?

    for sample in train_loader:
        img_pre = preprocess(sample['image'])
        img_aug = t_train(sample['image'])
        img_aug_pre = preprocess(img_aug)

        for i in range(batch_size):
            # 1x4 sublot
            fig, axs = plt.subplots(1, 4, figsize=(20, 20))
            plt.axis('off')
            plt.title(f'Label: {sample["label"][i]}, idx: {sample["idx"][i]}')

            plt.subplot(1, 4, 1)
            plt.title('Orig')
            plt.imshow(sample['image'][i].permute(1, 2, 0))
            plt.subplot(1, 4, 2)
            plt.title('Pre')
            plt.imshow(img_pre[i].permute(1, 2, 0))
            plt.subplot(1, 4, 3)
            plt.title('Aug')
            plt.imshow(img_aug[i].permute(1, 2, 0))
            plt.subplot(1, 4, 4)
            plt.title('Aug + Pre')
            plt.imshow(img_aug_pre[i].permute(1, 2, 0))
            plt.tight_layout()
            plt.show()
