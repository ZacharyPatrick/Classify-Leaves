import albumentations
from albumentations.pytorch.transforms import ToTensorV2

transform_train = albumentations.Compose(
    [
        albumentations.Resize(320, 320),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Rotate(limit=180, p=0.7),
        albumentations.RandomBrightnessContrast(),
        albumentations.ShiftScaleRotate(
            shift_limit=0.25, scale_limit=0.1, rotate_limit=0
        ),
        albumentations.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            max_pixel_value=255.0, always_apply=True
        ),
        ToTensorV2(p=1.0),
    ]
)

transform_test = albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)
        ]
    )
