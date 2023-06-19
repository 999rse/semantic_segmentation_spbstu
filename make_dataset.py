from imports import *

def unzipping(input_dir, output_dir):
    '''
    Unzip train data and train_mask data
    '''
    
    if os.path.isfile(os.path.join(output_dir,"train")):
        print('Unzipping to data/processed/train')
        with zipfile.ZipFile(os.path.join(input_dir,"train.zip"), 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print('Skip unzipping train data')

    if os.path.isfile(os.path.join(output_dir,"train_masks")):
        print('Unzipping to data/processed/train_masks')
        with zipfile.ZipFile(os.path.join(input_dir,"train_masks.zip"), 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print('Skip unzipping train_mask data')


def splitting(output_dir):
    '''
    Split dataset in train and validation parts
    '''

    train_val_images = glob.glob(os.path.join(output_dir,"train/*"))
    train_val_images = sorted(train_val_images)
    train_val_images = [s.split("/")[-1].split(".")[0] for s in train_val_images]

    print(f'Length of train_val: {len(train_val_images)}')

    # Split to train and validation part
    train_images, val_images = train_test_split(train_val_images, test_size=0.1, random_state=4)
    print(f'Lenth train data: {len(train_images)}, Length val data: {len(val_images)}')
    print('--Splitting done!--')

    return train_images, val_images

img_size = 576

class CustomImageDataset(Dataset):
    '''
    This class is responsible for working with the image.
    '''
    
    def __init__(self, img_ids, transform=None):
        self.img_ids = img_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = "data/processed/train/" + img_id + ".jpg"
        mask_path = "data/processed/train_masks/" + img_id + "_mask.gif"
        image = Image.open(img_path)
        image = image.resize((img_size, img_size))
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((img_size, img_size))
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask