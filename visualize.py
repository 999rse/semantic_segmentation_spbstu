from imports import *

def plot_images(images, mask = False):
    '''
    This function plotting images.
    
    Input:
        images - Pack of images

    Return: None
    '''
    fig = plt.figure(figsize=(10, 8))
    columns = len(images)
    rows = 1
    for i in range(columns*rows):
        img = images[i]
        fig.add_subplot(rows, columns, i+1)
        if mask:
            img = img > 0.5
        plt.imshow(img, cmap="gray")
    plt.show()

def visualize_result(dataloader, model, device):
    '''
    Visualizate results.
    
    Input:
        dataloader - Data [train, val]
        
    Return: None
    '''

    img_size = 576

    tr_imgs = []
    msk_imgs = []
    
    for x, y in dataloader:
        x = x.to(device)
        
        pred = model(x)
        
        for i in range(x.shape[0]):
            tr_img = x[i].cpu().detach().numpy()
            print(tr_img.shape)
            tr_img = np.einsum('kij->ijk',tr_img)
            tr_imgs.append(tr_img)
            
            msk_img = pred[i].cpu().detach().numpy()
            print(msk_img.shape)
            msk_img = msk_img.reshape((img_size, img_size))
            msk_imgs.append(msk_img)

        break
        
    plot_images(tr_imgs, mask=False)
    plot_images(msk_imgs, mask=True)
    