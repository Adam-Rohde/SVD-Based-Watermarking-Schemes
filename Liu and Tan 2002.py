'''
Liu, R., & Tan, T. (2002). An svd-based watermarking scheme for protecting
rightful ownership.IEEE Transactionson Multimedia,4(1), 121-128.
'''

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import numpy.linalg as la


###############################
###############################
###############################
## Functions


###############################
## create square images and convert to matrix
def MakeSquareMatrix(img):
    width, height = img.size
    if width == height:
        if len(np.array(img).shape)==3:
            return np.array(img)[:,:,0]
        else:
            return np.array(img)
    elif width > height:
        result = Image.new(img.mode, (width, width))
        result.paste(img, (0, (width - height) // 2))
        if len(np.array(result).shape)==3:
            return np.array(result)[:,:,0]
        else:
            return np.array(result)
    else:
        result = Image.new(img.mode, (height, height))
        result.paste(img, ((height - width) // 2, 0))
        if len(np.array(result).shape)==3:
            return np.array(result)[:,:,0]
        else:
            return np.array(result)

###############################
## Embed Watermark

def EmbedWatermark(A,W,a):
    #[1] Decompose A into U S Vh
    U, S, Vh = la.svd(A,full_matrices=False)
    S = np.diag(S)
    print(U.shape, S.shape, Vh.shape)

    #[2] Add watermark to S (scaled by constant factor a)
    #a = 0.1
    S_aW = S + a*W

    #[3] Decompose (S + aW) into Uw Sw Vwh
    Uw, Sw, Vwh = la.svd(S_aW,full_matrices=False)
    Sw = np.diag(Sw)

    #[4]Generate watermarked original image as U Sw Vh
    Aw = U.dot(Sw.dot(Vh))
    return(Aw,Uw,Vwh,S)

###############################
## Extract Watermark

def ExtractWatermark(Aw,Uw,Vwh,S,a):
    #[1] Decompose Aw into U Sw Vh
    U2, Sw2, Vh2 = la.svd(Aw,full_matrices=False)
    Sw2 = np.diag(Sw2)

    #[2] Create D as Uw Sw Vwh
    #(note you need Uw and Vwh from the embedding process to do this)
    D = Uw.dot(Sw2.dot(Vwh))

    #[3] Recover watermark as W = (1/a)(D - S)
    #(note you need a and S from the embedding process to do this)
    W_rec = (1/a)*(D - S)
    return(W_rec)

###############################
## Compare Two Arrays

def CompareArrays(a,b,display):
    #convert to 1D vectors
    a_1D = np.concatenate(a)
    b_1D = np.concatenate(b)

    #calc correlation
    corr = np.corrcoef(a_1D,b_1D)

    #display absolute difference image (scaled up by factor 1)
    if display==1:
        dif = np.abs(a - b)
        dif_image = Image.fromarray(dif*1)
        dif_image.show()
        return(corr,dif_image)
    else:
        return(corr)

###############################
## Create rectangular watermarked image

def crop_rectangle(Aw, img_Orig):
    img_W = Image.fromarray(Aw)
    w1, h1 = img_Orig.size
    w2, h2 = img_W.size
    img_W = img_W.crop(((w2 - w1) // 2,(h2 - h1) // 2,(w2 + w1) // 2,(h2 + h1) // 2))
    Aw_crop = np.array(img_W)
    return(img_W,Aw_crop)


###############################
###############################
###############################
## Run functions


###############################
# bring in images and embed watermark

#read original image and watermark and convert to grey scale
watermark = Image.open('..\images\Mountains.jpg').convert('LA')
original = Image.open('..\images\golden-retriever-puppy.jpg').convert('LA')

watermark.save(".\\results\\LiuTan2002_watermark.png", "png")
original.save(".\\results\\LiuTan2002_original.png", "png")


#create square images and convert to matrix
W = MakeSquareMatrix(watermark)
A = MakeSquareMatrix(original)

#embed watermark
a = 0.1
Aw,Uw,Vwh,S = EmbedWatermark(A,W,a)

# display watermarked image
original_w = Image.fromarray(Aw)
original_w.convert('LA').save(".\\results\\LiuTan2002_original_w.png", "png")


###############################
# checks

#absolute difference image
Aw_dif_image = Image.fromarray(np.abs(A - Aw)*100)
Aw_dif_image.convert('LA').save(".\\results\\LiuTan2002_Aw_dif_image.png", "png")

#check that we get the watermark back
W_rec = ExtractWatermark(Aw,Uw,Vwh,S,a)
watermark_rec_rect,W_rec_rect = crop_rectangle(W_rec,watermark)
watermark_rec_rect.convert('LA').save(".\\results\\LiuTan2002_watermark_rec_rect.png", "png")


###############################
###############################
###############################
## Experimental Results


###############################
# create noise as the "comparison" watermarks
noise = np.zeros((A.shape[0]*A.shape[1],50))
for i in range(49):
    noise[:,i] = np.random.normal(loc=0, scale=(i+1)/2,size=(A.shape[0]*A.shape[1]))
noise[:,49] = np.concatenate(W)


###############################
#[0] no_alteration

#alter image
Aw_no_alteration = Aw
original_w_no_alteration = Image.fromarray(Aw_no_alteration)

#display altered image
original_w_no_alteration.convert('LA').save(".\\results\\LiuTan2002_original_w_no_alteration.png", "png")

#extract watermark from altered image
W_rec_no_alteration = ExtractWatermark(Aw_no_alteration,Uw,Vwh,S,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_no_alteration)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Liu Tan 2002 No Alteration Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\LiuTan2002_Correlation_no_alteration.png')
plt.clf()

#display extracted watermark from altered image
extract_no_alteration = Image.fromarray(W_rec_no_alteration)
extract_no_alteration.convert('LA').save(".\\results\\LiuTan2002_extract_no_alteration.png", "png")


###############################
#[1] add gaussian noise to watermarked image

#alter image
Aw_GaussNoise = Aw + np.random.normal(loc=0, scale=10,size=(Aw.shape[0],Aw.shape[1]))
original_w_GaussNoise = Image.fromarray(Aw_GaussNoise)

#display altered image
original_w_GaussNoise.convert('LA').save(".\\results\\LiuTan2002_original_w_GaussNoise.png", "png")

#extract watermark from altered image
W_rec_GaussNoise = ExtractWatermark(Aw_GaussNoise,Uw,Vwh,S,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_GaussNoise)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Liu Tan 2002 Gaussian Noise Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\LiuTan2002_Correlation_GaussNoise.png')
plt.clf()

#display extracted watermark from altered image
extract_GaussNoise = Image.fromarray(W_rec_GaussNoise)
extract_GaussNoise.convert('LA').save(".\\results\\LiuTan2002_extract_GaussNoise.png", "png")


################################
##[2] blur filter of watermarked image

#alter image
original_w_blur = Image.fromarray(Aw).convert('LA').filter(ImageFilter.GaussianBlur(1))

#display altered image
original_w_blur.convert('LA').save(".\\results\\LiuTan2002_original_w_blur.png", "png")

#create matrix version
Aw_blur = np.array(original_w_blur)[:,:,0]

#extract watermark from altered image
W_rec_blur = ExtractWatermark(Aw_blur,Uw,Vwh,S,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_blur)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Liu Tan 2002 Blur Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\LiuTan2002_Correlation_Blur.png')
plt.clf()


#display extracted watermark from altered image
extract_blur = Image.fromarray(W_rec_blur)
extract_blur.convert('LA').save(".\\results\\LiuTan2002_extract_blur.png", "png")


###############################
#[3] image compression

#alter image
original_w_compress = Image.fromarray(Aw).convert('LA').resize((A.shape[0]//2,
                                     A.shape[1]//2)).resize((A.shape[0],A.shape[1]))

#display altered image
original_w_compress.convert('LA').save(".\\results\\LiuTan2002_original_w_compress.png", "png")

#create matrix version
Aw_compress = np.array(original_w_compress)[:,:,0]

#extract watermark from altered image
W_rec_compress = ExtractWatermark(Aw_compress,Uw,Vwh,S,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_compress)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Liu Tan 2002 Compression Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\LiuTan2002_Correlation_Compression.png')
plt.clf()

#display extracted watermark from altered image
extract_compress = Image.fromarray(W_rec_compress)
extract_compress.convert('LA').save(".\\results\\LiuTan2002_extract_compress.png", "png")


###############################
#[4] rotate image

#alter image
original_w_rotated = Image.fromarray(Aw).convert('LA').rotate(30)

#display altered image
original_w_rotated.convert('LA').save(".\\results\\LiuTan2002_original_w_rotated.png", "png")

#create matrix version
Aw_rotated = np.array(original_w_rotated)[:,:,0]

#extract watermark from altered image
W_rec_rotated = ExtractWatermark(Aw_rotated,Uw,Vwh,S,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_rotated)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Liu Tan 2002 Rotation Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\LiuTan2002_Correlation_Rotation.png')
plt.clf()

#display extracted watermark from altered image
extract_rotated = Image.fromarray(W_rec_rotated)
extract_rotated.convert('LA').save(".\\results\\LiuTan2002_extract_rotated.png", "png")



###############################
#[5] crop image

#alter image
original_w_crop = Image.fromarray(Aw).convert('LA').crop((0, 0,
                                 Image.fromarray(Aw).convert('LA').size[0]//2,
                                 Image.fromarray(Aw).convert('LA').size[1]))
Aw_crop = np.array(original_w_crop)[:,:,0]
blackspace = np.zeros((Aw_crop.shape[0],Aw_crop.shape[1]))
Aw_crop = np.concatenate((Aw_crop,blackspace),axis=1)
original_w_crop = Image.fromarray(Aw_crop)

#display altered image
original_w_crop.convert('LA').save(".\\results\\LiuTan2002_original_w_crop.png", "png")

#extract watermark from altered image
W_rec_crop = ExtractWatermark(Aw_crop,Uw,Vwh,S,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_crop)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Liu Tan 2002 Crop Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\LiuTan2002_Correlation_Crop.png')
plt.clf()


#display extracted watermark from altered image
extract_crop = Image.fromarray(W_rec_crop)
extract_crop.convert('LA').save(".\\results\\LiuTan2002_extract_crop.png", "png")

