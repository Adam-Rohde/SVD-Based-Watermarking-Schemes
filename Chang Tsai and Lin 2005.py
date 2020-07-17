'''
Chang, C., Tsai, P., & Lin, C.  (2005).  Svd-based digital image watermarking scheme.Pattern Recognition Letters,26, 1577-1586.
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

def EmbedWatermark(A,W,t,nb):

    np.random.seed(seed=100)

    #[1] divide original image into blocks and turn watermark into 1D vector
    bsize = int(A.shape[0]/nb)
    A_Blocks = np.zeros((bsize,bsize,int(A.shape[0]/bsize)*int(A.shape[0]/bsize)))
    k = 0
    for i in range(int(A.shape[0]/bsize)):
        for j in range(int(A.shape[1]/bsize)):
            A_Blocks[:,:,k] = A[(i)*bsize:min((i)*bsize+bsize,A.shape[0]),(j)*bsize:min((j)*bsize+bsize,A.shape[1])]
            k = k + 1
    W_long = W.reshape(int(W.shape[0]*W.shape[0]))

    #[2] SVD Decompose each block into U S Vh
    USV = np.zeros((A_Blocks.shape[0],A_Blocks.shape[1],3,A_Blocks.shape[2]))
    for i in range(A_Blocks.shape[2]):
        U, S, Vh = la.svd(A_Blocks[:,:,i],full_matrices=False)
        S = np.diag(S)
        USV[:,:,0,i],USV[:,:,1,i],USV[:,:,2,i] = U, S, Vh

    #[3] Determine block importance/complexity
    #D-Feature = # non zero components in D (here D is our S matrix)
    #the greater number of non-zero coefficients would indicate greater complexity
    #For a block-based watermarking scheme, a more complex block was favored for embedding a watermark with perceptibility
    #Applying the feature of the D component prevents the smooth blocks from being selected
    #and benefits the perceptibility of the watermarked image
    DFeature = np.zeros((A_Blocks.shape[2],2))
    for i in range(A_Blocks.shape[2]):
        DFeature[i,0] = i
        DFeature[i,1] = sum(np.round(np.diag(USV[:,:,1,i]),10)!=0)

    #[4] Select random sample of (W_long.shape[0]) blocks with max D-Feature
    #we sample the same number of blocks as we have pixels in the watermark
    #Using the pseudo random number generator (PRNG) increases the watermarking security.
    BlocksIDs = np.random.choice(DFeature[DFeature[:,1]==np.max(DFeature[:,1]),0],size=int(W_long.shape[0]),replace=False)

    #[5] For selected blocks, find U-feature
    #U-Feature = magnitude difference between the neighboring elements in first column of U
    #we only compare element (2,1) and (3,1) for (row, col)

    for i, b in enumerate(BlocksIDs):
        b=int(b)

        #coefficient reassignment
        #goal is to have a positive difference when the watermark pixel is 1
        #                a negative difference when the watermark pixel is 0
        #so when we already see the correct difference, we make it larger and
        #when we see the opposite difference, we switch it to the correct one

        d = (np.abs(USV[1,0,0,b])-np.abs(USV[2,0,0,b]))
        #positive difference
        if d>0:
            #positive difference, match
            if W_long[i]==True:
                USV[1,0,0,b] = -np.abs( np.abs(USV[1,0,0,b]) + (d+t)/2 )
                USV[2,0,0,b] = -np.abs( np.abs(USV[2,0,0,b]) - (d+t)/2 )
            #positive difference, no match
            if W_long[i]==False:
                USV[1,0,0,b] = -np.abs( np.abs(USV[1,0,0,b]) - (d+t)/2 )
                USV[2,0,0,b] = -np.abs( np.abs(USV[2,0,0,b]) + (d+t)/2 )

        #negaitve difference
        if d<=0:
            #negaitve difference, match
            if W_long[i]==False:
                USV[1,0,0,b] = -np.abs( np.abs(USV[1,0,0,b]) + (d-t)/2 )
                USV[2,0,0,b] = -np.abs( np.abs(USV[2,0,0,b]) - (d-t)/2 )
            #negaitve difference, no match
            if W_long[i]==True:
                USV[1,0,0,b] = -np.abs( np.abs(USV[1,0,0,b]) - (d-t)/2 )
                USV[2,0,0,b] = -np.abs( np.abs(USV[2,0,0,b]) + (d-t)/2 )

    #[6] inverse SVD (U Sw Vh) to get watermarked blocks
    A_Blocks_Water = np.zeros((A_Blocks.shape[0],A_Blocks.shape[1],A_Blocks.shape[2]))
    for i in range(USV.shape[3]):
        A_Blocks_Water[:,:,i] = USV[:,:,0,i].dot(USV[:,:,1,i].dot(USV[:,:,2,i]))

    #[7] Generate watermarked original image by composing blocks
    Aw = np.zeros((A.shape[0],A.shape[1]))
    k = 0
    for i in range(int(nb)):
        for j in range(int(nb)):
            Aw[(i)*bsize:min((i)*bsize+bsize,A.shape[0]),(j)*bsize:min((j)*bsize+bsize,A.shape[1])] = A_Blocks_Water[:,:,k]
            k = k + 1

    return(Aw,BlocksIDs)


###############################
## Extract Watermark

def ExtractWatermark(Aw,BlocksIDs,nb):

    np.random.seed(seed=100)

    #[1] divide watermarked image into blocks and create empty watermark to be populated
    bsize = int(Aw.shape[0]/nb)
    A_Blocks_Water2 = np.zeros((bsize,bsize,int(Aw.shape[0]/bsize)*int(Aw.shape[0]/bsize)))
    k = 0
    for i in range(int(Aw.shape[0]/bsize)):
        for j in range(int(Aw.shape[1]/bsize)):
            A_Blocks_Water2[:,:,k] = Aw[(i)*bsize:min((i)*bsize+bsize,Aw.shape[0]),(j)*bsize:min((j)*bsize+bsize,Aw.shape[1])]
            k = k + 1
    W_rec_long = np.ones(BlocksIDs.shape[0], dtype=bool)

    #[2] SVD Decompose each block into U S Vh
    USV_w = np.zeros((A_Blocks_Water2.shape[0],A_Blocks_Water2.shape[1],3,A_Blocks_Water2.shape[2]))
    for i in range(A_Blocks_Water2.shape[2]):
        U, S, Vh = la.svd(A_Blocks_Water2[:,:,i],full_matrices=False)
        S = np.diag(S)
        USV_w[:,:,0,i],USV_w[:,:,1,i],USV_w[:,:,2,i] = U, S, Vh

    #[3] Generate recovered watermark from U values of watermarked blocks
    for i, b in enumerate(BlocksIDs):
        b=int(b)

        d = (np.abs(USV_w[1,0,0,b])-np.abs(USV_w[2,0,0,b]))
        #positive relationship
        if d>0:
            W_rec_long[i] = True

        #negative relationship
        if d<=0:
            W_rec_long[i] = False

    #[4] Form recovered watermark
    W_rec = W_rec_long.reshape(int(np.sqrt(BlocksIDs.shape[0])),int(np.sqrt(BlocksIDs.shape[0])))

    return(W_rec)


###############################
## Compare Two Arrays

def CompareArrays(a,b,display):
    #convert to 1D vectors
    a_1D = 1*(np.concatenate(a))
    b_1D = 1*(np.concatenate(b))

    #calc correlation
    corr = np.corrcoef(a_1D,b_1D)

    #display absolute difference image (scaled up by factor 1)
    if display==1:
        dif = np.abs(1*a - 1*b)
        dif_image = Image.fromarray(dif*1)
        dif_image.show()
        return(corr,dif_image)
    else:
        return(corr)

###############################
## Create rectangular watermarked image

def crop_rectangle(Aw, img_Orig):
    img_W = Image.fromarray(Aw).convert('LA')
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

#read original image and convert to grey scale
original = Image.open('..\images\golden-retriever-puppy.jpg').convert('LA')

#read watermark image and convert to black and white
watermark = Image.open('..\images\mario1.png').convert('1')

#resize watermark
watermark = watermark.resize((32, 32))

watermark.save(".\\results\\ChangTsaiLin2005_watermark.png", "png")
original.save(".\\results\\ChangTsaiLin2005_original.png", "png")

#create square images and convert to matrix
W = MakeSquareMatrix(watermark)
A = MakeSquareMatrix(original)

#embed watermark
t=0.01
nb = 100
Aw,BlocksIDs = EmbedWatermark(A,W,t,nb)

# display watermarked image
original_w = Image.fromarray(Aw)
original_w.convert('LA').save(".\\results\\ChangTsaiLin2005_original_w.png", "png")


###############################
# checks

#absolute difference image
Aw_dif_image = Image.fromarray(np.abs(A - Aw)*100)
Aw_dif_image.convert('LA').save(".\\results\\ChangTsaiLin2005_Aw_dif_image.png", "png")

#check that we get the watermark back
W_rec = ExtractWatermark(Aw,BlocksIDs,nb)
watermark_rec_rect,W_rec_rect = crop_rectangle(W_rec,watermark)
watermark_rec_rect.convert('LA').save(".\\results\\ChangTsaiLin2005_watermark_rec_rect.png", "png")



###############################
###############################
###############################
## Experimental Results

###############################
# create noise as the "comparison" watermarks
noise = np.zeros((W.shape[0]*W.shape[1],50))
for i in range(49):
    noise[:,i] = np.random.binomial(p=0.5,n=1,size=(W.shape[0]*W.shape[1]))
noise[:,49] = np.concatenate(W)


###############################
#[0] no_alteration

#alter image
Aw_no_alteration = Aw
original_w_no_alteration = Image.fromarray(Aw_no_alteration)

#display altered image
original_w_no_alteration.convert('LA').save(".\\results\\ChangTsaiLin2005_original_w_no_alteration.png", "png")

#extract watermark from altered image
W_rec_no_alteration = ExtractWatermark(Aw_no_alteration,BlocksIDs,nb)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_no_alteration)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Chang Tsai Lin 2005 No Alteration Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\ChangTsaiLin2005_Correlation_no_alteration.png')
plt.clf()

#display extracted watermark from altered image
extract_no_alteration = Image.fromarray(W_rec_no_alteration)
extract_no_alteration.convert('LA').save(".\\results\\ChangTsaiLin2005_extract_no_alteration.png", "png")


###############################
#[1] add gaussian noise to watermarked image

#alter image
Aw_GaussNoise = Aw + np.random.normal(loc=0, scale=10,size=(Aw.shape[0],Aw.shape[1]))
original_w_GaussNoise = Image.fromarray(Aw_GaussNoise)

#display altered image
original_w_GaussNoise.convert('LA').save(".\\results\\ChangTsaiLin2005_original_w_GaussNoise.png", "png")

#extract watermark from altered image
W_rec_GaussNoise = ExtractWatermark(Aw_GaussNoise,BlocksIDs,nb)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_GaussNoise)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Chang Tsai Lin 2005 Gaussian Noise Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\ChangTsaiLin2005_Correlation_GaussNoise.png')
plt.clf()

#display extracted watermark from altered image
extract_GaussNoise = Image.fromarray(W_rec_GaussNoise)
extract_GaussNoise.convert('LA').save(".\\results\\ChangTsaiLin2005_extract_GaussNoise.png", "png")


################################
##[2] blur filter of watermarked image

#alter image
original_w_blur = Image.fromarray(Aw).convert('LA').filter(ImageFilter.GaussianBlur(1))

#display altered image
original_w_blur.convert('LA').save(".\\results\\ChangTsaiLin2005_original_w_blur.png", "png")

#create matrix version
Aw_blur = np.array(original_w_blur)[:,:,0]

#extract watermark from altered image
W_rec_blur = ExtractWatermark(Aw_blur,BlocksIDs,nb)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_blur)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Chang Tsai Lin 2005 Blur Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\ChangTsaiLin2005_Correlation_Blur.png')
plt.clf()

#display extracted watermark from altered image
extract_blur = Image.fromarray(W_rec_blur)
extract_blur.convert('LA').save(".\\results\\ChangTsaiLin2005_extract_blur.png", "png")


###############################
#[3] image compression

#alter image
original_w_compress = Image.fromarray(Aw).convert('LA').resize((A.shape[0]//2,A.shape[1]//2)).resize((A.shape[0],A.shape[1]))

#display altered image
original_w_compress.convert('LA').save(".\\results\\ChangTsaiLin2005_original_w_compress.png", "png")

#create matrix version
Aw_compress = np.array(original_w_compress)[:,:,0]

#extract watermark from altered image
W_rec_compress = ExtractWatermark(Aw_compress,BlocksIDs,nb)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_compress)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Chang Tsai Lin 2005 Compression Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\ChangTsaiLin2005_Correlation_Compression.png')
plt.clf()

#display extracted watermark from altered image
extract_compress = Image.fromarray(W_rec_compress)
extract_compress.convert('LA').save(".\\results\\ChangTsaiLin2005_extract_compress.png", "png")


###############################
#[4] rotate image

#alter image
original_w_rotated = Image.fromarray(Aw).convert('LA').rotate(30)

#display altered image
original_w_rotated.convert('LA').save(".\\results\\ChangTsaiLin2005_original_w_rotated.png", "png")

#create matrix version
Aw_rotated = np.array(original_w_rotated)[:,:,0]

#extract watermark from altered image
W_rec_rotated = ExtractWatermark(Aw_rotated,BlocksIDs,nb)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_rotated)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Chang Tsai Lin 2005 Rotation Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\ChangTsaiLin2005_Correlation_Rotation.png')
plt.clf()

#display extracted watermark from altered image
extract_rotated = Image.fromarray(W_rec_rotated)
extract_rotated.convert('LA').save(".\\results\\ChangTsaiLin2005_extract_rotated.png", "png")


###############################
#[5] crop image

#alter image
original_w_crop = Image.fromarray(Aw).convert('LA').crop((0, 0, Image.fromarray(Aw).convert('LA').size[0]//2, Image.fromarray(Aw).convert('LA').size[1]))
Aw_crop = np.array(original_w_crop)[:,:,0]
blackspace = np.zeros((Aw_crop.shape[0],Aw_crop.shape[1]))
Aw_crop = np.concatenate((Aw_crop,blackspace),axis=1)
original_w_crop = Image.fromarray(Aw_crop)

#display altered image
original_w_crop.convert('LA').save(".\\results\\ChangTsaiLin2005_original_w_crop.png", "png")

#extract watermark from altered image
W_rec_crop = ExtractWatermark(Aw_crop,BlocksIDs,nb)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_crop)
corr = np.zeros(50)
for i in range(50):
    corr[i] = np.corrcoef(W_check,noise[:,i])[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('Chang Tsai Lin 2005 Crop Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\results\\ChangTsaiLin2005_Correlation_Crop.png')
plt.clf()

#display extracted watermark from altered image
extract_crop = Image.fromarray(W_rec_crop)
extract_crop.convert('LA').save(".\\results\\ChangTsaiLin2005_extract_crop.png", "png")

