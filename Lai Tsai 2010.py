'''
Lai, C., & Tsai, C.  (2010).  Digital image watermarking using discrete wavelet transform and singular value decom-position.IEEE Transactions on Instrumentation and Measurement,59(11), 3060-3063.

Note that this code draws from the Liu and Tan 2002 code
'''

#Source: Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary (2019).
#PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237, https://doi.org/10.21105/joss.01237.
import pywt

#Preform DWT
coeffs2 = pywt.dwt2(A, 'haar')
LL, (LH, HL, HH) = coeffs2

#Make Watermark same size as the LH and HL matrices
W_sized = W[range(0,(int(W.shape[0]/2))),:]
W_sized = W_sized[:,range(int(W.shape[0]/2))]
halfW = .5*W_sized

#preform watermarking procedure, with watermark halfW, on LH and HL
Aw1,Uw1,Vwh1,S1 = EmbedWatermark(LH,halfW,0.1)
Aw2,Uw2,Vwh2,S2 = EmbedWatermark(HL,halfW,0.1)

#Preform reverse DWT procedure to get Aw
coefs = LL, (Aw1, Aw2, HH)
Aw = pywt.idwt2(coefs, 'haar')
Aw_img = Image.fromarray(Aw)
Aw_img.show()


def ExtractWatermarkDWT(Aw,Uw1,Vwh1,S1,Uw2,Vwh2,S2,a):
    #Break watermarked image using DWT into LL, LH,HL, HH
    coefs_Aw = pywt.dwt2(Aw, 'haar')
    LLw, (LHw, HLw, HHw)= coefs_Aw
    #extract half watermark from LHw and HLw, add together and display
    W_1 = ExtractWatermark(LHw,Uw1,Vwh1,S1,0.1)
    W_2 = ExtractWatermark(HLw,Uw2,Vwh2,S2,0.1)
    W_DWT = W_1 + W_2
    return W_DWT

W_DWT = ExtractWatermarkDWT(Aw,Uw1,Vwh1,S1,Uw2,Vwh2,S2,0.1)

#create random watermarks for half iamge
noise  = np.zeros((W_sized.shape[0],W_sized.shape[1],50))
for i in range(49):
    noise[:,:,i] = np.random.normal(loc=0, scale=(i+1)/2,size=(W_sized.shape[0],W_sized.shape[1]))
noise[:,:,49] = W_DWT


###############################
#[0] no_alteration
a = .01
#alter image
Aw_no_alteration = Aw
original_w_no_alteration = Image.fromarray(Aw_no_alteration)

#display altered image
original_w_no_alteration.convert('LA').save(".\\DWT_original_w_no_alteration.png", "png")

#extract watermark from altered image
W_rec_no_alteration = ExtractWatermarkDWT(Aw_no_alteration,Uw1,Vwh1,S1,Uw2,Vwh2,S2,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_no_alteration)
corr = np.zeros(50)
for i in range(50):
     corr[i] = np.corrcoef(W_check,np.concatenate(noise[:,:,i]))[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('DWT No Alteration Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\DWT_Correlation_no_alteration.png')
plt.clf()

#display extracted watermark from altered image
extract_no_alteration = Image.fromarray(W_rec_no_alteration)
extract_no_alteration.convert('LA').save(".\\DWT_extract_no_alteration.png", "png")


###############################
#[1] add gaussian noise to watermarked image

#alter image
Aw_GaussNoise = Aw + np.random.normal(loc=0, scale=10,size=(Aw.shape[0],Aw.shape[1]))
original_w_GaussNoise = Image.fromarray(Aw_GaussNoise)

#display altered image
original_w_GaussNoise.convert('LA').save(".\\DWT_original_w_GaussNoise.png", "png")

#extract watermark from altered image
W_rec_GaussNoise = ExtractWatermarkDWT(Aw_GaussNoise,Uw1,Vwh1,S1,Uw2,Vwh2,S2,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_GaussNoise)
corr = np.zeros(50)
for i in range(50):
     corr[i] = np.corrcoef(W_check,np.concatenate(noise[:,:,i]))[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('DWT Gaussian Noise Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\DWT_Correlation_GaussNoise.png')
plt.clf()

#display extracted watermark from altered image
extract_GaussNoise = Image.fromarray(W_rec_GaussNoise)
extract_GaussNoise.convert('LA').save(".\\DWT_extract_GaussNoise.png", "png")


################################
##[2] blur filter of watermarked image

#alter image
original_w_blur = Image.fromarray(Aw).convert('LA').filter(ImageFilter.GaussianBlur(1))

#display altered image
original_w_blur.convert('LA').save(".\\DWT_original_w_blur.png", "png")

#create matrix version
Aw_blur = np.array(original_w_blur)[:,:,0]

#extract watermark from altered image
W_rec_blur = ExtractWatermarkDWT(Aw_blur,Uw1,Vwh1,S1,Uw2,Vwh2,S2,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_blur)
corr = np.zeros(50)
for i in range(50):
     corr[i] = np.corrcoef(W_check,np.concatenate(noise[:,:,i]))[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('DWT Blur Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\DWT_Correlation_Blur.png')
plt.clf()

#display extracted watermark from altered image
extract_blur = Image.fromarray(W_rec_blur)
extract_blur.convert('LA').save(".\\DWT_extract_blur.png", "png")


###############################
#[3] image compression

#alter image
x = Image.fromarray(Aw).resize((100, 100))
Aw_compress = np.array(x.resize((Aw.shape[0],Aw.shape[1])))
original_w_compress = Image.fromarray(Aw_compress)

#display altered image
original_w_compress.convert('LA').save(".\\DWT_original_w_compress.png", "png")

#extract watermark from altered image
W_rec_compress = ExtractWatermarkDWT(Aw_compress,Uw1,Vwh1,S1,Uw2,Vwh2,S2,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_compress)
corr = np.zeros(50)
for i in range(50):
     corr[i] = np.corrcoef(W_check,np.concatenate(noise[:,:,i]))[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('DWT Compression Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\DWT_Correlation_Compression.png')
plt.clf()

#display extracted watermark from altered image
extract_compress = Image.fromarray(W_rec_compress)
extract_compress.convert('LA').save(".\\DWT_extract_compress.png", "png")


###############################
#[4] rotate image

#alter image
original_w_rotated = Image.fromarray(Aw).convert('LA').rotate(30)

#display altered image
original_w_rotated.convert('LA').save(".\\DWT_original_w_rotated.png", "png")

#create matrix version
Aw_rotated = np.array(original_w_rotated)[:,:,0]

#extract watermark from altered image
W_rec_rotated = ExtractWatermarkDWT(Aw_rotated,Uw1,Vwh1,S1,Uw2,Vwh2,S2,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_rotated)
corr = np.zeros(50)
for i in range(50):
     corr[i] = np.corrcoef(W_check,np.concatenate(noise[:,:,i]))[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('DWT Rotation Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\DWT_Correlation_Rotation.png')
plt.clf()

#display extracted watermark from altered image
extract_rotated = Image.fromarray(W_rec_rotated)
extract_rotated.convert('LA').save(".\\DWT_extract_rotated.png", "png")


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
original_w_crop.convert('LA').save(".\\DWT_original_w_crop.png", "png")

#extract watermark from altered image
W_rec_crop = ExtractWatermarkDWT(Aw_crop,Uw1,Vwh1,S1,Uw2,Vwh2,S2,a)

#plot correlations for different "comparison" watermarks
W_check = np.concatenate(W_rec_crop)
corr = np.zeros(50)
for i in range(50):
     corr[i] = np.corrcoef(W_check,np.concatenate(noise[:,:,i]))[0,1]
plt.plot(corr, 'bo')
plt.ylim(0,1)
plt.title('DWT Crop Correlation')
plt.xlabel('Noise and Watermark')
plt.ylabel('Correlation')
plt.savefig('.\\DWT_Correlation_Crop.png')
plt.clf()


#display extracted watermark from altered image
extract_crop = Image.fromarray(W_rec_crop)
extract_crop.convert('LA').save(".\\DWT_extract_crop.png", "png")



