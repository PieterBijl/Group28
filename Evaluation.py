from PIL import Image, ImageChops
import os.path
import numpy as np

# --- Change this directory to where the test pictures are stored
testim = '/Users/User/Documents/Pictures/Test'

# --- Gets all the test images' names stored and sorted in array 'a'
# --- It gets sorted as: Generated 1, Blurry 1, Original 1, Generated 2, Blurry 2, Original 2, etc.

a = []
for image in os.listdir(testim):
    a.append(image)
a.sort()


# -----------------------------------------------------------------------------
# --- This block will calculate the PSNR for the pixelated/blurry images compared to the sharp
# --- This value will not change for the same dataset so only has to be calculated once
# --- Feel free to comment this section out if you already know the dataset's PSNR
# --- Although some print statements later will use 'PSNRref'

imref = []
k = 0
for image in a:
    k+=1
    if k%3 == 2:
        imref.append(image)
    elif k%3 == 0:
        imref.append(image)


i = 0
PSNRref = np.zeros((int(len(imref)/2)))
for image in imref:
    i+=1
    if i%2 ==1:
        pixels = Image.open(os.path.join(testim,image)).resize((256,256))
        pixels = np.asarray(pixels).astype('int32')
    else:
        sharp = Image.open(os.path.join(testim,image))
        sharp = np.asarray(sharp).astype('int32')
        ref = abs(sharp-pixels).astype('uint8')
        mseref = np.sum(ref.astype('uint32')**2)/196608
        PSNRref[int(i/2-1)] = 20*np.log10(sharp.max())-10*np.log10(mseref)
        

# ---------------------------------------------------------------------------
# --- The next part is model dependent and calculates PSNR for the generated images

imgen = []
k = 0
for image in a:
    k += 1
    if k%3 == 1:
        imgen.append(image)
    elif k%3 == 0:
        imgen.append(image)

i = 0
PSNRgen = np.zeros((int(len(imgen)/2)))
for image in imgen:
    i += 1
    if i%2 == 1:
        fake = Image.open(os.path.join(testim,image))
        fake = np.asarray(fake).astype('int32')
    else:
        sharp = Image.open(os.path.join(testim,image))
        sharp = np.asarray(sharp).astype('int32')
        gen = abs(sharp-fake).astype('uint8')
        msegen = np.sum(gen.astype('uint32')**2)/196608
        PSNRgen[int(i/2-1)] = 20*np.log10(sharp.max())-10*np.log10(msegen)

print('Difference (negative = worse than pixelated):\n',PSNRgen-PSNRref)
avgref = np.mean(PSNRref)
avggen = np.mean(PSNRgen)
print()
print('Average PSNR for pixelated:',avgref)
print('Average PSNR for generated:',avggen)
print()


# --------------------------------------------------------------------------
# --- This section calculates the cosine similarity between pixelated and real (sharp) images
# --- Comment out the rest of the script if you're only interested in the PSNR calculated above

i = 0
simref = np.zeros((int(len(a)/3)))
for image in imref:
    i+=1
    if i%2 ==1:
        pixels = Image.open(os.path.join(testim,image)).resize((256,256))
        pixels = np.asarray(pixels).astype('int64')
        pixels = np.resize(pixels,(196608))
    else:
        sharp = Image.open(os.path.join(testim,image))
        sharp = np.asarray(sharp).astype('int64')
        sharp = np.resize(sharp,(196608))
        simref[int(i/2-1)] = np.dot(pixels,sharp)/(np.linalg.norm(pixels)*np.linalg.norm(sharp))

# --------------------------------------------------------------------------
# --- This section calculates the cosine similarity between generated and real (sharp) images

i = 0
simgen = np.zeros((int(len(a)/3)))
for image in imgen:
    i += 1
    if i%2 ==1:
        fake = Image.open(os.path.join(testim,image)).resize((256,256))
        fake = np.asarray(fake).astype('int64')
        fake = np.resize(fake,(196608))
    else:
        sharp = Image.open(os.path.join(testim,image))
        sharp = np.asarray(sharp).astype('int64')
        sharp = np.resize(sharp,(196608))
        simgen[int(i/2-1)] = np.dot(fake,sharp)/(np.linalg.norm(fake)*np.linalg.norm(sharp))

# --- This is something that we made up to compare the values for the cosine similarity

Diff = (simgen-simref)*100 # --- as a percentage now
print('A percentage-wise difference between the generated and pixelated images')
print(Diff)
print()
print('Average similarity for pixelated images:',np.mean(simref)*100,'%')
print('Average similarity for generated images:',np.mean(simgen)*100,'%')




# ---------------------------------------------------------------------------------------

# --- If you're interested in visualising the difference between pictures, use the following
# --- Enter the name of the picture you want to visualise
fake = '005001_fake_B.png'  #- meaning: generated image
blur = '005001_real_A.png'  #- meaning: pixelated image
sharp= '005001_real_B.png'  #- meaning: real image


# --- If you're interested in saving the result on your desktop or anywhere else, change this location to yours
# --- You don't have to save them to see them. They will pop up anyways
dird = '/Users/User/Desktop'


# --- If you want to see the difference in colour (c), use this part

fakec = Image.open(os.path.join(testim,fake))
fakec = np.asarray(fakec).astype('int16')
blurc = Image.open(os.path.join(testim,blur)).resize((256,256))
blurc = np.asarray(blurc).astype('int16')
sharpc = Image.open(os.path.join(testim,sharp))
sharpc = np.asarray(sharpc).astype('int16')

refc = abs(sharpc-blurc).astype('uint8')
genc = abs(sharpc-fakec).astype('uint8')

refexamplec = Image.fromarray(refc)
genexamplec = Image.fromarray(genc)

# --- Uncomment this to save the image to your directory
#refexamplec.save(os.path.join(dird,'refexamplec.png'))
#genexamplec.save(os.path.join(dird,'genexamplec.png'))

# --- Uncomment this to show the images
#refexamplec.show()
#genexamplec.show()



# --- If you want to see the difference in black and white (bw), use this part
fakebw = Image.open(os.path.join(testim,fake)).convert('L', (0.2989, 0.5870, 0.1140, 0))
fakebw = np.asarray(fakebw).astype('int16')
blurbw = Image.open(os.path.join(testim,blur)).convert('L', (0.2989, 0.5870, 0.1140, 0)).resize((256,256))
blurbw = np.asarray(blurbw).astype('int16')
sharpbw = Image.open(os.path.join(testim,sharp)).convert('L', (0.2989, 0.5870, 0.1140, 0))
sharpbw = np.asarray(sharpbw).astype('int16')

refbw = abs(sharpbw-blurbw).astype('uint8')
genbw = abs(sharpbw-fakebw).astype('uint8')

refexamplebw = Image.fromarray(refbw)
genexamplebw = Image.fromarray(genbw)

# --- Uncomment this to save the image to your directory
#refexamplebw.save(os.path.join(dird,'refexamplebw.png'))
#genexamplebw.save(os.path.join(dird,'genexamplebw.png'))

# --- Uncomment this to show the images
#refexamplebw.show()
#genexamplebw.show()
