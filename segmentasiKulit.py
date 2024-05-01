import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# Resize
citra = cv2.imread('/content/sample_data/Sample7.jpg')
# Ukuran Citra Semula
bar, kol, dlm = citra.shape
if bar > kol:
    maxLength = bar
    if maxLength >= 480:
        citra = cv2.resize(citra, (480, int(kol * (480 / bar))))
else:
    maxLength = kol
    if maxLength >= 480:
        citra = cv2.resize(citra, (int(bar * (480 / kol)), 480))

plt.imshow(cv2.cvtColor(citra, cv2.COLOR_BGR2RGB))
plt.title('Citra Resize'), plt.axis('off')
plt.show()

# Segmentasi
citraG = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
_, citraB = cv2.threshold(citraG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
citraF = cv2.morphologyEx(citraB, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
_, citraCC = cv2.connectedComponents(citraF)
citraL = np.zeros_like(citraCC)
for i in range(1, np.max(citraCC) + 1):
    citraL[citraCC == i] = i

# Ukuran Citra Setelah Resize
bar, kol, dlm = citra.shape

# Cari Connected Component Terbanyak
terluas = 0
index = 0
for i in range(1, np.max(citraCC) + 1):
    if np.sum(citraCC == i) > terluas:
        terluas = np.sum(citraCC == i)
        index = i

# Ambil Label
wajah = np.zeros_like(citra)
for i in range(bar):
    for j in range(kol):
        if citraL[i, j] == index:
            wajah[i, j, :] = citra[i, j, :]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(citraG, cmap='gray')
ax[0].set_title('Segmentasi Gray')

ax[1].imshow(citraB, cmap='gray')
ax[1].set_title('Segmentasi biner')

ax[2].imshow(citraF, cmap='gray')
ax[2].set_title('Segmentasi imfill')

ax[3].imshow(cv2.cvtColor(wajah, cv2.COLOR_BGR2RGB))
ax[3].set_title('Segmentasi wajah')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# Perbaikan Citra
# Sharpening
wajahG = cv2.cvtColor(wajah, cv2.COLOR_BGR2GRAY)
h = np.array([[0, 0, -1, 0, 0],
              [0, -1, -2, -1, 0],
              [-1, -2, 16, -2, -1],
              [0, -1, -2, -1, 0],
              [0, 0, -1, 0, 0]], dtype=np.float32)
m = cv2.filter2D(wajahG, -1, h)

# Labeling
candidate = m.astype(np.uint8)
_, labeledCandidate = cv2.connectedComponents(candidate)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(wajahG, cmap='gray')
ax[0].set_title('Perbaikaan Citra Wajah')

ax[1].imshow(candidate, cmap='gray')
ax[1].set_title('Candidate')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# Bentuk
# Eliminasi Berdasarkan Luas Dan Bentuk
blobMeasurements = regionprops(labeledCandidate)
allArea = [blob.area for blob in blobMeasurements]
allEccentricity = [blob.eccentricity for blob in blobMeasurements]
meanArea = np.mean(allArea)
stdArea = np.std(allArea)
indexBlob = np.where((np.array(allArea) >= 24) & (np.array(allArea) <= (meanArea + stdArea)) & (np.array(allEccentricity) < 0.81))[0]
ambilBlob = np.isin(labeledCandidate, indexBlob)
blobBW = ambilBlob.astype(np.uint8)
labeledBlob, numberOfBlobs = label(blobBW, connectivity=2, return_num=True)
plt.imshow(blobBW, cmap='gray')
plt.title('Bentuk'), plt.axis('off')
plt.show()

# Warna
# Eliminasi Berdasarkan Warna
red = citra[:, :, 0]
green = citra[:, :, 1]
blue = citra[:, :, 2]
r = regionprops(labeledBlob, intensity_image=red)
g = regionprops(labeledBlob, intensity_image=green)
b = regionprops(labeledBlob, intensity_image=blue)
fiturR = [blob.mean_intensity for blob in r]
fiturG = [blob.mean_intensity for blob in g]
fiturB = [blob.mean_intensity for blob in b]
fitur = np.column_stack((fiturR, fiturG, fiturB))
meanR = np.mean(fiturR)
meanG = np.mean(fiturG)
meanB = np.mean(fiturB)
stdR = np.std(fiturR)
stdG = np.std(fiturG)
stdB = np.std(fiturB)
indexJerawat = []
for i in range(numberOfBlobs):
    if (fiturR[i] >= (meanR - stdR * 1.75) and fiturR[i] <= (meanR + stdR * 1.75) and
        fiturG[i] >= (meanG - stdG * 1.75) and fiturG[i] <= (meanG + stdG * 1.75) and
        fiturB[i] >= (meanB - stdB * 1.75) and fiturB[i] <= (meanB + stdB * 1.75)):
        indexJerawat.append(i)
jumlahJerawat = len(indexJerawat)
jerawatBW = np.isin(labeledBlob, indexJerawat).astype(np.uint8)
plt.imshow(jerawatBW, cmap='gray')
plt.title('Warna'), plt.axis('off')
plt.show()

# Marking
jerawatEdge = cv2.Canny(jerawatBW, 100, 200)
hasil = citra.copy()
for i in range(bar):
    for j in range(kol):
        if jerawatEdge[i, j] == 255:
            hasil[i, j, 0] = 0
            hasil[i, j, 1] = 255
            hasil[i, j, 2] = 0
hasil = hasil.astype(np.uint8)
plt.subplot(121), plt.imshow(jerawatEdge, cmap='gray')
plt.title('Hasil Akhir 1'), plt.axis('off')
plt.subplot(122), plt.imshow(cv2.cvtColor(hasil, cv2.COLOR_BGR2RGB))
plt.title('Hasil Akhir 2'), plt.axis('off')
plt.show()
