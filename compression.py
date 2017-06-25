from skimage import data
import skimage
import skimage.transform as transform
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

angles = [0, 90, 180, 270]
directions = [1, -1]

def rotation(block, angle):
    return transform.rotate(block, angle)


def reflection(block, direction):
    return block[::direction, :]


def down_sample(block, factor):
    return transform.downscale_local_mean(block, (factor, factor))


def apply_transforms(block, angle, direction):
    return rotation(reflection(block, direction), angle)


def apply_contrat_transforms(block, angle, direction, contrast, brightness):
    return contrast*rotation(reflection(block, direction), angle) + brightness

def fit_contrast_brightness(block1, block2):
    A = np.c_[np.ones(block2.size), block2.reshape(block2.size)]
    b = block1.reshape(block1.size)
    x, residuals, rank, s = np.linalg.lstsq(A,b)
    return x[1], x[0]


def get_transformed_blocks(img, region_size, domaine_size):
    transformed_blocks = list()
    factor = domaine_size//region_size
    for i in range(len(img)//domaine_size + 1):
        for j in range(len(img)//domaine_size + 1):
            for direction, angle in [(direction, angle) for direction in directions for angle in angles]:
                    block = img[i*region_size:i*region_size + domaine_size, j*region_size:j*region_size + domaine_size]
                    transformed_blocks.append([i, j, angle, direction, apply_transforms(down_sample(block, factor), angle, direction)])
    return transformed_blocks


def compress(img, region_size, domaine_size, to_csv=False, name="Save.csv"):
    transformed_blocks = get_transformed_blocks(img, region_size, domaine_size)
    
    transforms = [[None for k in range(len(img) // region_size)] for j in range(len(img) // region_size)]
    c = 0
    for i in range(len(img) // region_size):
        for j in range(len(img) // region_size):
            if c%64 == 0:
                print("Step {}/{}".format(c+1, len(img) // region_size * len(img) // region_size))
            min_ms = np.inf
            for k, l, angle, direction, transformed_block in transformed_blocks:
                domaine = transformed_block
                region = img[i*region_size:(i+1)*region_size, j*region_size:(j+1)*region_size]
                contrast, brightness = fit_contrast_brightness(region, domaine)
                ms_diff = np.sum(np.square(contrast*domaine + brightness - region))    
                if ms_diff < min_ms:
                    min_ms = ms_diff
                    i_, j_, angle_, direction_, contrast_, brightness_ = k, l, angle, direction, contrast, brightness
            transforms[i][j] = [i_, j_, angle_, direction_, contrast_, brightness_]
            c += 1
    return transforms


def decompress(transformations, region_size, domaine_size, img_size, nbr_iter, save_every=2, name="name", save_fig=False):
    current_image = np.random.rand(img_size[0], img_size[1])
    factor = domaine_size // region_size
    checkpoints = list()
    for it in range(nbr_iter):
        if save_fig:
            checkpoints.append(current_image)
            plt.imshow(current_image, cmap="gray")
            plt.savefig("{}{}.png".format(name, it))
        print("Step {}/{}".format(it+1, nbr_iter))
        for i in range(img_size[0] // region_size):
            for j in range(img_size[0] // region_size):
                k, l, angle, direction, contrast, brightness = transformations[i][j]
                domaine = down_sample(current_image[k*region_size:k*region_size + domaine_size, l*region_size:l*region_size + domaine_size], factor)
                block = apply_contrat_transforms(domaine, angle, direction, contrast, brightness)
                current_image[i*region_size:(i+1)*region_size, j*region_size:(j+1)*region_size] = block
    return current_image


def test_compress():
    #img = rgb2gray(data.astronaut())
    img = skimage.io.imread("monkey.gif", as_grey=True)
    img = down_sample(img, 2)
    plt.imshow(img, cmap="gray")
    plt.show()
    print(img.shape)
    b = compress(img, 4, 8)
    print("-- Image compressed")
    img_dec = decompress(b, 4, 8, img.shape, 8, name="astronaut", save_fig=True)
    print("-- Image decompressed")
    plt.imshow(img_dec, cmap="gray")
    plt.show()
    return img_dec


def compute_PSNR(nbr_step):
    ref = down_sample(skimage.io.imread("monkey.gif", as_grey=True), 4)
    psnr = list()
    init_region = 2
    init_domaine = 64
    for k in range(nbr_step):
        print("Compressing image with {} domaine size - {} region size".format(init_domaine//(2**k), init_region))
        transforms = compress(ref, init_region, init_domaine//(2**k))
        print("Decompressing image")
        img_dec = decompress(transforms, init_region, init_domaine//(2**k), ref.shape, 8)
        EQM_ = np.mean(np.sum(np.square(ref - img_dec)))
        psnr_ = (10/np.log(10)) * np.log(255**2/EQM_)
        psnr.append(psnr_)
    psnr_2 = list()
    init_region = 8
    for k in range(nbr_step):
        print("Compressing image with {} domaine size - {} region size".format(init_domaine//(2**2), init_region//(2**k)))
        transforms = compress(ref, init_region//(2**k), init_domaine//(2**2))
        print("Decompressing image")
        img_dec = decompress(transforms, init_region//(2**k), init_domaine//(2**2), ref.shape, 8)
        EQM_ = np.mean(np.sum(np.square(ref - img_dec)))
        psnr_2_ = (10/np.log(10)) * np.log(255**2/EQM_)
        psnr_2.append(psnr_2_)
    plt.figure()
    plt.plot([init_domaine//(2**k) for k in range(nbr_step)], psnr, marker='x', linestyle="--", label="psnr")
    plt.title("PSNR vs Domain Size")
    plt.xlabel("Domain Size")
    plt.ylabel("PSNR")
    plt.legend()
    plt.plot([init_domaine//(2**k) for k in range(nbr_step)], psnr_2, marker='x', linestyle="--", label="psnr", color="r")
    plt.savefig("error_rate_3.png")
    plt.show()
    return psnr


def compute_SSIM(nbr_step):
    ref = down_sample(skimage.io.imread("monkey.gif", as_grey=True), 4)
    print(ref.shape)
    SSIM = list()
    init_region = 4
    init_domaine = 64
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2
    c3 = c2/2
    for k in range(nbr_step):
        print("Compressing image with {} domaine size - {} region size".format(init_domaine//(2**k), init_region))
        transforms = compress(ref, init_region, init_domaine//(2**k))
        print("Decompressing image")
        img_dec = decompress(transforms, init_region, init_domaine//(2**k), ref.shape, 8)
        print(img_dec.shape)
        SSIM_ = list()
        for l in range(img_dec.shape[0] // 8):
            for j in range(img_dec.shape[0] // 8):
                x, y = ref[8*k:8*(k+1), 8*j:8*(j+1)], img_dec[8*k:8*(k+1), 8*j:8*(j+1)]
                mux = np.mean(x)
                muy = np.mean(y)
                sigx = np.std(x)
                sigy = np.std(y)
                cov = np.cov(x, y)
                l = (2*mux*muy + c1)/(mux**2 + muy**2 + c1)
                c = (2*sigx*sigy + c2)/(sigx**2 + sigy**2 + c2)
                s = (cov + c3)/(sigx*sigy + c3)
                
                SSIM_.append(l*c*s)
        SSIM.append(np.mean(SSIM_))
        print(len(SSIM_))
        print(np.mean(SSIM_))
        plt.imshow(img_dec, cmap="gray", interpolation=None)
        plt.savefig("monkey{}-{}.png".format(init_domaine//(2**k), init_region))
    plt.figure()
    plt.plot([init_domaine//(2**k) for k in range(nbr_step)], SSIM, marker='x', linestyle="--", label="SSIM")
    plt.xticks([init_domaine//(2**k) for k in range(nbr_step)])
    plt.title("SSIM vs Domain Size")
    plt.xlabel("Domain Size")
    plt.ylabel("SSIM")
    plt.legend()
    plt.savefig("error_SSIM5.png")
    plt.show()

compute_SSIM(4)
