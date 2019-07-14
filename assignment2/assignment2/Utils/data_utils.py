from scipy.ndimage.filters import gaussian_filter1d
from PIL import Image
from six.moves import urllib
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import tarfile
import zipfile
import os
import sys
from glob import glob
import re
import random
import tensorflow as tf

#Global variable for image normalization
CIFAR10_MEAN = None 
CIFAR10_STD = None
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_images():
    ''' Load sample images '''
    _kitten, _puppy = Image.open('./Utils/kitten.jpg'), Image.open('./Utils/puppy.jpg')
    kitten, puppy = np.array(_kitten), np.array(_puppy)
    # kitten is wide, and puppy is already square
    d = kitten.shape[1] - kitten.shape[0]
    kitten_cropped = kitten[:, d//2:-d//2, :]
    _kitten_cropped = Image.fromarray(kitten_cropped.astype('uint8'),'RGB')

    img_size = 200   # Make this smaller if it runs too slow
    x = np.zeros((2, img_size, img_size, 3))
    x[0, :, :, :] = np.array(_puppy.resize((img_size, img_size)))
    x[1, :, :, :] = np.array(_kitten_cropped.resize((img_size, img_size)))
    return x

def _normalize_image(img):
    img_max, img_min = np.max(img), np.min(img)
    img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype('uint8')

def plot_conv_images(x, out):
    ''' Plot original and convolution output images '''
    plt.subplot(2, 3, 1)
    plt.imshow(_normalize_image(x[0]))
    plt.title('Original image')
    plt.axis('off')        
    
    plt.subplot(2, 3, 2)
    plt.imshow(_normalize_image(out[0,:,:,0]))
    plt.title('Grayscale')
    plt.axis('off')        
    
    plt.subplot(2, 3, 3)
    plt.imshow(_normalize_image(out[0,:,:,1]))
    plt.title('Edges')
    plt.axis('off')        
    
    plt.subplot(2, 3, 4)
    plt.imshow(_normalize_image(x[1]))
    plt.axis('off')        
    
    plt.subplot(2, 3, 5)
    plt.imshow(_normalize_image(out[1,:,:,0]))
    plt.axis('off')        
    
    plt.subplot(2, 3, 6)
    plt.imshow(_normalize_image(out[1,:,:,1]))
    plt.axis('off')        
    
def maybe_download_and_extract(DATA_URL):
    ''' Download and extract the data if it doesn't already exist. '''
    dest_directory = './Utils'
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        if '.tar.gz' in filepath:
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        if '.zip' in filepath:
            zipfile.ZipFile(filepath).extractall(dest_directory)  
        print("Successfully downloaded and unpacked")
    else:
        print("Data has already been downloaded and unpacked.")

def _unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

def _preprocess_image(img, MEAN=IMAGENET_MEAN, STD=IMAGENET_STD):
    """ Preprocess an image: subtracts the pixel mean and divides by the standard deviation.  """
    return (img.astype(np.float32)/255.0 - MEAN) / STD

def _deprocess_image(img, MEAN=IMAGENET_MEAN, STD=IMAGENET_STD):
    """ Undo preprocessing on an image and convert back to uint8. """
    return np.clip(255 * (img * STD + MEAN), 0.0, 255.0).astype(np.uint8)

def load_CIFAR10(val_batch=[5]):
    ''' Load CIFAR10 dataset '''
    maybe_download_and_extract('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    X_train, Y_train, X_val, Y_val = [], [], [], []
    for i in range(1,6):
        data = _unpickle("./Utils/cifar-10-batches-py/data_batch_%d" % i)
        if i not in val_batch:
            X_train.append(data['data'])
            Y_train.append(data['labels'])
        else:
            X_val.append(data['data'])
            Y_val.append(data['labels']) 
    test = _unpickle('./Utils/cifar-10-batches-py/test_batch')
    
    X_train = np.concatenate(X_train, axis=0).reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
    Y_train = np.concatenate(Y_train, axis=0) 
    X_val = np.concatenate(X_val, axis=0).reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
    Y_val = np.concatenate(Y_val, axis=0) 
    X_test = test['data'].reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])
    Y_test = np.array(test['labels'])
    
    #Normalize input images
    global CIFAR10_MEAN, CIFAR10_STD
    CIFAR10_MEAN = np.mean(X_train.astype(np.float32)/255.0, axis=0)
    CIFAR10_STD = np.std(X_train.astype(np.float32)/255.0, axis=0)
    X_train = np.array([_preprocess_image(img, CIFAR10_MEAN, CIFAR10_STD) for img in X_train])
    X_val   = np.array([_preprocess_image(img, CIFAR10_MEAN, CIFAR10_STD) for img in X_val])
    X_test  = np.array([_preprocess_image(img, CIFAR10_MEAN, CIFAR10_STD) for img in X_test])
    
    # Load the class-names
    Class_names = _unpickle("./Utils/cifar-10-batches-py/batches.meta")['label_names']
        
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, Class_names

def load_kitti_road(image_shape):
    """
    Download Kitti Road Dataset, Generate function to create batches of training data
    image_shape : Tuple - Shape of image
    """  
    
    '''Download Kitti Road Dataset from the source'''
    maybe_download_and_extract('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip')
    
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        data_folder = os.path.join('.', 'Utils', 'data_road', 'training')
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        road_color = np.array([255, 0, 255])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = Image.open(image_file)
                image = np.array(image.resize(image_shape))
                image = np.transpose(image,(1,0,2))
                gt_image = Image.open(gt_image_file)
                gt_image = np.array(gt_image.resize(image_shape))
                gt_image = np.transpose(gt_image,(1,0,2))

                gt_bg = np.all(gt_image == road_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((np.invert(gt_bg), gt_bg), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def gen_test_output(sess, logits, is_training, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
#        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        image = Image.open(image_file)
        image = np.array(image.resize(image_shape))# scipy.misc.imresize(imageio.imread(image_file), image_shape)
        image = np.transpose(image,(1,0,2))
#
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {is_training: False, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = Image.fromarray(mask.astype('uint8'),'RGBA')
        street_im = Image.fromarray(image.astype('uint8'),'RGB')
        street_im.paste(mask, box=None, mask=mask)

#        yield os.path.basename(image_file), np.array(street_im)
        yield os.path.basename(image_file), street_im

def save_test_samples(output_dir, sess, image_shape, logits, is_training, input_image):
    # Run NN on test images and save them
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, is_training, input_image, os.path.join('Utils', 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        image.transpose(Image.ROTATE_270).save(os.path.join(output_dir, name))

def load_ImageNet_val(num=None):
    ''' Load a handful of validation images from ImageNet '''
    maybe_download_and_extract('http://cs231n.stanford.edu/imagenet_val_25.npz')
    
    f = np.load('./Utils/imagenet_val_25.npz')
    X = f['X']
    Y= f['y']
    Class_names = f['label_map'].item()
    if num is not None:
        X = X[:num]
        Y = Y[:num]
      
    #Normalize input images
    global IMAGENET_MEAN, IMAGENET_STD
    X = np.array([_preprocess_image(img, IMAGENET_MEAN, IMAGENET_STD) for img in X])
    Class_names = {i:c.split(',')[0] for i, c in Class_names.items()}
    
    return X, Y, Class_names
    
    
def plot_images(X, Y, C, idx=0, Each_Category=False, SaliencyMaps=None, ClassRepresentatve=None, Adversarial=None, Target_y=None):
    ''' Plot images '''
    if Each_Category:
        Category = set(Y)
        for i in range(10):
            while(1):
                if Y[idx] in Category:
                    Category.remove(Y[idx])
                    break
                else:
                    idx += 1
            
            plt.subplot(2, 5, Y[idx]+1)
            plt.imshow(_deprocess_image(X[idx], CIFAR10_MEAN, CIFAR10_STD))
            plt.title(C[Y[idx]])
            plt.axis('off')
            
    elif SaliencyMaps is not None:    
        for i in range(5):
            plt.subplot(2, 5, i+1)
            plt.imshow(_deprocess_image(X[idx+i], IMAGENET_MEAN, IMAGENET_STD))
            plt.title(C[Y[idx+i]])
            plt.axis('off')
            
            plt.subplot(2, 5, i+6)
            plt.imshow(SaliencyMaps[idx+i], cmap=plt.cm.hot)
            plt.title(C[Y[idx+i]])
            plt.axis('off')
            
    elif ClassRepresentatve is not None:    
        Iter = int(X.shape[0] / 4)
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(_deprocess_image(X[Iter*(i+1)], IMAGENET_MEAN, IMAGENET_STD))
            plt.title('%s\nIteration %d' % (C[Y], Iter*(i+1)))
            plt.axis('off')
            plt.gcf().set_size_inches(8, 8)
            
    elif Adversarial is not None:
        plt.subplot(1, 4, 1)
        plt.imshow(_deprocess_image(X[0]))
        plt.title(C[Y[0]])
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(_deprocess_image(Adversarial[0]))
        plt.title(C[Target_y])
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title('Difference')
        plt.imshow(_deprocess_image((X-Adversarial)[0]))
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title('Magnified difference (10x)')
        plt.imshow(_deprocess_image(10 * (X-Adversarial)[0]))
        plt.axis('off')          
        
    else:
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(_deprocess_image(X[idx+i], IMAGENET_MEAN, IMAGENET_STD))
            plt.title(C[Y[idx+i]])
            plt.axis('off')
    plt.show()
   
def random_noise_image(num_iterations=100):
    X = 255 * np.random.rand(num_iterations, 224, 224, 3)
    X = _preprocess_image(X)
    return X

def jitter_image(X, ox, oy):
    Xi = np.roll(np.roll(X, ox, 1), oy, 2)
    return Xi

def unjitter_image(X, ox, oy):
    Xi = np.roll(np.roll(X, -ox, 1), -oy, 2)
    Xi = np.clip(Xi, -IMAGENET_MEAN/IMAGENET_STD, (1.0 - IMAGENET_MEAN)/IMAGENET_STD)
    return Xi

def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X