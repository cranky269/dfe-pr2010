import argparse

from defocus_estimate import *


def get_args():
    '''

    :return: dictionary of arguments
    '''
    parser = argparse.ArgumentParser(description='Defocus map estimation from a single image, '
                                                 'S. Zhuo, T. Sim - Pattern Recognition, 2011 - Elsevier \n')

    parser.add_argument('-i', metavar='--image', required=True,
                        type=str, help='Defocused image \n')

    args = parser.parse_args()
    image = args.i
    print('Image: ', image)
    return {'image': image}


if __name__ == '__main__':

    #args = get_args()
    args = {'image': r"C:\Users\lenovo\Desktop\ASP\Distance_Question\dfe-pr2010\input0.png"}
    print("The input image is: ", args['image'])
    print("Defocus map estimation started...")
    
    img = cv2.imread(args['image'])
    print("Input image loaded.")
    cv2.imshow('Input image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    fblurmap = estimate_bmap_laplacian(img, sigma_c = 1, std1 = 1, std2 = 1.5)
    print("Defocus map estimation completed.")
    cv2.imwrite(args['image'] + '_bmap.png', np.uint8((fblurmap / fblurmap.max()) * 255))
    print("Defocus map saved as: ", args['image'] + '_bmap.png')
    np.save(args['image'] + '_bmap.npy', fblurmap)
    print("Defocus map saved as: ", args['image'] + '_bmap.npy')


