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
    args = {'image': r"C:\Users\lenovo\Desktop\ASP\Distance_Question\dfe-pr2010\input3.png"}
    print("The input image is: ", args['image'])
    print("Defocus map estimation started...")
    
    img = cv2.imread(args['image'])
    rows, cols, ch = img.shape
    print("Input image loaded.")
    print("Input image shape: ", rows, cols, ch)
    cv2.imshow('Input image', img)
    cv2.waitKey(0) # wait for key press
    cv2.destroyAllWindows() # close all windows
    
    # Defocus Sparse Blur map estimation 
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0    #
    edge_map = feature.canny(gimg, 1)
    print("Edge map generated")
    cv2.imwrite(args['image'] + '_edge_map.png', np.uint8(edge_map * 255))
    print("Edge map saved as: ", args['image'] + '_edge_map.png')

    sparse_bmap = estimate_sparse_blur(gimg, edge_map, std1 = 1, std2 = 1.5)
    h, w = sparse_bmap.shape
    print("Sparse blur map generated")
    print("w:", w, "h:", h)
    cv2.imwrite(args['image'] + '_sparse_bmap.png', np.uint8((sparse_bmap / sparse_bmap.max()) * 255))
    print("Defocus map saved as: ", args['image'] + '_sparse_bmap.png')
    np.save(args['image'] + '_sparse_bmap.npy', sparse_bmap)
    print("Defocus map saved as: ", args['image'] + '_sparse_bmap.npy')



    # Defocus map estimation
    fblurmap = estimate_bmap_laplacian(img, sigma_c = 1, std1 = 1, std2 = 1.5)
    print("Defocus map estimation completed.")
    cv2.imwrite(args['image'] + '_bmap.png', np.uint8((fblurmap / fblurmap.max()) * 255))
    print("Defocus map saved as: ", args['image'] + '_bmap.png')
    np.save(args['image'] + '_bmap.npy', fblurmap)
    print("Defocus map saved as: ", args['image'] + '_bmap.npy')


