import argparse

from defocus_estimate import *
from graph_draw import *

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
    np.save(args['image']+'_edge_map.npy', edge_map)    

    sparse_bmap = estimate_sparse_blur(gimg, edge_map, std1 = 1, std2 = 1.5)
    h, w = sparse_bmap.shape
    print("Sparse blur map generated")
    print("w:", w, "h:", h)
    cv2.imwrite(args['image'] + '_sparse_bmap.png', np.uint8((sparse_bmap / sparse_bmap.max()) * 255))
    print("Defocus map saved as: ", args['image'] + '_sparse_bmap.png')
    np.save(args['image'] + '_sparse_bmap.npy', sparse_bmap)
    print("Defocus map saved as: ", args['image'] + '_sparse_bmap.npy')

    # JBF filtered defocus map estimation
    filtered_bmap = jbf_filtered(sparse_bmap, edge_map, gimg, d=3, sigma_color=-1, sigma_space=-1)
    print("JBF filtered defocus map estimation completed.")
    cv2.imwrite(args['image'] + '_filtered_bmap.png', np.uint8((filtered_bmap / filtered_bmap.max()) * 255))
    print("JBF filtered defocus map saved as: ", args['image'] + '_filtered_bmap.png')
    np.save(args['image'] + '_filtered_bmap.npy', filtered_bmap)

    # angle_mask 生成
    # h,w = gimg.shape
    # i_diff = np.full((h,w),-1)
    # half_h = h // 2
    # half_w = w // 2
    # # 某一行某部分赋值为0
    # i_diff[0, half_w:] = 0
    # # 左下角区域赋值为1
    # i_diff[half_h:, :half_w] = 1
    # angle_mask = generate_angle_mask(i_diff, gimg, threshold=0.1)
    # show_cool_warm_photo(angle_mask, title='Angle Mask', if_save=True, save_path=args['image'] + '_angle_mask.png')
    # print("Angle mask saved as: ", args['image'] + '_angle_mask.png')
    # np.save(args['image'] + '_angle_mask.npy', angle_mask)
    # print("Angle mask saved as: ", args['image'] + '_angle_mask.npy')
    angle_mask = np.full((h,w),1)
    angle_mask[:h//2, :w//2] = -1
    show_cool_warm_photo(angle_mask, title='Angle Mask', if_save=True, save_path=args['image'] + '_angle_mask.png')
    print("Angle mask saved as: ", args['image'] + '_angle_mask.png')
    np.save(args['image'] + '_angle_mask.npy', angle_mask)
    print("Angle mask saved as: ", args['image'] + '_angle_mask.npy')

    # imambiguous_sparse_bmap 生成
    imambiguous_sparse_bmap = generate_imambiguous_sparse_map(sparse_bmap, angle_mask, edge_map)
    show_cool_warm_photo(imambiguous_sparse_bmap, title='Imambiguous Sparse Map', if_save=True, save_path=args['image'] + '_imambiguous_sparse_map.png')
    print("Imambiguous sparse map saved as: ", args['image'] + '_imambiguous_sparse_map.png')
    np.save(args['image'] + '_imambiguous_sparse_map.npy', imambiguous_sparse_bmap)
    print("Imambiguous sparse map saved as: ", args['image'] + '_imambiguous_sparse_map.npy')
    mi = np.min(imambiguous_sparse_bmap)
    ma = np.max(imambiguous_sparse_bmap)
    imambiguous_sparse_bmap[imambiguous_sparse_bmap<0] = imambiguous_sparse_bmap[imambiguous_sparse_bmap<0]-mi+ma
    imambiguous_sparse_bmap = (imambiguous_sparse_bmap)/2 # 规范到一个大于0的区间里面
    show_cool_warm_photo(imambiguous_sparse_bmap, title='Imambiguous Sparse Map', if_save=True, save_path=args['image'] + '_imambiguous_sparse_map_norm.png')
    print("Imambiguous sparse map saved as: ", args['image'] + '_imambiguous_sparse_map_norm.png')

    # Defocus map estimation
    # fblurmap = estimate_bmap_laplacian(img, sigma_c = 1, std1 = 1, std2 = 1.5)
    # print("Defocus map estimation completed.")
    # cv2.imwrite(args['image'] + '_bmap.png', np.uint8((fblurmap / fblurmap.max()) * 255))
    # print("Defocus map saved as: ", args['image'] + '_bmap.png')
    # np.save(args['image'] + '_bmap.npy', fblurmap)
    # print("Defocus map saved as: ", args['image'] + '_bmap.npy')

    # Non-ambiguous defocus map estimation 
    L1 = get_laplacian(img / 255.0) 
    A, b = make_system(L1, imambiguous_sparse_bmap.T)   
    print("System generated")
    print(f"A的维度: {A.shape}")       # 查看N的大小（N×N）
    print(f"A的非零元素数: {A.nnz}")  # 非零元素越多，计算越慢
    fblurmap = scipy.sparse.linalg.spsolve(A, b).reshape(w, h).T  
    fblurmap = fblurmap*2
    fblurmap[fblurmap>=ma] = fblurmap[fblurmap>=ma]-ma+mi  # 还原到原来的区间
    show_cool_warm_photo(fblurmap, title='Non-ambiguous Defocus Map', if_save=True, save_path=args['image'] + '_non_ambiguous_bmap.png')
    print("Defocus map saved as: ", args['image'] + '_non_ambiguous_bmap.png')
    np.save(args['image'] + 'non_ambiguous_bmap.npy', fblurmap)
    print("Defocus map saved as: ", args['image'] + 'non_ambiguous_bmap.npy')
