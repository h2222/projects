# coding=utf-8
import os
import cv2
import time
import pickle
import numpy as np
from collections import Counter, OrderedDict
from multiprocessing import Pool, cpu_count
from max_area_land import *
from test_flat_img import show_in_one, show_in_one_v2


def is_png(p): return True if p.endswith('.png') else False


# time
def start(): return time.time()


def end(start): return int(round((time.time() - start) * 1000))


def get_sample(path, heigth_rate=0.4, width_rate=0.6):
    assert is_png(path)
    frame = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    l, w, _ = frame.shape
    origin_frame = frame[:int(l * heigth_rate), :int(w * width_rate)]
    gray_cut_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2GRAY)
    return origin_frame, gray_cut_frame


def dcolor_pool(frame, origin_frame, dcolor=3, pooler=5):
    gray_frame = frame // dcolor
    n_new = gray_frame.shape[0] - gray_frame.shape[0] % pooler
    m_new = gray_frame.shape[1] - gray_frame.shape[1] % pooler
    gray_frame = gray_frame[:n_new, :m_new]
    # second cut
    flat_frame = Counter(gray_frame.flatten())
    center = {}
    for c in flat_frame:
        h_list = []
        w_list = []
        h_ave = -1
        w_ave = -1
        for i in range(n_new):
            for j in range(m_new):
                if c == gray_frame[i, j]:
                    h_list.append(i)
                    w_list.append(j)
        h_ave = np.mean(h_list)
        w_ave = np.mean(w_list)
        center[c] = (h_ave, w_ave)
    # second cutting
    for c in center:
        h_ave = center[c][0]
        w_ave = center[c][1]
        af_hs = max(0, h_ave - n_new * 0.40)
        af_he = min(n_new, h_ave + n_new * 0.40)
        af_ws = max(0, w_ave - m_new * 0.40)
        af_we = min(m_new, w_ave + m_new * 0.40)
        atten_frame = gray_frame[int(af_hs):int(af_he), int(af_ws):int(af_we)]
        cut_frame = origin_frame[int(af_hs):int(af_he), int(af_ws):int(af_we)]
        af_h, af_w = atten_frame.shape
        n = af_h // pooler
        m = af_w // pooler
        pooled_frame = np.zeros((n, m))
        for i, i_index in zip(range(0, af_h, pooler), range(n)):
            for j, j_index in zip(range(0, af_w, pooler), range(m)):
                block = atten_frame[i:i + pooler, j:j + pooler]
                max_value = np.argmax(np.bincount(block.flatten()))
                pooled_frame[i_index, j_index] = max_value
        yield pooled_frame, c, cut_frame


def create_threading(
        img_path,
        root_path,
        heigth_rate,
        width_rate,
        dcolor,
        pooler,
        fuzz_ids,
        margin):
    save_path = ''
    result_mat_list = []
    result_sample_list = []
    img_code = img_path.split("\\")[-1].split(".")[0]
    ct = 0
    origin_frame, gray_cut_frame = get_sample(
        img_path, heigth_rate=heigth_rate, width_rate=width_rate)
    print('imgcode: {} origin_shape:{} gray_frame_shape:{}'.format(
        img_code, origin_frame.shape, gray_cut_frame.shape))
    for pooled_frame, c, cut_frame in dcolor_pool(
            gray_cut_frame, origin_frame, dcolor=dcolor, pooler=pooler):
        pooled_frame = pooled_frame.astype(np.uint8)
        n, m = pooled_frame.shape
        ct += 1
        # cv2.imshow(img_code, mat=cut_frame)
        # cv2.waitKey()
        try:
            bi_martrix = np.zeros((n, m), dtype=np.float32)
            for i in range(n):
                for j in range(m):
                    if (c + 1) >= pooled_frame[i, j] >= (c - 1):  # increase 'c' range c +/-1
                        bi_martrix[i, j] = 1.0
            input_matrix = get_candidate_rectangle(
                bi_martrix, fuzz_dis=fuzz_ids)
            # ma:int,   loc:((min_h, min_w), (max_h, max_w))
            ma, loc = maxAreaOfIsland_v2(input_matrix)
            xss, yss = loc[0]
            xee, yee = loc[1]
            if xss >= xee or yss >= yee:
                continue
            xs = max(xss * pooler - margin, 0)
            ys = max(yss * pooler - margin, 0)
            xe = min(xee * pooler + margin, cut_frame.shape[0] - 1)
            ye = min(yee * pooler + margin + 8, cut_frame.shape[1] - 1)
            save_path = os.path.join(root_path, img_code)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_file_path = os.path.join(
                save_path, img_code + '_' + str(ct) + '.png')
            result = cut_frame[int(xs):int(xe), int(ys):int(ye)].astype(np.uint8)
            print(result.shape)
            if min([result.shape[1], result.shape[0]]) > 0:
                # if max([result.shape[1], result.shape[0]]) / \
                # float(min([result.shape[1], result.shape[0]])) <= 3.5:
                result_sample = cv2.resize(result, (200, 200))
                result_sample_list.append(result_sample)
                result_mat_list.append((save_file_path, result))
        except Exception as e:
            print(e)
            continue
    return (result_mat_list, result_sample_list, save_path)


def main(dir_path, height_rate, width_rate, pooler, fuzz_ids, margin, dcolor):
    def comb_and_save(results):
        result_mat_list, images, save_path = results
        comb_file_path = os.path.join(save_path, '_comb.png')
        position_path = os.path.join(save_path, 'position.pkl')
        merge_img, position = show_in_one_v2(images)
        for path, img in result_mat_list:
            print("save one", path)
            cv2.imwrite(path, img)
        cv2.imwrite(comb_file_path, merge_img)
        position_pkl = open(position_path, 'wb')
        pickle.dump(position, position_pkl)

    root_path = 'test_result_pooler_{}_fuzz_dis_{}_mergin_{}_dcolor_{}'.format(
        pooler, fuzz_ids, margin, dcolor)
    f_list = []
    if os.path.isdir(root_path):
        os.system('rm -rf %s' % root_path)
        os.system('mkdir %s' % root_path)
    else:
        os.system('mkdir %s' % root_path)
    for f in os.listdir(dir_path):
        print(f)
        f_list.append(os.path.join(dir_path, f))

    process_num = max(1, int(cpu_count() * 0.75))
    print("process num: %d" % process_num)
    p = Pool(process_num)
    for i, f_path in enumerate(f_list[0:50]):
        print(i, f_path)
        p.apply_async(create_threading, args=(f_path,
                                              root_path,
                                              height_rate,
                                              width_rate,
                                              dcolor,
                                              pooler,
                                              fuzz_ids,
                                              margin),
                                        callback=comb_and_save)
    p.close()
    p.join()


# single processing test func
def test(img_path, height_rate, width_rate, pooler, fuzz_ids, margin, dcolor):
    def pick_and_save(results):
        save_path_img = show_in_one(results)
        if len(save_path_img) != 0:
            for path, img in save_path_img:
                print("save one", path)
                cv2.imwrite(path, img)

    root_path = 'test_result_pooler_{}_fuzz_dis_{}_mergin_{}_dcolor_{}'.format(
        pooler, fuzz_ids, margin, dcolor)
    f_list = []
    if os.path.isdir(root_path):
        os.system('rm -rf %s' % root_path)
        os.system('mkdir %s' % root_path)
    else:
        os.system('mkdir %s' % root_path)
    for f in os.listdir(dir_path):
        print(f)
        f_list.append(os.path.join(dir_path, f))
    for i, f_path in enumerate(f_list[0:10]):
        results = create_threading(f_path,
                         root_path,
                         height_rate,
                         width_rate,
                         dcolor,
                         pooler,
                         fuzz_ids,
                         margin)
        pick_and_save(results)


if __name__ == "__main__":
    pooler = 5
    height_rate = 0.3
    width_rate = 0.6
    fuzz_dis = 10
    margin = 60
    dcolor = 3
    dir_path = '../../../image_project/new_pic'
    # test(dir_path, height_rate, width_rate, pooler, fuzz_dis, margin, dcolor)
    # 点击截图模式下不能使用多进程 main
    main(dir_path, height_rate, width_rate, pooler, fuzz_dis, margin, dcolor)
