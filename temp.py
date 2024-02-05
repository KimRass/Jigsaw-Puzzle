def get_sorted_dists(patches):
    dists = dict()
    for idx1 in patches:
        patch1 = patches[idx1]
        for idx2 in patches:
            patch2 = patches[idx2]
            if idx1 >= idx2:
                continue

            for edge_idx1 in range(4):
                for edge_idx2 in range(4):
                    for flip_idx in range(2):
                        edge1 = edge_idx_to_edge(patch1, edge_idx1)
                        edge2 = edge_idx_to_edge(patch2, edge_idx2)
                        if edge1.shape != edge2.shape:
                            continue

                        if flip_idx == 0:
                            dist = get_l2_dist(edge1, edge2)
                        else:
                            edge_h, edge_w, _ = edge2.shape
                            if edge_h >= edge_w:
                                dist = get_l2_dist(edge1, vflip(edge2))
                            else:
                                dist = get_l2_dist(edge1, hflip(edge2))
                        
                        dists[(idx1, idx2, edge_idx1, edge_idx2, flip_idx)] = dist
    
    sorted_dists = dict(sorted(dists.items(), key=lambda item: item[1]))
    return sorted_dists


breaker = False
    for idx1 in patches:
        if breaker:
            break

        dists = dict()
        patch1 = patches[idx1]
        for idx2 in patches:
            patch2 = patches[idx2]
            if idx1 >= idx2:
                continue

            for edge_idx1 in range(4):
                edge1 = edge_idx_to_edge(patch1, edge_idx1)
                for edge_idx2 in range(4):
                    edge2 = edge_idx_to_edge(patch2, edge_idx2)
                    if edge1.shape != edge2.shape:
                        continue

                    for flip_idx in range(2):
                        if flip_idx == 0:
                            dist = get_l2_dist(edge1, edge2)
                        else:
                            edge_h, edge_w, _ = edge2.shape
                            if edge_h >= edge_w:
                                dist = get_l2_dist(edge1, vflip(edge2))
                            else:
                                dist = get_l2_dist(edge1, hflip(edge2))
                        if (idx2 not in dists) or (idx2 in dists and dist < dists[idx2][3]):
                            dists[idx2] = (edge_idx1, edge_idx2, flip_idx, dist)
        sorted_dists = dict(sorted(dists.items(), key=lambda item: item[1][3]))

        cnt = 0
        for idx2, (edge_idx1, edge_idx2, flip_idx, dist) in sorted_dists.items():
            cnt += 1
            if cnt > 2:
                break
            # if global_cnt >= n_row_splits * n_col_splits:
            if (arr1 != 255).sum() >= n_row_splits * n_col_splits:
                breaker = True
                break

            if not get_coord_of_idx(arr1, idx1) and not get_coord_of_idx(arr1, idx2):
                fill_arr(arr1, coord, idx1)
                fill_arr(arr2, coord, dist)
                # global_cnt += 1
            coord1 = get_coord_of_idx(arr1, idx1)
            if coord1 and not get_coord_of_idx(arr1, idx2):
                coord = deepcopy(coord1)
                change_coord(coord, edge_idx1)
                fill_arr(arr1, coord, idx2)
                # global_cnt += 1
            elif not get_coord_of_idx(arr1, idx1):
                coord = get_coord_of_idx(arr1, idx2)
                change_coord(coord, edge_idx2)
                fill_arr(arr1, coord, idx1)
                # global_cnt += 1

            # merged = merge_two_patches(patches, idx1, idx2, edge_idx1, edge_idx2, flip_idx)
            print(idx1, idx2, edge_idx1, edge_idx2, flip_idx)
            print(arr1)
    # arr1
            # show_image(merged)


    # cnt = 0
    first = True
    for (patch_idx1, patch_idx2), (edge_idx1, edge_idx2, flip_idx, dist) in sorted_dists.items():

        if (idx_arr != 255).sum() >= n_row_splits * n_col_splits:
            break

        if all(
            [
                not get_coord_of_idx(idx_arr, patch_idx1),
                not get_coord_of_idx(idx_arr, patch_idx2)
            ]
        ):
            if first:
                fill_arr(idx_arr, coord, patch_idx1)
                fill_arr(dist_arr, coord, dist)
                first = False
            else:
                print(patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx, dist)
                continue
            # global_cnt += 1

        coord1 = get_coord_of_idx(idx_arr, patch_idx1)
        if coord1 and not get_coord_of_idx(idx_arr, patch_idx2):
            coord = deepcopy(coord1)
            change_coord(coord, edge_idx1)
            fill_arr(idx_arr, coord, patch_idx2)
            # global_cnt += 1
        elif not get_coord_of_idx(idx_arr, patch_idx1):
            coord = get_coord_of_idx(idx_arr, patch_idx2)
            change_coord(coord, edge_idx2)
            fill_arr(idx_arr, coord, patch_idx1)
            # global_cnt += 1

        print(patch_idx1, patch_idx2, edge_idx1, edge_idx2, flip_idx)
        print(idx_arr)
        # merged = merge_two_patches(patches, idx1, idx2, edge_idx1, edge_idx2, flip_idx)


def get_all_variants(img):
    imgs = list()
    imgs.extend([img, hflip(img), vflip(img), vflip(hflip(img))])
    img = rotate(img)
    imgs.extend([img, hflip(img), vflip(img), vflip(hflip(img))])
    return imgs