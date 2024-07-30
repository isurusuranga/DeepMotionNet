import os


def get_drr_image_dict(drr_img_paths):
    drr_img_dict = {}
    for i, path in enumerate(drr_img_paths):
        key_with_angle = os.path.basename(path).rsplit('.', 1)[0]
        # key hold the volumetric mesh name (i.e. volume-10) by removing the gantry angle (i.e. volume-10_72)
        key = os.path.basename(key_with_angle).rsplit('_', 1)[0]

        if key in drr_img_dict:
            drr_img_dict[key].append(path)
        else:
            drr_img_dict[key] = [path]

    return drr_img_dict
