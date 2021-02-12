import mesh2tex.utils.FID.fid_score as FID
import mesh2tex.utils.FID.feature_l1 as feature_l1
import mesh2tex.utils.SSIM_L1.ssim_l1_score as SSIM
import os
import sys


if __name__ == '__main__':

    # gt = f"{os.environ['nihalsid']}/baseline_renders/{sys.argv[1]}"
    # tf = f"{os.environ['nihalsid']}/baseline_renders/{sys.argv[2]}"

    data_file = os.path.join(os.getcwd(), "data/test.lst")
    gt = os.path.join(os.getcwd(), "data/rendered_images/gt")
    tf = os.path.join(os.getcwd(), "data/rendered_images/tf")

    print(SSIM.calculate_ssim_l1_given_paths([gt, tf], subfolder_mode=False)) # (0.8873921200526393, 0.03657292955186642)
    print(feature_l1.calculate_feature_l1_given_paths([gt, tf], 1, True, 2048, subfolder_mode=False)) # 0.224288380815339
    print(FID.calculate_fid_given_paths([gt, tf], 1, True, 2048, subfolder_mode=False)) # 61.92403253977176
