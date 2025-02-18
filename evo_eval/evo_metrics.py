import subprocess
import os

base_path = "evo_eval/results"
gt_ext = '_gt.txt'
estimated_ext = '_est.txt'

gt_files = [file for file in os.listdir(base_path) if file.endswith(gt_ext)]
est_files = [file for file in os.listdir(base_path) if file.endswith(estimated_ext)]

print(gt_files)
print(est_files)

def trim_gt_file(gt_file_path, est_file_path):
    with open(gt_file_path, 'r') as gt_file:
        gt_lines = gt_file.readlines()
    with open(est_file_path, 'r') as est_file:
        est_lines = est_file.readlines()

    if len(gt_lines) > len(est_lines):
        trimmed_gt_path = gt_file_path.replace("_gt.txt", "_gt_trimmed.txt")
        with open(trimmed_gt_path, 'w') as trimmed_gt_file:
            trimmed_gt_file.writelines(gt_lines[:len(est_lines)])
        print(f"Trimmed {gt_file_path} to {len(est_lines)} poses. Saved to {trimmed_gt_path}")
        return trimmed_gt_path
    elif len(gt_lines) < len(est_lines):
        print(f"Error: Estimated file {est_file_path} has more poses than ground truth!")
        return None
    else:
        # No trimming needed
        return gt_file_path

for gt_file in gt_files:
    est_file = gt_file.split('_')[0] + estimated_ext
    if est_file in est_files:
        gt_path = os.path.join(base_path, gt_file)
        est_path = os.path.join(base_path, est_file)

        # Trim ground truth if needed
        trimmed_gt_path = trim_gt_file(gt_path, est_path)
        if not trimmed_gt_path:
            continue  # Skip this file if trimming failed

        # EVO - Plot trajectory
        command = f"evo_traj kitti {est_path} {trimmed_gt_path} -p --plot_mode=xz"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
        
        # EVO APE - Plot and save metrics (Absolute Pose Error - APE)
        command = f"evo_ape kitti {est_path} {trimmed_gt_path} -va --plot --plot_mode xz --save_results evo_eval/results/{gt_file.split('_')[0]}.zip"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")

# For RPE
for gt_file in gt_files:
    est_file = gt_file.split('_')[0] + estimated_ext
    if est_file in est_files:
        gt_path = os.path.join(base_path, gt_file)
        est_path = os.path.join(base_path, est_file)

        # Trim ground truth if needed
        trimmed_gt_path = trim_gt_file(gt_path, est_path)
        if not trimmed_gt_path:
            continue  # Skip this file if trimming failed

        # EVO - Plot and save metrics (RPE - Relative Pose Error)
        command = f"evo_rpe kitti {trimmed_gt_path} {est_path} --delta 5 --delta_unit f --plot --pose_relation trans_part"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
