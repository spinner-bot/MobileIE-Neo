"""把 flicker_baseline.py 导出的帧序列合成 mp4（独立进程，避免与 torch/CUDA 争内存）。

用法：python research/3.video-flicker/scripts/make_videos.py [-j results/video_jobs.json]
"""
import argparse
import json
import os
import shutil
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', '--jobs', default=os.path.join(os.path.dirname(__file__), '..', 'results', 'video_jobs.json'))
    args = ap.parse_args()

    with open(args.jobs, encoding='utf-8') as f:
        jobs = json.load(f)

    for kind, filename in jobs['files'].items():
        tmp = 'F:/pip-tmp/' + filename  # ffmpeg 写不了中文路径
        r = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(jobs['fps']),
             '-i', os.path.join(jobs['frame_dir'], kind, '%04d.jpg'),
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-threads', '1', tmp],
            capture_output=True, text=True, errors='replace')
        if r.returncode != 0:
            raise RuntimeError('ffmpeg failed (%s): %s' % (r.returncode, r.stderr[-600:]))
        shutil.move(tmp, os.path.join(jobs['out_dir'], filename))
        print('wrote', filename)

    # 三联对比视频：输入 | MobileIE 输出 | GT
    out = jobs['out_dir']
    tmp_cmp = 'F:/pip-tmp/compare_side_by_side.mp4'
    r = subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error',
         '-i', os.path.join(out, jobs['files']['input']),
         '-i', os.path.join(out, jobs['files']['output']),
         '-i', os.path.join(out, jobs['files']['ground_truth']),
         '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', '-threads', '1', tmp_cmp],
        capture_output=True, text=True, errors='replace')
    if r.returncode != 0:
        raise RuntimeError('ffmpeg compare failed (%s): %s' % (r.returncode, r.stderr[-600:]))
    shutil.move(tmp_cmp, os.path.join(out, 'compare_side_by_side.mp4'))
    print('wrote compare_side_by_side.mp4')

    shutil.rmtree(jobs['frame_dir'], ignore_errors=True)


if __name__ == '__main__':
    main()
