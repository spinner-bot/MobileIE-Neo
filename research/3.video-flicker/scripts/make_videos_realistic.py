"""把 flicker_realistic.py 导出的帧序列合成视频（独立进程，避免与 torch 争内存）。"""
import json
import os
import shutil
import subprocess

JOBS = os.path.join(os.path.dirname(__file__), '..', 'results_realistic', 'video_jobs.json')


def ff(args):
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + args,
                       capture_output=True, text=True, errors='replace')
    if r.returncode != 0:
        raise RuntimeError('ffmpeg failed: %s' % r.stderr[-500:])


def main():
    jobs = json.load(open(JOBS, encoding='utf-8'))
    frame_dir, out = jobs['frame_dir'], jobs['out_dir']
    for kind, name in jobs['files'].items():
        tmp = 'F:/pip-tmp/' + name
        ff(['-framerate', str(jobs['fps']), '-i', os.path.join(frame_dir, kind, '%04d.jpg'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-threads', '1', tmp])
        shutil.move(tmp, os.path.join(out, name))
        print('wrote', name)

    tmp = 'F:/pip-tmp/compare_real.mp4'
    ff(['-i', os.path.join(out, jobs['files']['input']),
        '-i', os.path.join(out, jobs['files']['output']),
        '-i', os.path.join(out, jobs['files']['ground_truth']),
        '-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', '-threads', '1', tmp])
    shutil.move(tmp, os.path.join(out, 'compare_side_by_side.mp4'))
    print('wrote compare_side_by_side.mp4')

    tmp = 'F:/pip-tmp/compare_input_output.mp4'
    ff(['-i', os.path.join(out, jobs['files']['input']),
        '-i', os.path.join(out, jobs['files']['output']),
        '-filter_complex', '[0:v][1:v]hstack=inputs=2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', '-threads', '1', tmp])
    shutil.move(tmp, os.path.join(out, 'compare_input_output.mp4'))
    print('wrote compare_input_output.mp4')

    shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
