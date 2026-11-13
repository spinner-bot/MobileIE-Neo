"""Read only official test-dev ZIP directory/text; never load image members."""
import hashlib
import json
from urllib.request import Request, urlopen
import zipfile

from dynamic_training_r34 import BASE, ROOT, digest
from inspect_davis_external import RemoteZip

URL = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip'


class SerialMetadataZip(RemoteZip):
    def __init__(self, url):
        super().__init__(url)
        self.ranges = []

    def read(self, count=-1):
        count = min(self.size-self.position, count if count >= 0 else self.size-self.position)
        assert count < 16*1024*1024, 'Metadata only; refuse unexpectedly large read'
        if count == 0:
            return b''
        start, end = self.position, self.position+count-1
        request = Request(self.url, headers={'Range': f'bytes={start}-{end}', 'If-Match': self.etag})
        with urlopen(request, timeout=30) as response:
            assert response.status == 206 and response.headers['Content-Range'] == f'bytes {start}-{end}/{self.size}'
            value = response.read()
        assert len(value) == count
        self.position += count
        self.transferred += count
        self.ranges.append([start, end])
        return value


def main():
    path = BASE/'data/external-davis/test-dev-metadata-reserve.json'
    assert not path.exists()
    reader = SerialMetadataZip(URL)
    with zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        image_infos = [i for i in archive.infolist() if '/JPEGImages/480p/' in i.filename and i.filename.endswith('.jpg')]
        scenes = sorted({i.filename.split('/480p/')[1].split('/')[0] for i in image_infos})
        texts = {}
        for name in names:
            if '/ImageSets/2017/' in name and name.endswith('.txt'):
                value = archive.read(name)  # The only ZIP payloads read are split text.
                texts[name] = {'sha256': hashlib.sha256(value).hexdigest(), 'videos': sorted(value.decode().split())}
        counts = {scene: min(64, sum('/480p/'+scene+'/' in i.filename for i in image_infos)) for scene in scenes}
    prior_path = BASE/'data/external-davis/fastdvdnet-training-overlap.json'
    prior = json.loads(prior_path.read_text())
    selection = json.loads((BASE/'data/exploration-r32-training/selection.json').read_text())
    assert len(scenes) == 30 and min(counts.values()) >= 4
    assert not set(scenes) & set(prior['documented_training_videos'])
    assert not set(scenes) & (set(selection['training_videos']) | set(selection['excluded_development_videos']))
    value = {'archive_url': URL, 'archive_bytes': reader.size, 'etag': reader.etag, 'last_modified': reader.last_modified,
             'metadata_bytes_transferred': reader.transferred, 'metadata_ranges': reader.ranges, 'split_text': texts,
             'reserved_video_ids': scenes, 'video_count': len(scenes), 'future_first_min64_frame_counts': counts,
             'future_total_frames': sum(counts.values()), 'prior_trainlist_sha256': digest(prior_path),
             'policy_sha256': digest(BASE/'docs/heldout-video-source-policy.md'),
             'overlap_project_train_and_DEV': [], 'overlap_documented_FastDVDnet_training': [],
             'JPEG_members_read': 0, 'decoded_images': 0, 'model_inferences': 0,
             'scope': 'Segmentation-source metadata reserve only; no final pixels, quality results, official LLVE test or guarantee about undocumented pretraining/semantic duplication.'}
    path.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: value[k] for k in ['video_count', 'future_total_frames', 'metadata_bytes_transferred', 'JPEG_members_read']}), flush=True)


if __name__ == '__main__':
    main()
