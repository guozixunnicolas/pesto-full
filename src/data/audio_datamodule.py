import hashlib
import itertools
import json
import logging
from pathlib import Path
from typing import Sequence, Tuple, Any
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data

import torchaudio
from lightning import LightningDataModule

from src.data.hcqt import HarmonicCQT
from nnAudio.features import STFT
import torchaudio.transforms as T
import torch.nn.functional as F

log = logging.getLogger(__name__)


def hz_to_mid(freqs):
    return np.where(freqs > 0, 12 * np.log2(freqs / 440) + 69, 0)
def low_pass_filter(waveform, cutoff_freq, sample_rate, filter_length=1001):
    # Compute the normalized cutoff frequency
    nyquist_rate = sample_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist_rate

    # Create a sinc filter
    t = torch.arange(-(filter_length // 2), (filter_length // 2) + 1, dtype=torch.float32)
    sinc_filter = torch.sin(2 * torch.pi * normalized_cutoff * t) / (torch.pi * t)
    sinc_filter[filter_length // 2] = 2 * normalized_cutoff  # Avoid division by zero
    
    # Apply a window function (e.g., Hamming window)
    window = torch.hamming_window(filter_length)
    sinc_filter *= window
    
    # Normalize the filter to ensure the gain at DC is 1
    sinc_filter /= sinc_filter.sum()
    
    # Apply the filter using convolution (in the time domain)
    filtered_waveform = F.conv1d(waveform.unsqueeze(1), sinc_filter.unsqueeze(0).unsqueeze(1))
    
    return filtered_waveform.squeeze(1)

def resample_waveform(waveform, original_sr, target_sr, lowpass_filter_width=6, rolloff=0.99, resampling_method='sinc_interpolation'):
    # Create a Resample transform with anti-aliasing
    resampler = torchaudio.transforms.Resample(
        orig_freq=original_sr,
        new_freq=target_sr,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        resampling_method=resampling_method
    )
    
    # Resample the waveform
    resampled_waveform = resampler(waveform)
    
    return resampled_waveform


class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels=None, filter_unvoiced: bool = False) -> None:
        assert labels is None or len(inputs) == len(labels), "Lengths of inputs and labels do not match"
        if filter_unvoiced and labels is None:
            log.warning("Cannnot filter out unvoiced frames without annotations.")
            filter_unvoiced = False
        if filter_unvoiced:
            self.inputs = inputs[labels > 0]
            self.labels = labels[labels > 0]
        else:
            self.inputs = inputs
            self.labels = labels

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[item] if self.labels is not None else 0
        return torch.view_as_complex(torch.from_numpy(self.inputs[item])), label

    def __len__(self):
        return len(self.inputs)


class AudioDataModule(LightningDataModule):
    def __init__(self,
                 audio_files: str,
                 annot_files: str | None = None,
                 val_audio_files: str | None = None,
                 val_annot_files: str | None = None,
                 harmonics: Sequence[float] = (1,),
                 hop_duration: float = 10.,
                 fmin: float = 27.5,
                 fmax: float | None = None,
                 stft_window: str='hann',
                 stft_freq_scale: str='log2',
                 stft_center: bool=True,
                 stft_fft_size: int = 1024,
                 bins_per_semitone: int = 1,
                 n_bins: int = 84,
                 center_bins: bool = False,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 transforms: Sequence[torch.nn.Module] | None = None,
                 fold: int | None = None,
                 n_folds: int = 5,
                 cache_dir: str = "/import/research_c4dm/zg032/pesto_cache",
                 filter_unvoiced: bool = False,
                 mmap_mode: str | None = None,
                 preprocessing_method="stft",
                 cutoff_freq = 2048., 
                 resample_sr = 4096
                 ):
        r"""

        Args:
            audio_files: path to csv file containing the list of audio files to process

        """
        super(AudioDataModule, self).__init__()

        # sanity checks
        assert val_audio_files is None or val_annot_files is not None, "Validation set (if it exists) must be annotated"
        assert val_audio_files is None or fold is None, "Specify `val_audio_files` OR cross-validation `fold`, not both"
        assert annot_files is not None or fold is None, "Cannot perform cross-validation without any annotations."

        self.audio_files = Path(audio_files)
        self.annot_files = Path(annot_files) if annot_files is not None else None

        if val_audio_files is not None:
            self.val_audio_files = Path(val_audio_files)
            self.val_annot_files = Path(val_annot_files)
        else:
            self.val_audio_files = None
            self.val_annot_files = None

        self.fold = fold
        self.n_folds = n_folds

        # HCQT
        self.hcqt_sr = None
        self.hcqt_kernels = None
        self.hop_duration = hop_duration

        self.hcqt_kwargs = dict(
            harmonics=list(harmonics),
            fmin=fmin,
            fmax=fmax,
            bins_per_semitone=bins_per_semitone,
            n_bins=n_bins,
            center_bins=center_bins
        )
        # STFT
        self.stft_sr = None
        self.stft_kernels = None
        self.hop_duration = hop_duration

        self.stft_kwargs = dict(
            fmin=fmin,
            fmax=fmax,
            window=stft_window,
            freq_scale=stft_freq_scale,
            center=stft_center,
            n_fft=stft_fft_size
        )

        self.preprocessing_method = preprocessing_method

        self.cutoff_freq=cutoff_freq 
        self.resample_sr=resample_sr 
        
        # dataloader keyword-arguments
        self.dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # transforms
        self.transforms = nn.Sequential(*transforms) if transforms is not None else nn.Identity()

        # misc
        self.cache_dir = Path(cache_dir)
        self.filter_unvoiced = filter_unvoiced
        self.mmap_mode = mmap_mode

        # placeholders for datasets and samplers
        self.train_dataset = None
        self.train_sampler = None
        self.val_dataset = None
        self.val_sampler = None

    def prepare_data(self) -> None:
        self.train_dataset = self.load_data(self.audio_files, self.annot_files)
        if self.val_audio_files is not None:
            self.val_dataset = self.load_data(self.val_audio_files, self.val_annot_files)

    def setup(self, stage: str) -> None:
        # If the dataset is labeled, we split it randomly and keep 20% for validation only
        # Otherwise we train on the whole dataset
        if self.val_dataset is not None:
            return

        if not self.annot_files:
            # create dummy validation set
            self.val_dataset = NpyDataset(np.zeros_like(self.train_dataset.inputs[:1]))
            return

        self.val_dataset = self.load_data(self.audio_files, self.annot_files)

        if self.fold is not None:
            # see https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
            from sklearn.model_selection import KFold

            # We fix random_state=0 for the train/val split to be consistent across runs, even if the global seed changes
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
            iterator = kfold.split(self.train_dataset)
            train_idx, val_idx = None, None  # just to make the linter shut up
            for _ in range(self.fold + 1):
                train_idx, val_idx = next(iterator)

            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        else:
            self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
            self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, sampler=self.train_sampler, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, sampler=self.val_sampler, **self.dl_kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        x, y = batch
        return self.transforms(x), y

    def load_data(self, audio_files: Path, annot_files: Path | None = None) -> torch.utils.data.Dataset:
        if self.preprocessing_method == "hcqt":
            cache_filename = self.build_cqt_filename(audio_files)
        elif self.preprocessing_method == "stft":
            cache_filename = self.build_stft_filename(audio_files)

        if cache_filename.exists():
            inputs = np.load(cache_filename, mmap_mode=self.mmap_mode)
            cache_annot = cache_filename.with_suffix(".csv")
            annotations = np.loadtxt(cache_annot, dtype=np.float32) if cache_annot.exists() else None
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.preprocessing_method == "hcqt":
                inputs, annotations = self.precompute_hcqt(audio_files, annot_files)
            elif self.preprocessing_method == "stft":
                inputs, annotations = self.precompute_stft_LPF_downsample(audio_files, annot_files)
            np.save(cache_filename, inputs, allow_pickle=False)
            print(f"file sucessfully saved to:{cache_filename}")
            if annotations is not None:
                np.savetxt(cache_filename.with_suffix(".csv"), annotations)
        return NpyDataset(inputs, labels=annotations, filter_unvoiced=self.filter_unvoiced)

    def build_cqt_filename(self, audio_files) -> Path:
        # build a hash
        dict_str = json.dumps({
            "audio_files": str(audio_files),
            "hop_duration": self.hop_duration,
            **self.hcqt_kwargs
        }, sort_keys=True)
        hash_id = hashlib.sha256(dict_str.encode()).hexdigest()[:8]

        # build filename
        fname = "hcqt_" + hash_id + ".npy"
        return self.cache_dir / fname

    def build_stft_filename(self, audio_files) -> Path:
        # build a hash
        dict_str = json.dumps({
            "audio_files": str(audio_files),
            "hop_duration": self.hop_duration,
            **self.stft_kwargs
        }, sort_keys=True)
        hash_id = hashlib.sha256(dict_str.encode()).hexdigest()[:8]

        # build filename
        fname = "stft_" + hash_id + ".npy"
        return self.cache_dir / fname


    def precompute_hcqt(self, audio_path: Path, annot_path: Path | None = None) -> Tuple[np.ndarray,np.ndarray]:
        data_dir = audio_path.parent

        cqt_list = []
        with audio_path.open('r') as f:
            audio_files = f.readlines()

        if annot_path is not None:
            with annot_path.open('r') as f:
                annot_files = f.readlines()
            annot_list = []
        else:
            annot_files = []
            annot_list = None

        log.info("Precomputing HCQT...")
        pbar = tqdm(itertools.zip_longest(audio_files, annot_files, fillvalue=None),
                    total=len(audio_files),
                    leave=False)
        for fname, annot in pbar:
            fname = fname.strip()
            pbar.set_description(fname)
            x, sr = torchaudio.load(os.path.join(data_dir, fname))
            out = self.hcqt(x.mean(dim=0), sr)  # convert to mono and compute HCQT  -->  # (time, harmonics, freq_bins, 2)

            if annot is not None:
                annot = annot.strip()
                timesteps, freqs = np.loadtxt(data_dir / annot, delimiter=',', dtype=np.float32).T
                hop_duration = 1000 * (timesteps[1] - timesteps[0])

                # Badly-aligned annotations is a fucking nightmare
                # so we double-check for each file that hop sizes and lengths do match.
                # Since hop sizes are floats we put a tolerance of 1e-6 in the equality
                assert abs(hop_duration - self.hop_duration) < 1e-6, \
                    (f"Inconsistency between {fname} and {annot}:\n"
                     f"the resolution of the annotations ({len(freqs):d}) "
                     f"does not match the number of CQT frames ({len(out):d}). "
                     f"The hop duration between CQT frames should be identical "
                     f"but got {hop_duration:.1f} ms vs {self.hop_duration:.1f} ms. "
                     f"Please either adjust the hop duration of the CQT or resample the annotations.")
                assert len(out) == len(freqs), \
                    (f"Inconsistency between {fname} and {annot}:"
                     f"the resolution of the annotations ({len(freqs):d}) "
                     f"does not match the number of CQT frames ({len(out):d}) "
                     f"despite hop durations match. "
                     f"Please check that your annotations are correct.")
                annot_list.append(hz_to_mid(freqs))

            cqt_list.append(out.cpu().numpy())

        return np.concatenate(cqt_list), np.concatenate(annot_list) if annot_list is not None else None #(concated_time (batch_dimension), harmonics, freq_bins, 2)

    def hcqt(self, audio: torch.Tensor, sr: int):
        # compute CQT kernels if it does not exist yet
        if sr != self.hcqt_sr:
            self.hcqt_sr = sr
            hop_length = int(self.hop_duration * sr / 1000 + 0.5)
            self.hcqt_kernels = HarmonicCQT(sr=sr, hop_length=hop_length, **self.hcqt_kwargs)

        return self.hcqt_kernels(audio).squeeze(0).permute(2, 0, 1, 3)  # (time, harmonics, freq_bins, 2)

    def precompute_stft(self, audio_path: Path, annot_path: Path | None = None) -> Tuple[np.ndarray,np.ndarray]:
        data_dir = audio_path.parent

        stft_list = []
        with audio_path.open('r') as f:
            audio_files = f.readlines()

        if annot_path is not None:
            with annot_path.open('r') as f:
                annot_files = f.readlines()
            annot_list = []
        else:
            annot_files = []
            annot_list = None

        log.info("Precomputing STFT...")
        pbar = tqdm(itertools.zip_longest(audio_files, annot_files, fillvalue=None),
                    total=len(audio_files),
                    leave=False)
        
        batch_size = 256
        frame_size = 80000
        counter = 0
        # batched_audio = []
        buffer = []

        for fname, annot in tqdm(pbar):  # Assuming pbar is a list of (filename, annotation) tuples
            fname = fname.strip()
            x, sr = torchaudio.load(os.path.join(data_dir, fname))
            buffer.append(x.mean(dim=0))
        buffer =  torch.cat(buffer, dim = 0) #(stacked_time, )


        # Calculate the total number of frames we can create
        num_frames = buffer.size(0) // frame_size

        # Reshape the buffer into (num_frames, frame_size)
        buffer = buffer[:num_frames * frame_size].view(num_frames, frame_size)

        # Batch the frames
        num_batches = num_frames // batch_size
        batched_audio = buffer[:num_batches * batch_size].view(num_batches, batch_size, frame_size)

        for i, batch in enumerate(batched_audio):
            out = self.stft_preprocess(batch.to("cuda:0"), sr)  # convert to mono and compute STFT  -->  # (time, 1, freq_bins, 2)
            #TODO how to make this faster?  
            print(f"processed: {i}/{len(batched_audio)}, {out.shape}") 

            stft_list.append(out.cpu().numpy()) #(num_samples*time_steps, 1, freq_bins, 2)
        print("pre-processing finished!")
        return np.concatenate(stft_list), np.concatenate(annot_list) if annot_list is not None else None #(concated_time (batch_dimension), harmonics, freq_bins, 2)

    def precompute_stft_LPF_downsample(self, audio_path: Path, annot_path: Path | None = None) -> Tuple[np.ndarray,np.ndarray]:
        data_dir = audio_path.parent

        stft_list = []
        with audio_path.open('r') as f:
            audio_files = f.readlines()

        if annot_path is not None:
            with annot_path.open('r') as f:
                annot_files = f.readlines()
            annot_list = []
        else:
            annot_files = []
            annot_list = None

        log.info("Precomputing STFT...")
        pbar = tqdm(itertools.zip_longest(audio_files, annot_files, fillvalue=None),
                    total=len(audio_files),
                    leave=False)
        
        batch_size = 256 #process this many samples at once
        frame_size = 80000 #number of samples in a frame
        counter = 0
        buffer = []
        
        for fname, annot in tqdm(pbar):  # Assuming pbar is a list of (filename, annotation) tuples
            fname = fname.strip()
            x, sr = torchaudio.load(os.path.join(data_dir, fname))
            waveform_LPF_resampled = self.LPF_resample(x, sr, self.cutoff_freq, self.resample_sr)
            buffer.append(waveform_LPF_resampled.mean(dim=0))

        buffer =  torch.cat(buffer, dim = 0) #(stacked_time, )

        # Calculate the total number of frames we can create
        num_frames = buffer.size(0) // frame_size

        # Reshape the buffer into (num_frames, frame_size)
        buffer = buffer[:num_frames * frame_size].view(num_frames, frame_size)

        # Batch the frames
        num_batches = num_frames // batch_size
        batched_audio = buffer[:num_batches * batch_size].view(num_batches, batch_size, frame_size)

        for i, batch in enumerate(batched_audio):
            out = self.stft_preprocess(batch.to("cuda"), self.resample_sr)  # convert to mono and compute STFT  -->  # (time, 1, freq_bins, 2)
            print(f"processed: {i}/{len(batched_audio)}") 
            stft_list.append(out.cpu().numpy()) #(num_samples*time_steps, 1, freq_bins, 2)
        return np.concatenate(stft_list), np.concatenate(annot_list) if annot_list is not None else None #(concated_time (batch_dimension), harmonics, freq_bins, 2)

    def precompute_stft_with_memmap(self, audio_path: Path, annot_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
        data_dir = audio_path.parent

        with audio_path.open('r') as f:
            audio_files = f.readlines()

        if annot_path is not None:
            with annot_path.open('r') as f:
                annot_files = f.readlines()
            annot_list = []
        else:
            annot_files = []
            annot_list = None

        log.info("Precomputing STFT...")
        pbar = tqdm(itertools.zip_longest(audio_files, annot_files, fillvalue=None),
                    total=len(audio_files),
                    leave=False)

        batch_size = 256
        frame_size = 80000
        buffer = []

        # Collect all audio data into a single buffer
        for fname, annot in tqdm(pbar):
            fname = fname.strip()
            x, sr = torchaudio.load(os.path.join(data_dir, fname))
            buffer.append(x.mean(dim=0))
        buffer = torch.cat(buffer, dim=0)  # (stacked_time,)

        # Calculate the total number of frames we can create
        num_frames = buffer.size(0) // frame_size

        # Reshape the buffer into (num_frames, frame_size)
        buffer = buffer[:num_frames * frame_size].view(num_frames, frame_size)

        # Batch the frames
        num_batches = num_frames // batch_size
        batched_audio = buffer[:num_batches * batch_size].view(num_batches, batch_size, frame_size)

        # Memory-map file for STFT results
        # stft_shape = (num_batches * batch_size, frame_size // 2 + 1, 2)  # Assume freq_bins = frame_size // 2 + 1  #TODO: figure out shape
        
        #test shape
        example_batch = batched_audio[0].to("cuda:0")  # A single batch to determine the shape
        example_out = self.stft_preprocess(example_batch, sr)

        # The final output shape is (batch_size * time_steps, 1, freq_bins, 2)
        num_time_steps = example_out.shape[0] // batch_size
        freq_bins = example_out.shape[2]

        print(f"num_time_steps:{num_time_steps}, freq_bins:{freq_bins}")

        stft_shape = (num_batches, batch_size * num_time_steps, 1, freq_bins, 2)

        
        stft_memmap = np.memmap('stft_output.dat', dtype='float32', mode='w+', shape=stft_shape)

        for i, batch in enumerate(batched_audio):
            out = self.stft_preprocess(batch.to("cuda:0"), sr)  # convert to mono and compute STFT
            print(f"processed: {i}/{len(batched_audio)}, {out.shape}")

            # Write the output directly to the memory-mapped file
            stft_memmap[i * batch_size:(i + 1) * batch_size] = out.cpu().numpy()

            # Free GPU memory after processing each batch
            # torch.cuda.empty_cache()

        # Flush changes to disk
        stft_memmap.flush()

        # Handle annotations similarly if annot_list is not None
        if annot_list is not None:
            annot_memmap = np.concatenate(annot_list)
            return stft_memmap, annot_memmap
        else:
            return stft_memmap, None

    def stft_preprocess(self, audio: torch.Tensor, sr: int):
        # compute CQT kernels if it does not exist yet
        if sr != self.stft_sr:
            self.stft_sr = sr
            hop_length = int(self.hop_duration * sr / 1000 + 0.5)
            self.stft_kernels_preprocess = STFT(sr=sr, hop_length=hop_length, **self.stft_kwargs).to("cuda")

        batch_time_freq_2 = self.stft_kernels_preprocess(audio).permute(0, 2, 1, 3) #batch, time, freq, 2
        batchtime_freq_2 = batch_time_freq_2.reshape(batch_time_freq_2.shape[0]*batch_time_freq_2.shape[1], batch_time_freq_2.shape[2], batch_time_freq_2.shape[3])
        batchtime_1_freq_2 = batchtime_freq_2.unsqueeze(1)
        return  batchtime_1_freq_2#(batch_size*time_steps, 1, freq_bins,, 2)

    def stft(self, audio: torch.Tensor, sr: int):
        if sr != self.stft_sr:
            self.stft_sr = sr
            hop_length = int(self.stft_kwargs['n_fft']/4*3)
            self.stft_kernels = STFT(sr=sr, hop_length=hop_length, **self.stft_kwargs)
        return self.stft_kernels(audio).permute(2, 0, 1, 3)  # (1, freq_bins,time_steps, 2) --> (time_steps, 1, freq_bins,2); will need to be in this format: (time, harmonics, freq_bins, 2) 

    def LPF_resample(self, audio, sr, cutoff_freq, new_sr):
        #LPF
        waveform_LPF = low_pass_filter(audio, cutoff_freq, sr)
        #Downsample
        waveform_LPF_resampled = resample_waveform(waveform_LPF, sr, new_sr)
        return waveform_LPF_resampled


if __name__=="__main__":
    audio_files = "/import/research_c4dm/zg032/LibriSpeech_train_clean_100/LibriSpeech_train-clean-100.csv"

        # Initialize the AudioDataModule
    audio_data_module = AudioDataModule(
        audio_files=audio_files,
        annot_files=None,
        harmonics=[1],
        hop_duration=10.0,
        fmin=0.01,
        fmax=500,
        bins_per_semitone=3,
        n_bins=99 * 3 - 1,  # Based on the configuration provided
        center_bins=True,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        transforms=[],  # Assuming ToLogMagnitude is defined somewhere
        fold=None,
        n_folds=5,
        cache_dir='./cache',
        filter_unvoiced=False,
        mmap_mode=None,
        preprocessing_method = "stft",
        stft_window = "hann", 
        stft_freq_scale = "log",
        stft_center = True
    )
    audio_data_module.prepare_data()
    print(f"audio_data_module:{audio_data_module.train_dataset[0]}")
