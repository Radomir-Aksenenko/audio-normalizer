# Audio Normalizer

A multifunctional application for audio normalization in video files with multi-language support and batch processing.

## Features

- 🎯 Automatic audio normalization to professional standards
- 📁 Processing of both single files and entire folders
- 🌐 Support for 10 interface languages:
  - English (en)
  - Russian (ru)
  - Spanish (es)
  - French (fr)
  - German (de)
  - Italian (it)
  - Portuguese (pt)
  - Japanese (ja)
  - Chinese (zh)
  - Korean (ko)
- ⚡ Multi-threaded file processing
- 📊 Progress tracking
- 🚫 Operation cancellation support
- 🎨 Modern dark-themed interface

## Requirements

- Python 3.8 or higher
- FFmpeg
- The following Python packages:
  - customtkinter
  - pillow
  - moviepy

## Installation

1. Install FFmpeg:
   - Windows: [Download FFmpeg](https://ffmpeg.org/download.html)
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Launch the application:
```bash
python normalize_audio.py
```

2. Select the operation mode:
   - "Single File" - for processing a single file
   - "Folder" - for processing all video files in a folder

3. Select a file or folder to process

4. Select a folder to save processed files

5. Click "Normalize" to start processing

## Supported Formats

- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)

## Processing Features

- Automatic audio parameter analysis
- Normalization to standard loudness level (-23 LUFS)
- Dynamic normalization for even sound
- Original video stream preservation
- High-quality audio (320k bitrate, 48kHz)

## License

MIT License 