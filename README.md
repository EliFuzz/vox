# Vox

**Vox** is a small macOS menu bar app for local speech-to-text.
Hold a shortcut, talk, and it drops the text right where your cursor already is.

No giant window. No extra editor. No copy/paste dance.

> Speak. Release. Keep moving.

<video controls autoplay loop muted src="https://github.com/user-attachments/assets/33c9994b-2b58-421b-b949-b9faedea1ba8"></video>

## Features

- **local transcription** - your speech becomes text on-device
- **cursor-first output** - words land in the app you're already using
- **two recording styles** - push-to-talk or toggle-to-talk
- **pick your mic, pick your model** - without digging through settings files

## Quick start

1. Use GitHub Releases to grab the latest version of **Vox** for macOS. Just download, unzip, and move it to your Applications folder.
2. Open **Vox**
3. Allow **Accessibility** on macOS _(and **Input Monitoring** too, if macOS asks)_
4. Hold **Fn + Left Control** to talk
5. Release to paste text or wait a half-second for it to paste automatically while you keep talking
6. Double-press the shortcut for **toggle-to-talk**

## Under the hood

| area             | details                                      |
| ---------------- | -------------------------------------------- |
| input            | selectable microphone device                 |
| speech detection | VAD-gated recording                          |
| transcription    | local ASR pipeline                           |
| models           | Multilingual, English, English Small         |
| engines          | Parakeet, Moonshine                          |
| post-processing  | sentence normalization + punctuation cleanup |
| output           | paste at cursor + restore previous clipboard |
| settings         | `~/.vox/settings.json`                       |
| model cache      | `~/.vox/models/`                             |
| model delivery   | downloads models when needed                 |

## project layout

- `src/main.rs` - boot, workers, update checks
- `src/processing.rs` - speech segmentation + transcription flow
- `src/tray/` - menu bar state and menus
- `src/platform/macos/` - keyboard hooks, Accessibility, paste, CoreML bridge
- `src/model_download.rs` - model fetch + unzip
- `src/settings.rs` - mode, shortcut, device, model
