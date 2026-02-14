DOWNLOADS: 
https://huggingface.co/datasets/lovepineapples/PAIT-Downloads/tree/main

This is my collection of portable AI packages, mainly for video/audio production to run it fast without anxious headache in console, sort of "Awesome N" repos, but for portables for win. initially, I made these tools for myself, but maybe someone else will need them. All portables can work offline and tested on gtx 1050 ti 4gb(cuda) and core i3 1005g1 (cpu). OK, heres the list:

!!! IF SOMETHING ISN'T WORKING, MAKE SURE THAT PATH TO TOOL DON'T HAVE SPACES OR NON-ENGLISH SYMBOLS !!!

!!! TO AVOID CUDA OUT OF MEMORY ERRORS, INSTALL NVIDIA DRIVERS 535 OR NEWER !!!

### -TEXT-

Koboldai (without models) [CPU/CUDA] - [link](https://github.com/KoboldAI/KoboldAI-Client/releases/) - also in downloads / [online demo](https://lite.koboldai.net/)

### -CHAT-
Google Gemma 3 4B Instruct Q4_0(QAT) koboldcpp webui [Vulkan/OpenCL/CPU] - in downloads / [source](https://github.com/ggerganov/llama.cpp) / [webui](https://github.com/LostRuins/koboldcpp/releases) / [model](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf)

ChatterUI (same as koboldcpp, but on android) [CPU] - [link](https://github.com/Vali-98/ChatterUI/releases/) - also in downloads


### -MIDI MUSIC GENERATION-

Midi composer app [CUDA][CPU] - [link](https://github.com/SkyTNT/midi-model/releases/tag/v1.1.0) - also in downloads / [source](https://github.com/SkyTNT/midi-model) / [online demo](https://huggingface.co/spaces/skytnt/midi-composer) 

 Multitrack midi music generator (generates short jingles, each instrument generated separately) [CPU] - in downloads / [webui](https://huggingface.co/spaces/juancopi81/multitrack-midi-music-generator) 

### -TEXT TO MUSIC/AUDIO-

Stable Audio Open 1.0 [CUDA/CPU] - in downloads / [source](https://github.com/Stability-AI/stable-audio-tools) / [model](https://huggingface.co/stabilityai/stable-audio-open-1.0) / [online demo](https://huggingface.co/spaces/ameerazam08/stableaudio-open-1.0) 

### -TEXT TO SPEECH-

Coqui XTTS2 webui (voice cloning is more "stable" than bark, but less artistic) [CUDA/CPU] - in downloads / [source](https://github.com/coqui-ai/TTS) / [webui](https://github.com/BoltzmannEntropy/xtts2-ui)  

Suno ai Bark webui (tts is more chaotic than xtts, but if you have patience, you can roll ideal variant) (with zeroshot voice conversion) [CUDA/CPU] - in downloads / [source](https://github.com/suno-ai/bark) / [webui](https://github.com/C0untFloyd/bark-gui) / [online demo](https://huggingface.co/spaces/suno/bark) 

TorToiSe webui (english-only) [CUDA/CPU] - in downloads / [source](https://github.com/neonbjb/tortoise-tts) / [webui](https://git.ecker.tech/mrq/ai-voice-cloning) / [online demo](https://replicate.com/afiaka87/tortoise-tts) 

### -VOICE CONVERSION VIA TRAINING-

RVC singing voice cloning webui [CUDA] - [link](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) - also in downloads / [source](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/README.en.md)  

### -VOICE ZEROSHOT CONVERSION-

FreeVC webui [CPU] - in downloads / [source](https://github.com/OlaWod/FreeVC) / [webui](https://huggingface.co/spaces/OlaWod/FreeVC) 

### -VOICE TO TEXT-

Whispercpp GUI [DirectX/CPU] - [link](https://github.com/Const-me/Whisper/releases/) - also in downloads / [source](https://github.com/ggerganov/whisper.cpp) / [gui](https://github.com/Const-me/Whisper) / [online demo](https://replicate.com/openai/whisper) 

### -UNIVERSAL AUDIO RESTORATION-

AudioSR (cli interface) [CUDA/CPU] - in downloads / [source](https://github.com/haoheliu/versatile_audio_super_resolution) 

### -VOCALS RESTORATION-

VoiceFixer webui [CPU] - in downloads / [source](https://github.com/haoheliu/voicefixer) / [webui](https://huggingface.co/spaces/Kevin676/VoiceFixer) 

### -DUAL SPEAKER SPEECH SEPARATION-

Dual Path RNN (w/o gui) [CPU] - in downloads / [source](https://github.com/JusperLee/Dual-Path-RNN-Pytorch) 

### -STEMS EXTRACTION BY PROMPT-

AudioSep webui [CUDA/CPU] - in downloads / [source](https://github.com/Audio-AGI/AudioSep) / [webui](https://huggingface.co/spaces/Audio-AGI/AudioSep) 

### -VOCALS/STEMS EXTRACTION-

UVR [CPU/CUDA] - [link](https://github.com/Anjok07/ultimatevocalremovergui/releases/) - also in downloads / [online demo](https://mvsep.com/)  

Demucs gui [CPU][CUDA] - [link](https://carlgao4.github.io/demucs-gui/) - also in downloads / [source](https://github.com/facebookresearch/demucs) / [gui](https://github.com/CarlGao4/Demucs-Gui) 

### -IMAGE COLORIZATION-

DeOldify .NET gui [CPU] - [link](https://github.com/ColorfulSoft/DeOldify.NET/releases) - also in downloads / [source](https://github.com/jantic/DeOldify) / [gui](https://github.com/ColorfulSoft/DeOldify.NET) / [online demo](https://huggingface.co/spaces/leonelhs/deoldify) 

### -ZEROSHOT IMAGE MATTING-

DIS (BRIAAI RMBG 1.4 model) webui [CPU] - in downloads / [source](https://github.com/xuebinqin/DIS) / [webui](https://huggingface.co/spaces/ECCV2022/dis-background-removal) / [model](https://huggingface.co/briaai/RMBG-1.4) 

### -MONOCULAR-DEPTH-ESTIMATION-

ZoeDepth-webui [CUDA/CPU][CPU] - in downloads / [source](https://github.com/isl-org/ZoeDepth) / [webui](https://huggingface.co/spaces/shariqfarooq/ZoeDepth)  

### -IMAGE UPSCALING-

real-ESRGAN-gui [Vulkan] - [link](https://github.com/TransparentLC/realesrgan-gui/releases) - also in downloads / [source](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) / [gui](https://github.com/TransparentLC/realesrgan-gui) / [online demo](https://replicate.com/xinntao/realesrgan) 

ChaiNNer (supports a LOT of upscaling methods) [CUDA/Vulkan] - [link](https://github.com/chaiNNer-org/chaiNNer/releases) - also in downloads / [gui](https://github.com/chaiNNer-org/chaiNNer) 

Automatic1111 sdwebui with StableSR extension [CUDA/CPU] - in downloads / [source](https://github.com/IceClear/StableSR) / [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) / [extension](https://github.com/pkuliyi2015/sd-webui-stablesr) 

### -IMAGE RELIGHTING-

IC-Light webui [CUDA] - in downloads / [source](https://github.com/lllyasviel/IC-Light)

### -TEXT2IMAGE-

Automatic1111 Stable Diffusion webui base (without models) - [link](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases)  / [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 

Automatic1111 sd-webui deliberate v2 (sd1.5) model [CUDA/CPU][DIRECTX/CPU] - in downloads / [source](https://github.com/CompVis/stable-diffusion) / [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  / [directx webui](https://github.com/lshqqytiger/stable-diffusion-webui-directml) / [model](https://huggingface.co/XpucT/Deliberate) 

lllyasviel sd-webui-forge (faster and less vram usage on nvidia cards) deliberate v2 (sd1.5) model [CUDA] - in downloads / [source](https://github.com/lllyasviel/stable-diffusion-webui-forge) / [model](https://huggingface.co/XpucT/Deliberate) 

Automatic1111 sd-webui SDXL 1.0 (sdxl) model [CUDA/CPU] - in downloads  / [source](https://github.com/CompVis/stable-diffusion) / [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  / [model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) / [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) 

Fooocus (sdxl) [CUDA] - [link](https://github.com/lllyasviel/Fooocus/releases) - also in downloads / [source](https://github.com/Stability-AI/generative-models) / [webui](https://github.com/lllyasviel/Fooocus) / [model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) / [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)  

ComfyUI (without models) [CUDA/CPU] - [link](https://github.com/comfyanonymous/ComfyUI/releases/tag/latest) - also in downloads / [source](https://github.com/CompVis/stable-diffusion) / [webui](https://github.com/comfyanonymous/ComfyUI) 

### -IMAGE EDITING BY PROMPT-

Automatic1111 Instructpix2pix (sd1.5) model (you also can download just model and use in default automatic1111 if you want, webui doesnt downloads any other files while loading this one) [DIRECTX/CPU][CUDA/CPU] - in downloads / [source](https://github.com/CompVis/stable-diffusion) / [ip2p source](https://github.com/timothybrooks/instruct-pix2pix) / [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  / [directx webui](https://github.com/lshqqytiger/stable-diffusion-webui-directml) / [model](http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt) 

### -IMAGE TO IMAGE VARIATIONS-

Automatic1111 sd-unclip (sd2.1) model (there is an alternative that works without any models - controlnet reference) [CUDA/CPU] - in downloads / [source](https://github.com/CompVis/stable-diffusion) / [webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  / [model](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip) 

### -IMAGE EDITING BY CONCEPTS-

LEDITS webui [CUDA/CPU] - in downloads / [source](https://editing-images-project.hf.space/index.html) / [webui](https://huggingface.co/spaces/editing-images/ledits) 

### -OBJECT REMOVING-

lama cleaner [CPU] - in downloads / [source](https://github.com/advimman/lama) / [webui](https://github.com/Sanster/lama-cleaner) / [online demo](https://huggingface.co/spaces/Sanster/Lama-Cleaner-lama)

### -VIDEO FRAMES INTERPOLATION-

Flowframes [CUDA/Vulkan] - in downloads / [source](https://github.com/megvii-research/ECCV2022-RIFE) / [gui](https://nmkd.itch.io/flowframes) 

### -VIDEO UPSCALING-

RealBasicVSR (cli interface) [CUDA/CPU] - in downloads / [source](https://github.com/ckkelvinchan/RealBasicVSR) 

### -VIDEO HUMAN MATTING-

RobustVideoMatting (w/o gui) [CUDA/CPU] - in downloads / [source](https://github.com/PeterL1n/RobustVideoMatting) / [online demo](https://replicate.com/arielreplicate/robust_video_matting) 

### -VIDEO ZERO-SHOT MATTING-

Track-anything webui [CPU] - in downloads / [webui](https://github.com/gaomingqi/Track-Anything) / [online demo](https://huggingface.co/spaces/VIPLab/Track-Anything) 

### -VIDEO FEW-SHOT MATTING VIA TRAINING-

DeepXTools by Iperov [CUDA] - [link](https://github.com/iperov/DeepXTools/releases/tag/install_win_nvidia) - also in downloads

### -ZERO-SHOT DEEPFAKING-

Roop neurogen mod (Refacer model) (lightning fast, has realtime deepfake on webcam function) (the refacer model swaps faces better than simswap, but have only 128px resolution and may have more artifacts when head is on side) [DirectX/CUDA/CPU] - in downloads / [source](https://github.com/deepinsight/insightface/tree/master/web-demos/swapping_discord) / [webui](https://github.com/s0md3v/roop) / [mod by](https://t.me/neurogen_news) 

Deepinsight Refacer gradio webui (replaces only certain faces, has cool face upscale feature) [CUDA] - in downloads / [source](https://github.com/deepinsight/insightface/tree/master/web-demos/swapping_discord) / [webui](https://github.com/xaviviro/refacer) / [mod by](https://t.me/neurogen_news) 

Simswap (w/o gui) [CUDA/CPU] - in downloads / [source](https://github.com/neuralchen/SimSwap) 

### -DEEPFAKING VIA TRAINING-

DeepFaceLab (w/o gui) [DirectX][CUDA] - [link](https://mega.nz/folder/Po0nGQrA#dbbttiNWojCt8jzD4xYaPw) - also in downloads / [source](https://github.com/iperov/DeepFaceLab) 

DeepfaceLive [DirectX][CUDA] - [link](https://mega.nz/folder/m10iELBK#Y0H6BflF9C4k_clYofC7yA)  - also in downloads / [source](https://github.com/iperov/DeepFaceLive) 

### -LIPS MANIPULATION ON VIDEO-

wav2lip gui [CUDA/CPU] - [link](https://github.com/dunnousername/Wav2Lip/releases) - also in downloads / [source](https://github.com/Rudrabha/Wav2Lip) / [gui](https://github.com/dunnousername/Wav2Lip)  

### -SINGLE IMAGE To MESH-

TripoSR (outputs is still rough, but better, than shap-e) [CUDA/CPU] - in downloads / [source](https://github.com/VAST-AI-Research/TripoSR) / [online demo](https://huggingface.co/spaces/stabilityai/TripoSR) 

### -MESH GENERATION BY IMAGES-

Dust3r webui (one model that does end-to-end photogrammetry, useful when traditional photogrammetry software like metashape dont determines camera positions, but quality may be bad) [CUDA/CPU] - in downloads / [source](https://github.com/naver/dust3r) 


### -NOVEL VIEWS GENERATION BY IMAGES-

NERFStudio (splatfacto, nerfacto) [CUDA/CPU(cpu is extremely slow,but working)] - in downloads / [source](https://github.com/nerfstudio-project/nerfstudio) 


--------------------------------------------------------------

You can theoretically run these tools on windows 7, just download [this](https://huggingface.co/datasets/lovepineapples/PAIT-Downloads/resolve/main/api-ms-win-core-path-l1-1-0.dll?download=true) file and place it along with python.exe

--------------------------------------------------------------

Page on itch.io: https://lovepineapples.itch.io/ai-portable-tools

Alternative downloads with torrents on Archive.org: https://archive.org/details/@takeonme1?tab=uploads


