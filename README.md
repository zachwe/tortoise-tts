# AI Voice Cloning for Retards and Savants

This [rentry](https://rentry.org/AI-Voice-Cloning/) aims to serve as both a foolproof guide for setting up AI voice cloning tools for legitimate, local use on Windows (with an Nvidia GPU), as well as a stepping stone for anons that genuinely want to play around with [TorToiSe](https://github.com/neonbjb/tortoise-tts).

Similar to my own findings for Stable Diffusion image generation, this rentry may appear a little disheveled as I note my new findings with TorToiSe. Please keep this in mind if the guide seems to shift a bit or sound confusing.

>\>B-but what about the colab notebook/hugging space instance??

I link those a bit later on as alternatives for Windows+AMD users. You're free to skip the installation section and jump after that.

>\>Ugh... why bother when I can just abuse 11.AI?

I very much encourage (You) to use 11.AI while it's still viable to use. For the layman, it's easier to go through the hoops of coughing up the $5 or abusing the free trial over actually setting up a TorToiSe environment and dealing with its quirks.

However, I also encourage your own experimentation with TorToiSe, as it's very, very promising, it just takes a little love and elbow grease.

## Modifications

My fork boasts the following additions, fixes, and optimizations:
* a competent web UI made in Gradio to expose a lot of tunables and options
* cleaned up output structure of resulting audio files
* caching computed conditional latents for faster re-runs
	- additionally, regenerating them if the script detects they're out of date
* uses the entire audio sample instead of the first four seconds of each sound file for better reproducing
* activated unused DDIM sampler
* ease of setup for the most inexperienced Windows users
* use of some optimizations like `kv_cache`ing for the autoregression sample pass, and keeping data on GPU 
* and more!

## Installing

Outside of the very small prerequisites, everything needed to get TorToiSe working is included in the repo.

For Windows users with an AMD GPU, tough luck, as ROCm drivers are not (easily) available for Windows, and requires inane patches with PyTorch. Consider using the [Colab notebook](https://colab.research.google.com/drive/1wVVqUPqwiDBUVeWWOUNglpGhU3hg_cbR?usp=sharing), or the [Hugging Face space](https://huggingface.co/spaces/mdnestor/tortoise), for `tortoise-tts`.

### Pre-Requirements

Windows:
* Python 3.9: https://www.python.org/downloads/release/python-3913/
* Git (optional): https://git-scm.com/download/win

Linux:
* python3.x
* git
* ROCm for AMD, CUDA for NVIDIA

### Setup

#### Windows

Download Python and Git and run their installers.

After installing Python, open the Start Menu and search for `Command Prompt`. Type `cd `, then drag and drop the folder you want to work in (experienced users can just `cd <path>` directly), then hit Enter.

Paste `git clone https://git.ecker.tech/mrq/tortoise-tts` to download TorToiSe and additional scripts, then hit Enter. Inexperienced users can just download the repo as a ZIP, and extract.

Afterwards, run `setup.bat` to automatically set things up.

If you've done everything right, you shouldn't have any errors.

#### Linux

First, make sure you have both `python3.x` and `git` installed, as well as the required compute platform according to your GPU (ROCm or CUDA)

```
git clone https://git.ecker.tech/mrq/tortoise-tts
cd tortoise-tts
chmod +x *.sh
```

Then, depending on your GPU:
`./setup-rocm.sh # if AMD`

`./setup-cuda.sh # if NVIDIA`

And you should be done!

### Updating

To check for updates, simply run `update.bat` (or `update.sh`). It should pull from the repo, as well as fetch for any new dependencies.

### Pitfalls You May Encounter

I'll try and make a list of "common" (or what I feel may be common that I experience) issues with getting TorToiSe set up:
* `CUDA is NOT available for use.`: If you're on Linux, you failed to set up CUDA (if NVIDIA) or ROCm (if AMD). Please make sure you have these installed on your system.
	If you're on Windows with an AMD card, you're stuck out of luck, as ROCm is not available on Windows (without major hoops to be jumped). If you're on an NVIDIA GPU, then I'm not sure what went wrong.
* `failed reading zip archive: failed finding central directory`: You had a file fail to download completely during the model downloading initialization phase. Please open either `.\models\tortoise\` or `.\models\transformers\`, and delete the offending file.
	You can deduce what that file is by reading the stack trace. A few lines above the last like will be a line trying to read a model path.
* `torch.cuda.OutOfMemoryError: CUDA out of memory.`: You most likely have a GPU with low VRAM (~4GiB), and the small optimizations with keeping data on the GPU is enough to OOM. Please open the `start.bat` file and add `--low-vram` to the command (for example: `py app.py --low-vram`) to disable those small optimizations.
* `WavFileWarning: Chunk (non-data) not understood, skipping it.`: something about your WAVs are funny, and its best to remux your audio files with FFMPEG (included batch file in `.\convert\`).
	Honestly, I don't know if this does impact output quality, as I feel it's placebo when I do try and correct this.

## Preparing Voice Samples

Now that the tough part is dealt with, it's time to prepare voice clips to use.

Unlike training embeddings for AI image generations, preparing a "dataset" for voice cloning is very simple.

As a general rule of thumb, try to source clips that aren't noisy, solely the subject you are trying to clone, and doesn't contain any non-words (like yells, guttural noises, etc.). If you must, run your source through a background music/noise remover (how to is an exercise left to the reader). It isn't entirely a detriment if you're unable to provide clean audio, however. Just be wary that you might have some headaches with getting acceptable output.

Nine times out of ten, you should be fine using as many clips as possible. There's (now) no preference between combining your audio into one file, or leaving it split. However, if you're aiming for a specific delivery, it *should* be best for you to narrow down to just using that as your provided source (for example, changing one word in a line).

There's no hard specifics on how many, or how long, your sources should be.

If you're looking to trim your clips, in my opinion, ~~Audacity~~ Tenacity works good enough, as you can easily output your clips into the proper format (22050 Hz sampling rate).

Power users with FFMPEG already installed can simply used the provided conversion script in `.\convert\`.

After preparing your clips as WAV files at a sample rate of 22050 Hz, open up the `tortoise-tts` folder you're working in, navigate to `./tortoise/voice/`, create a new folder in whatever name you want, then dump your clips into that folder. While you're in the `voice` folder, you can take a look at the other provided voices.

## Using the Software

Now you're ready to generate clips. With the command prompt still open, simply enter `start.bat` (or `start.sh`), and wait for it to print out a URL to open in your browser, something like `http://127.0.0.1:7860`.

If you're looking to access your copy of TorToiSe from outside your local network, pass `--share` into the command (for example, `python app.py --share`). You'll get a temporary gradio link to use.

### Generate

You'll be presented with a bunch of options in the default `Generate` tab, but do not be overwhelmed, as most of the defaults are sane, but below are a rough explanation on which input does what:
* `Prompt`: text you want to be read. You wrap text in `[brackets]` for "prompt engineering", where it'll affect the output, but those words won't actually be read.
* `Line Delimiter`: String to split the prompt into pieces. The stitched clip will be stored as `combined.wav`
	- Setting this to `\n` will generate each line as one clip before stitching it. Leave blank to disable this.
* `Emotion`: the "emotion" used for the delivery. This is a shortcut to utilizing "prompt engineering" by starting with `[I am really <emotion>,]` in your prompt. This is merely a suggestion, not a guarantee.
* `Custom Emotion + Prompt`: a non-preset "emotion" used for the delivery. This is a shortcut to utilizing "prompt engineering" by starting with `[<emotion>]` in your prompt.
* `Voice`: the voice you want to clone. You can select `microphone` if you want to use input from your microphone.
* `Microphone Source`: Use your own voice from a line-in source.
* `Candidates`: number of outputs to generate, starting from the best candidate. Depending on your iteration steps, generating the final sound files could be cheap, but they only offer alternatives to the samples generated to pull from (in other words, the later candidates perform worse), so don't be compelled to generate a ton of candidates.
* `Seed`: initializes the PRNG to this value. Use this if you want to reproduce a generated voice.
* `Preset`: shortcut values for sample count and iteration steps. Clicking a preset will update its corresponding values. Higher presets result in better quality at the cost of computation time.
* `Samples`: analogous to samples in image generation. More samples = better resemblance / clone quality, at the cost of performance. This strictly affects clone quality.
* `Iterations`: influences audio sound quality in the final output. More iterations = higher quality sound. This step is relatively cheap, so do not be discouraged from increasing this. This strictly affects quality in the actual sound.
* `Temperature`: how much randomness to introduce to the generated samples. Lower values = better resemblance to the source samples, but some temperature is still required for great output. This value is very inconsistent and entirely depends on the input voice. In other words, some voices will be receptive to playing with this value, while others won't make much of a difference.
* `Pause Size`: Governs how large pauses are at the end of a clip (in token size, not seconds). Increase this if your output gets cut off at the end.
* `Diffusion Sampler`: sampler method during the diffusion pass. Currently, only `P` and `DDIM` are added, but does not seem to offer any substantial differences in my short tests.
	`P` refers to the default, vanilla sampling method in `diffusion.py`.
	To reiterate, this ***only*** is useful for the diffusion decoding path, after the autoregressive outputs are generated.

After you fill everything out, click `Run`, and wait for your output in the output window. The sampled voice is also returned, but if you're using multiple files, it'll return the first file, rather than a combined file.

All outputs are saved under `./result/[voice name]/[timestamp]/` as `result.wav`, and the settings in `input.txt`. There doesn't seem to be an inherent way to add a Download button in Gradio, so keep that folder in mind.

To save you from headaches, I strongly recommend playing around with shorter sentences first to find the right values for the voice you're using before generating longer sentences.

As a quick optimization, I modified the script to have the `conditional_latents` are saved after loading voice samples, and subsequent uses will load that file directly (at the cost of not returning the `Sample voice` to the web UI). Additionally, these `conditional_latents` are also computed in a way to use the entire clip, rather than the first four seconds the original tortoise-tts uses. If there's voice samples that have a modification time newer than this cached file, it'll skip loading it and load the normal WAVs instead.

**!**NOTE**!**: cached `latents.pth` files generated before 2023.02.05 will be ignored, due to a change in computing the conditiona latents. This *should* help bump up voice cloning quality. Apologies for the inconvenience.

### Utilities

In this tab, you can find some helper utilities that might be of assistance.

For now, an analog to the PNG info found in Voldy's Stable Diffusion Web UI resides here. With it, you can upload an audio file generated with this web UI to view the settings used to generate that output. Additionally, the voice latents used to generate the uploaded audio clip can be extracted.

If you want to reuse its generation settings, simply click "Copy Settings".

### Settings

This tab (should) hold a bunch of other settings, from tunables that shouldn't be tampered with, to settings pertaining to the web UI itself.

Below are settings that override the default launch arguments. Some of these require restarting to work.
* `Public Share Gradio`: overrides `--share`. Tells Gradio to generate a public URL for the web UI
* `Check for Updates`: checks for updates on page load and notifies in console. Only works if you pulled this repo from a gitea instance.
* `Low VRAM`: disables optimizations in TorToiSe that increases VRAM consumption. Suggested if your GPU has under 6GiB.
* `Embed Output Metadata`: enables embedding the settings and latents used to generate that audio clip inside that audio clip. Metadata is stored as a JSON string in the `lyrics` tag.
* `Slimmer Computed Latents`: falls back to the original, 12.9KiB way of storing latents (without the extra bits required for using the CVVP model).
* `Voice Latent Max Chunk Size`: during the voice latents calculation pass, this limits how large, in bytes, a chunk can be. Large values can run into VRAM OOM errors.
* `Sample Batch Size`: sets the batch size when generating autoregressive samples. Bigger batches result in faster compute, at the cost of increased VRAM consumption. Leave to 0 to calculate a "best" fit.
* `Concurrency Count`: how many Gradio events the queue can process at once. Leave this over 1 if you want to modify settings in the UI that updates other settings while generating audio clips.

Below are an explanation of experimental flags. Messing with these might impact performance, as these are exposed only if you know what you are doing.
* `Half-Precision`: (attempts to) hint to PyTorch to auto-cast to float16 (half precision) for compute. Disabled by default, due to it making computations slower.
* `Conditional Free`: a quality boosting improvement at the cost of some performance. Enabled by default, as I think the penaly is negligible in the end.
* `CVVP Weight`: governs how much weight the CVVP model should influence candidates. The original documentation mentions this is deprecated as it does not really influence things, but you're still free to play around with it.
	Currently, setting requires regenerating your voice latents, as I forgot to have it return some extra data that weighing against the CVVP model uses. Oops.
	Setting this to 1 leads to bad behavior.

## Example(s)

Below are some (rather outdated) outputs I deem substantial enough to share. As I continue delving into TorToiSe, I'll supply more examples and the values I use.

Source (Patrick Bateman): 
* https://files.catbox.moe/skzumo.zip

Output (`My name is Patrick Bateman.`, `fast` preset):
* https://files.catbox.moe/cw88t5.wav
* https://files.catbox.moe/bwunfo.wav
* https://files.catbox.moe/ppxprv.wav

I trimmed up some of the samples to end up with ten short clips of about 10 seconds each. With a 2060, it took a hair over a minute to generate the initial samples, then five to ten seconds for each clip of a total of three. Not too bad for something running on consumer grade shitware.

Source (Harry Mason):
* https://files.catbox.moe/n2xor1.mp3
* https://files.catbox.moe/bbfke3.mp3

Output (The McDonalds building creepypasta, custom preset of 128 samples, 256 iterations):
* https://voca.ro/16XSgdlcC5uT

This took quite a while, over the course of a day half-paying-attention at the command prompt to generate the next piece. I only had to regenerate one section that sounded funny, but compared to 11.AI requiring tons of regenerations for something usable, this is nice to just let run and forget. Initially he sounds rather passable as Harry Mason, but as it goes on it seems to kinda falter. Sound effects and music are added in post and aren't generated by TorToiSe.

Source (James Sunderland):
* https://files.catbox.moe/ynoeld.mp3
* https://files.catbox.moe/lxgbsm.mp3

Output (The McDonalds building creepypasta, 256 samples, 256 iterations, 0.1 temp, pause size 8, DDIM, conditioning free, seed 1675690127):
* https://vocaroo.com/1nXmip0oJu8Z

This took a while to generate while I slept (and even managed to wake up before it finished). Using the batch function, this took 6.919 hours on my 2060 to generate the 27 pieces with zero editing on my end.

I'm providing this even with its nasty warts to highlight the quirks: the weird gaps where there's a strange sound instead, the random pauses for "thought", etc.

I think this also highlights how just combining your entire source sample gung-ho isn't a good idea, as he's not as high of a pitch in his delivery compared to how he usually is throughout most of the game (a sort of average between his two ranges). I can't gauge how well it did in reproducing it, since my ears are pretty much burnt out from listening to so many clips, but I believe he's pretty believable as a James Sunderland.

Output (`Is that really you, Mary?`, Ultra Fast preset, settings and latents embedded)
* https://files.catbox.moe/gy1jvz.wav

This was just a quick test for an adjustable setting, but this one turned out really nice on the off chance. It's not the original delivery, and it definitely sounds robotic still, but it's on the Ultra Fast preset, as expected.

## Caveats (and Upsides)

To me, I find a few problems with TorToiSe over 11.AI:
* computation time is quite an issue. Despite Stable Diffusion proving to be adequate on my 2060, TorToiSe takes quite some time with modest settings.
	- If it did bother me (or bothers you), I would just rent out a Paperspace instance.
	- There's still new gains to be had in diminishing the tax on computing.
* reproducability in a voice depends on the "compatability" with the model TorToiSe was trained on.
	- However, this also appears to be similar to 11.AI, where it was mostly trained on audiobook readings.
* the lack of an obvious analog to the "stability" and "similarity" sliders kind of sucks, but it's not the end of the world.
	However, the `temperature` option seems to prove to be a proper analog to either of these.

However, I can look past these as TorToiSe offers, in comparison to 11.AI:
* the "speaking too fast" issue does not exist with TorToiSe. I don't need to fight with it by pretending I'm a Gaia user in the early 2000s by sprinkling ellipses.
* the overall delivery seems very natural, sometimes small, dramatic pauses gets added at the legitimately most convenient moments, and the inhales tend to be more natural. Many of vocaroos from 11.AI where it just does not seem properly delivered.
* being able to run it locally means I do not have to worry about some Polack seeing me use the "dick" word.