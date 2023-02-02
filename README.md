# AI Voice Cloning for Retards and Savants

This [rentry](https://rentry.org/AI-Voice-Cloning/) aims to serve as both a foolproof guide for setting up AI voice cloning tools for legitimate, local use on Windows (with an Nvidia GPU), as well as a stepping stone for anons that genuinely want to play around with TorToiSe.

Similar to my own findings for Stable Diffusion image generation, this rentry may appear a little disheveled as I note my new findings with TorToiSe. Please keep this in mind if the guide seems to shift a bit or sound confusing.

>\>B-but what about the colab notebook/hugging space instance??

I link those a bit later on as alternatives for Windows+AMD users. You're free to skip the installation section and jump after that.

>\>Ugh... why bother when I can just abuse 11.AI?

I very much encourage (You) to use 11.AI while it's still viable to use. For the layman, it's easier to go through the hoops of coughing up the $5 or abusing the free trial over actually setting up a TorToiSe environment and dealing with its quirks.

However, I also encourage your own experimentation with TorToiSe, as it's very, very promising, it just takes a little love and elbow grease.

## Installing

Below is a very retard-proof guide for getting the software set up. In the future, I'll include a batch script to use for those that don't need tight handholding.

For setting up on Linux, the general framework should be the same, but left as an exercise to the reader.

For Windows users with an AMD GPU, tough luck, as ROCm drivers are not (easily) available for Windows, and requires inane patches with PyTorch. Consider using the [Colab notebook](https://colab.research.google.com/drive/1wVVqUPqwiDBUVeWWOUNglpGhU3hg_cbR?usp=sharing), or the [Hugging Face space](https://huggingface.co/spaces/mdnestor/tortoise), for `tortoise-tts`.

Lots of available RAM seems to be a requirement, as I see Python eating up 8GiB for generations, and if I'm not careful I'll get OOM errors from the software, so be cautious of memory problems if you're doing other things while it runs in the background. For long text generations, you might also exhaust your available VRAM with how the software automatically calculates batch size (for example, a 6GiB of VRAM card using 4GiB for the autoregressive sampling step, but the CLVP matching step requiring more than what's available).

### Pre-Requirements

Anaconda: https://www.anaconda.com/products/distribution

Git (optional): https://git-scm.com/download/win

### Setup

Download Anaconda and run the installer.

After installing `conda`, open the Start Menu and search for `Anaconda Powershell Prompt`. Type `cd `, then drag and drop the folder you want to work in (experienced users can just `cd <path>` directly).

Paste `git clone https://git.ecker.tech/mrq/tortoise-tts` to download TorToiSe and additional scripts. Inexperienced users can just download the repo as a ZIP, and extract.

Then move into that folder with `cd tortoise-tts`. Afterwards, enter `setup.bat` to automatically enter all the remaining commands.

If you've done everything right with installing Anaconda, you shouldn't have any errors.

## Preparing Voice Samples

Now that the tough part is dealt with, it's time to prepare voice sample clips to use.

Unlike training embeddings for AI image generations, preparing a "dataset" for voice cloning is very simple. While the repo suggests using short clips of about ten seconds each, you aren't required to manually snip them up. I'm not sure which way is "better", as some voices work perfectly fine with two clips with minutes each worth of audio, while other voices work better with ten short clips.

As a general rule of thumb, try to source clips that aren't noisy, and are entirely just the subject you are trying to clone. If you must, run your source sample through a background music/noise remover (how to is an exercise left to the reader). It isn't entirely a detriment if you're unable to provide clean audio, however. Just be wary that you might have some headaches with getting acceptable output.

After sourcing your clips, you have two options:
* use all of your samples for voice cloning, providing as much coverage for whatever you may want
* isolate the best of your samples into a few clips (around ten clips each of about ten seconds each), focusing on samples that best match what you're looking to get out of it

Either methods work, but some workloads tend to favor one over the other. If you're running out of options on improving overall cloning quality, consider switching to the other method. In my opinion, the first one seems to work better overall, and rely on other means of improving the quality of cloning.

If you're looking to trim your clips, in my opinion, ~~Audacity~~ Tenacity works good enough, as you can easily output your clips into the proper format (22050 Hz sampling rate, 32-bit float encoding), but some of the time, the software will print out some warning message (`WavFileWarning: Chunk (non-data) not understood, skipping it.`), it's safe to assume you need to properly remux it with `ffmpeg`, simply with `ffmpeg -i [input] -ar 22050 -c:a pcm_f32le [output].wav`. Power users can use the previous command instead of relying on Tenacity to remux.

After preparing your clips as WAV files at a sample rate of 22050 Hz, open up the `tortoise-tts` folder you're working in, navigate to `./tortoise/voice/`, create a new folder in whatever name you want, then dump your clips into that folder. While you're in the `voice` folder, you can take a look at the other provided voices.

**!**NOTE**!**: having a ton of files, regardless of size, substantially increases the time it takes to initialize the voice. I've had it take a while to load 227 or so samples of SA2 Shadow this way. Consider combining them all in one file through Tenacity, with dropping all of your audio files, then Select > Tracks > All, then Tracks > Align Tracks > Align End to End, then exporting the WAV. This does not introduce padding, however.

## Using the Software

Now you're ready to generate clips. With the `conda` prompt still open, simply run the web UI with `python app.py`, and wait for it to print out a URL to open in your browser, something like `http://127.0.0.1:7861`.

If you're looking to access your copy of TorToiSe from outside your local network, pass `--share` into the command (for example, `python app.py --share`). You'll get a temporary gradio link to use.

You'll be presented with a bunch of options, but do not be overwhelmed, as most of the defaults are sane, but below are a rough explanation on which input does what:
* `Text`: text you want to be read
* `Emotion`: the "emotion" used for the delivery. This is a shortcut to starting with `[I am really ${emotion}],` in your text box. I assume the emotion is deduced during the CLVP pass.
* `Voice`: the voice you want to clone. You can select `custom` if you want to use input from your microphone.
* `Record voice`: Not required, unless you use `custom`.
* `Preset`: shortcut values for sample count and iteration steps. Use `none` if you want to provide your own values. Better presets rresult in better quality at the cost of computation time.
* `Seed`: initializes the PRNG initially to this value, use this if you want to reproduce a generated voice. Currently, I don't have a way to expose the seed used.
* `Candidates`: number of outputs to generate, starting from the best candidate. Depending on your iteration steps, generating the final sound files could be cheap, but they only offer alternatives to the samples generated to pull from (in other words, the later candidates perform worse), so don't be compelled to generate a ton of candidates.
* `Autoregressive samples`: analogous to samples in image generation. More samples = better resemblance / clone quality, at the cost of performance.
* `Diffusion iterations`: influences audio sound quality in the final output. More iterations = higher quality sound. This step is relatively cheap, so do not be discouraged from increasing this.
* `Temperature`: how much randomness to introduce to the generated samples. Lower values = better resemblance to the source samples, but some temperature is still required for great output. This value definitely requires playing around depending on the voice you use.

After you fill everything out, click `Submit`, and wait for your outpu in the output window. The sampled voice is also returned, but if you're using multiple files, it'll return the first file, rather than a combined file.



### Command Line Use (legacy)

Ignore this if you're using the web UI, as I'm leaving this in for power users who insist on using the command line. Information below may be a bit outdated.

With the `conda` command prompt still open, simply paste (without the `[]`:

`python tortoise/do_tts.py --text "[text you want]" --voice [voice folder name]`

and wait for the magic. If you have a beefy enough GPU, or a short enough prompt, you should have your output relatively quick.

If nothing funny is printed to the console window, your outputs will show up in the `results` folder under the `tortoise-tts` workspace.

If you want something super quick to test, add `--preset ultra_fast` for a cost in quality. If your samples seem good enough, or the default preset (`fast`) is not quite right, you can add `--preset standard` to get higher quality outputs at a cost of time.

By default, a batch of three clips will be outputted, as it's easier to grab clips from the generate samples, than it is to generate the samples itself. If you want more (or less) clips generated at once, add `--candidates [number]` to the command. It's a pretty good idea to bump this up and pick from the best, rather than re-run the script multiple times. However, it appears the quality drops the more it generates.

With the `--voice [name]` flag, you can combine voices to be "mixed" together by passing `--voice [name1]&[name2]`. I believe there's an upward limit of 3. I have not yet experimented with this, but only noticed this from reading the `do_tts.py` script.

With the `--voice [name]` flag, you can have it iterate through a list of voices to read your text with one command by passing `--voice [name1],[name2]`. I do not believe there's a limit, but I'm not sure why you would need to have multiple voices recite the same text.

If you want a little variety (*sort of* similar to the stability slider), you can use the `--cvvp_amount` flag. From my quick experiments, a value of `0.1` will noticeably vary the voice, so use this value with caution. Additionally, it seems to increase the `Computing best candidates` pass immensely.

Similar to image generation, you can utilize prompt editing to change emotion. In the documentation, you can use something like `[I am really sad,]` before the part where you want an (attempted) sad delivery. **!**NOTE**!**: Prompt engineering seems to not work, as it just outputs eldritch noises.

**!**NOTE**!**: If you see a `WavFileWarning: Chunk (non-data) not understood, skipping it.` error, one of your source files is malformed. It's recommended to re-encode your sound files to make sure it works. For me, running it back in ~~Audacity~~ Tenacity doesn't seem to fix it, but remuxing it with `ffmpeg -i [source].wav -ar 22050 [fixed].wav` seemed to fix it. Outputs where any file gets this error seems to give whacky behavior, from pretty random voices to guttural noises.

There seems to be a huge suite of additional flags to mess around with if you're using the `api.py` script, and seems to be relatively easy to adjust by adding more flags in the `do_tts.py` script, but most of them seem to be fine-tuned and not worth adjusting.

## Example(s)

Below are some outputs I deem substantial enough to share. As I continue delving into TorToiSe, I'll supply more examples and the values I use.

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

This took quite a while, over the course of a day half-paying-attention at the command prompt to generate the next piece. I only had to regenerate one section that sounded funny, but compared to 11.AI requiring tons of regenerations for something usable, this is nice to just let run and forget. Initially he sounds rather passable as Harry Mason, but as it goes on it seems to kinda falter. **!**NOTE**!**: sound effects and music are added in post and aren't generated by TorToiSe.

## Caveats (and Upsides)

To me, I find a few problems:
* a voice's "clonability" depends on the "compatability" with the model TorToiSe was initially trained on.
	It's pretty much a gamble on what plays nicely. Patrick Bateman and Harry Mason will work nice, while James Sunderland, SA2 Shadow, and Mitsuru will refuse to get anything consistently decent. 
* generation time takes quite a while on cards with low compute power (for example, a 2060) for substantial texts, and gets worse for voices with "low compatability" as more samples are required.
	For me personally, if it bothered me, I could rent out a Paperspace instance again and nab the non-pay-as-you-go A100 to crank out audio clips. My 2060 is my secondary card, so it might as well get some use.
* the content of your text could ***greatly*** affect the delivery for the entire text.
	For example, if you lose the die roll and the wrong emotion gets deduced, then it'll throw off the entire clip and subsequent candidates.
	For example, just having the James Sunderland voice say "Mary?" will have it generate as a female voice some of the time.
* the lack of an obvious analog to the "stability" and "similarity" sliders kind of sucks, but it's not the end of the world.
	However, the `temperature` option seems to prove to be a proper analog to either of these.
* I'm not sure if this is specifically an """algorithm""" problem, or is just the nature of sampling, but the GPU is grossly underutilized for compute. I could be wrong and I actually have something misconfigured.

However, I can look past these as TorToiSe offers, in comparison to 11.AI:
* the "speaking too fast" issue does not exist with TorToiSe. I don't need to fight with it by pretending I'm a Gaia user in the early 2000s by sprinkling ellipses.
* the overall delivery seems very natural, sometimes small, dramatic pauses gets added at the legitimately most convenient moments, and the inhales tend to be more natural. Many of vocaroos from 11.AI where it just does not seem properly delivered.
* being able to run it locally means I do not have to worry about some Polack seeing me use the "dick" word.