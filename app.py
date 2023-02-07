import os
import argparse
import time
import json
import base64
import re
import urllib.request

import torch
import torchaudio
import music_tag
import gradio as gr

from datetime import datetime

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
from tortoise.utils.text import split_and_recombine_text


def generate(text, delimiter, emotion, prompt, voice, mic_audio, seed, candidates, num_autoregressive_samples, diffusion_iterations, temperature, diffusion_sampler, breathing_room, experimentals, progress=gr.Progress(track_tqdm=True)):
    if voice != "microphone":
        voices = [voice]
    else:
        voices = []

    if voice == "microphone":
        if mic_audio is None:
            raise gr.Error("Please provide audio from mic when choosing `microphone` as a voice input")
        mic = load_audio(mic_audio, 22050)
        voice_samples, conditioning_latents = [mic], None
    else:
        progress(0, desc="Loading voice...")
        voice_samples, conditioning_latents = load_voice(voice)
    
    if voice_samples is not None:
        sample_voice = voice_samples[0]
        conditioning_latents = tts.get_conditioning_latents(voice_samples, progress=progress, max_chunk_size=args.cond_latent_max_chunk_size)
        if voice != "microphone":
            torch.save(conditioning_latents, f'./tortoise/voices/{voice}/cond_latents.pth')
        voice_samples = None
    else:
        sample_voice = None

    if seed == 0:
        seed = None

    start_time = time.time()

    settings = {
        'temperature': temperature, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
        'top_p': .8,
        'cond_free_k': 2.0, 'diffusion_temperature': 1.0,

        'num_autoregressive_samples': num_autoregressive_samples,
        'sample_batch_size': args.sample_batch_size,
        'diffusion_iterations': diffusion_iterations,

        'voice_samples': voice_samples,
        'conditioning_latents': conditioning_latents,
        'use_deterministic_seed': seed,
        'return_deterministic_state': True,
        'k': candidates,
        'diffusion_sampler': diffusion_sampler,
        'breathing_room': breathing_room,
        'progress': progress,
        'half_p': "Half Precision" in experimentals,
        'cond_free': "Conditioning-Free" in experimentals,
    }

    if delimiter == "\\n":
        delimiter = "\n"

    if delimiter != "" and delimiter in text:
        texts = text.split(delimiter)
    else:
        texts = split_and_recombine_text(text)
 
 
    timestamp = int(time.time())
    outdir = f"./results/{voice}/{timestamp}/"
 
    os.makedirs(outdir, exist_ok=True)
 

    audio_cache = {}
    for line, cut_text in enumerate(texts):
        if emotion == "Custom":
            if prompt.strip() != "":
                cut_text = f"[{prompt},] {cut_text}"
        else:
            cut_text = f"[I am really {emotion.lower()},] {cut_text}"

        print(f"[{str(line+1)}/{str(len(texts))}] Generating line: {cut_text}")

        gen, additionals = tts.tts(cut_text, **settings )
        seed = additionals[0]
 
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                audio = g.squeeze(0).cpu()
                audio_cache[f"candidate_{j}/result_{line}.wav"] = {
                    'audio': audio,
                    'text': cut_text,
                }

                os.makedirs(f'{outdir}/candidate_{j}', exist_ok=True)
                torchaudio.save(f'{outdir}/candidate_{j}/result_{line}.wav', audio, 24000)
        else:
            audio = gen.squeeze(0).cpu()
            audio_cache[f"result_{line}.wav"] = {
                'audio': audio,
                'text': cut_text,
            }
            torchaudio.save(f'{outdir}/result_{line}.wav', audio, 24000)
 
    output_voice = None
    if len(texts) > 1:
        for candidate in range(candidates):
            audio_clips = []
            for line in range(len(texts)):
                if isinstance(gen, list):
                    audio = audio_cache[f'candidate_{candidate}/result_{line}.wav']['audio']
                else:
                    audio = audio_cache[f'result_{line}.wav']['audio']
                audio_clips.append(audio)
            
            audio = torch.cat(audio_clips, dim=-1)
            torchaudio.save(f'{outdir}/combined_{candidate}.wav', audio, 24000)

            audio = audio.squeeze(0).cpu()
            audio_cache[f'combined_{candidate}.wav'] = {
                'audio': audio,
                'text': cut_text,
            }

            if output_voice is None:
                output_voice = audio
    else:
        if isinstance(gen, list):
            output_voice = gen[0]
        else:
            output_voice = gen
    
    if output_voice is not None:
        output_voice = (24000, output_voice.numpy())

    info = {
        'text': text,
        'delimiter': '\\n' if delimiter == "\n" else delimiter,
        'emotion': emotion,
        'prompt': prompt,
        'voice': voice,
        'mic_audio': mic_audio,
        'seed': seed,
        'candidates': candidates,
        'num_autoregressive_samples': num_autoregressive_samples,
        'diffusion_iterations': diffusion_iterations,
        'temperature': temperature,
        'diffusion_sampler': diffusion_sampler,
        'breathing_room': breathing_room,
        'experimentals': experimentals,
        'time': time.time()-start_time,
    }
    
    with open(f'{outdir}/input.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(info, indent='\t') )

    if voice is not None and conditioning_latents is not None:
        with open(f'./tortoise/voices/{voice}/cond_latents.pth', 'rb') as f:
            info['latents'] = base64.b64encode(f.read()).decode("ascii")


    for path in audio_cache:
        info['text'] = audio_cache[path]['text']

        metadata = music_tag.load_file(f"{outdir}/{path}")
        metadata['lyrics'] = json.dumps(info) 
        metadata.save()
 
    if sample_voice is not None:
        sample_voice = (22050, sample_voice.squeeze().cpu().numpy())
 
    print(f"Saved to '{outdir}'")

    info['seed'] = settings['use_deterministic_seed']
    del info['latents']
    with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(info, indent='\t') )

    return (
        sample_voice,
        output_voice, 
        seed
    )

def update_presets(value):
    PRESETS = {
        'Ultra Fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
        'Fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
        'Standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
        'High Quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
    }
    
    if value in PRESETS:
        preset = PRESETS[value]
        return (gr.update(value=preset['num_autoregressive_samples']), gr.update(value=preset['diffusion_iterations']))
    else:
        return (gr.update(), gr.update())

def read_generate_settings(file, save_latents=True):
    j = None
    latents = None

    if file is not None:
        if hasattr(file, 'name'):
            metadata = music_tag.load_file(file.name)
            if 'lyrics' in metadata:
                j = json.loads(str(metadata['lyrics']))
        elif file[-5:] == ".json":
            with open(file, 'r') as f:
                j = json.load(f)
    
    if 'latents' in j and save_latents:
        latents = base64.b64decode(j['latents'])
        del j['latents']

    if latents and save_latents:
        outdir='/voices/.temp/'
        os.makedirs(outdir, exist_ok=True)
        with open(f'{outdir}/cond_latents.pth', 'wb') as f:
            f.write(latents)
        latents = f'{outdir}/cond_latents.pth'

    return (
        j,
        latents
    )

def import_generate_settings(file="./config/generate.json"):
    settings, _ = read_generate_settings(file, save_latents=False)
    
    if settings is None:
        return None

    return (
        settings['text'],
        settings['delimiter'],
        settings['emotion'],
        settings['prompt'],
        settings['voice'],
        settings['mic_audio'],
        settings['seed'],
        settings['candidates'],
        settings['num_autoregressive_samples'],
        settings['diffusion_iterations'],
        settings['temperature'],
        settings['diffusion_sampler'],
        settings['breathing_room'],
        settings['experimentals'],
    )

def curl(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Python'})
        conn = urllib.request.urlopen(req)
        data = conn.read()
        data = data.decode()
        data = json.loads(data)
        conn.close()
        return data
    except Exception as e:
        print(e)
        return None

def check_for_updates():
    if not os.path.isfile('./.git/FETCH_HEAD'):
        print("Cannot check for updates: not from a git repo")
        return False

    with open(f'./.git/FETCH_HEAD', 'r', encoding="utf-8") as f:
        head = f.read()
    
    match = re.findall(r"^([a-f0-9]+).+?https:\/\/(.+?)\/(.+?)\/(.+?)\n", head)
    if match is None or len(match) == 0:
        print("Cannot check for updates: cannot parse FETCH_HEAD")
        return False

    match = match[0]

    local = match[0]
    host = match[1]
    owner = match[2]
    repo = match[3]

    res = curl(f"https://{host}/api/v1/repos/{owner}/{repo}/branches/") #this only works for gitea instances

    if res is None or len(res) == 0:
        print("Cannot check for updates: cannot fetch from remote")
        return False

    remote = res[0]["commit"]["id"]

    if remote != local:
        print(f"New version found: {local[:8]} => {remote[:8]}")
        return True

    return False

def update_voices():
    return gr.Dropdown.update(choices=sorted(os.listdir("./tortoise/voices")) + ["microphone"])

def export_exec_settings( share, check_for_updates, low_vram, cond_latent_max_chunk_size, sample_batch_size, concurrency_count ):
    args.share = share
    args.low_vram = low_vram
    args.check_for_updates = check_for_updates
    args.cond_latent_max_chunk_size = cond_latent_max_chunk_size
    args.sample_batch_size = sample_batch_size
    args.concurrency_count = concurrency_count

    settings = {
        'share': args.share,
        'low-vram':args.low_vram,
        'check-for-updates':args.check_for_updates,
        'cond-latent-max-chunk-size': args.cond_latent_max_chunk_size,
        'sample-batch-size': args.sample_batch_size,
        'concurrency-count': args.concurrency_count,
    }

    with open(f'./config/exec.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(settings, indent='\t') )


def main():
    if not torch.cuda.is_available():
        print("CUDA is NOT available for use.")

    with gr.Blocks() as webui:
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(lines=4, label="Prompt")
                    delimiter = gr.Textbox(lines=1, label="Line Delimiter", placeholder="\\n")

                    emotion = gr.Radio(
                        ["Happy", "Sad", "Angry", "Disgusted", "Arrogant", "Custom"],
                        value="Custom",
                        label="Emotion",
                        type="value",
                        interactive=True
                    )
                    prompt = gr.Textbox(lines=1, label="Custom Emotion + Prompt (if selected)")
                    voice = gr.Dropdown(
                        sorted(os.listdir("./tortoise/voices")) + ["microphone"],
                        label="Voice",
                        type="value",
                    )
                    mic_audio = gr.Audio(
                        label="Microphone Source",
                        source="microphone",
                        type="filepath",
                    )
                    refresh_voices = gr.Button(value="Refresh Voice List")
                    refresh_voices.click(update_voices,
                        inputs=None,
                        outputs=voice
                    )
                    
                    prompt.change(fn=lambda value: gr.update(value="Custom"),
                        inputs=prompt,
                        outputs=emotion
                    )
                    mic_audio.change(fn=lambda value: gr.update(value="microphone"),
                        inputs=mic_audio,
                        outputs=voice
                    )
                with gr.Column():
                    candidates = gr.Slider(value=1, minimum=1, maximum=6, step=1, label="Candidates")
                    seed = gr.Number(value=0, precision=0, label="Seed")

                    preset = gr.Radio(
                        ["Ultra Fast", "Fast", "Standard", "High Quality"],
                        label="Preset",
                        type="value",
                    )
                    num_autoregressive_samples = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Samples")
                    diffusion_iterations = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Iterations")

                    temperature = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.1, label="Temperature")
                    breathing_room = gr.Slider(value=8, minimum=1, maximum=32, step=1, label="Pause Size")
                    diffusion_sampler = gr.Radio(
                        ["P", "DDIM"], # + ["K_Euler_A", "DPM++2M"],
                        value="P",
                        label="Diffusion Samplers",
                        type="value",
                    )

                    preset.change(fn=update_presets,
                        inputs=preset,
                        outputs=[
                            num_autoregressive_samples,
                            diffusion_iterations,
                        ],
                    )
                with gr.Column():
                    selected_voice = gr.Audio(label="Source Sample")
                    output_audio = gr.Audio(label="Output")
                    usedSeed = gr.Textbox(label="Seed", placeholder="0", interactive=False) 
                    
                    submit = gr.Button(value="Generate")
                    #stop = gr.Button(value="Stop")
        with gr.Tab("Utilities"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.File(type="file", label="Audio Input", file_types=["audio"])
                    copy_button = gr.Button(value="Copy Settings")
                with gr.Column():
                    metadata_out = gr.JSON(label="Audio Metadata")
                    latents_out = gr.File(type="binary", label="Voice Latents")

                    audio_in.upload(
                        fn=read_generate_settings,
                        inputs=audio_in,
                        outputs=[
                            metadata_out,
                            latents_out
                        ]
                    )
        with gr.Tab("Settings"):
            with gr.Row():
                with gr.Column():
                    with gr.Box():
                        exec_arg_share = gr.Checkbox(label="Public Share Gradio", value=args.share)
                        exec_check_for_updates = gr.Checkbox(label="Check For Updates", value=args.check_for_updates)
                        exec_arg_low_vram = gr.Checkbox(label="Low VRAM", value=args.low_vram)
                        exec_arg_cond_latent_max_chunk_size = gr.Number(label="Voice Latents Max Chunk Size", precision=0, value=args.cond_latent_max_chunk_size)
                        exec_arg_sample_batch_size = gr.Number(label="Sample Batch Size", precision=0, value=args.sample_batch_size)
                        exec_arg_concurrency_count = gr.Number(label="Concurrency Count", precision=0, value=args.concurrency_count)


                    experimentals = gr.CheckboxGroup(["Half Precision", "Conditioning-Free"], value=["Conditioning-Free"], label="Experimental Flags")

                    check_updates_now = gr.Button(value="Check for Updates")

                    exec_inputs = [exec_arg_share, exec_check_for_updates, exec_arg_low_vram, exec_arg_cond_latent_max_chunk_size, exec_arg_sample_batch_size, exec_arg_concurrency_count]

                    for i in exec_inputs:
                        i.change(
                            fn=export_exec_settings,
                            inputs=exec_inputs
                        )

                    check_updates_now.click(check_for_updates)

        input_settings = [
            text,
            delimiter,
            emotion,
            prompt,
            voice,
            mic_audio,
            seed,
            candidates,
            num_autoregressive_samples,
            diffusion_iterations,
            temperature,
            diffusion_sampler,
            breathing_room,
            experimentals,
        ]

        submit_event = submit.click(generate,
            inputs=input_settings,
            outputs=[selected_voice, output_audio, usedSeed],
        )

        copy_button.click(import_generate_settings,
            inputs=audio_in, # JSON elements cannt be used as inputs
            outputs=input_settings
        )

        if os.path.isfile('./config/generate.json'):
            webui.load(import_generate_settings, inputs=None, outputs=input_settings)
        
        if args.check_for_updates:
            webui.load(check_for_updates)

        #stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event])

    webui.queue(concurrency_count=args.concurrency_count).launch(share=args.share)


if __name__ == "__main__":

    default_arguments = {
        'share': False,
        'check-for-updates': False,
        'low-vram': False,
        'cond-latent-max-chunk-size': 1000000,
        'sample-batch-size': None,
        'concurrency-count': 3,
    }

    if os.path.isfile('./config/exec.json'):
        with open(f'./config/exec.json', 'r', encoding="utf-8") as f:
            overrides = json.load(f)
            for k in overrides:
                default_arguments[k] = overrides[k]

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action='store_true', default=default_arguments['share'], help="Lets Gradio return a public URL to use anywhere")
    parser.add_argument("--check-for-updates", action='store_true', default=default_arguments['check-for-updates'], help="Checks for update on startup")
    parser.add_argument("--low-vram", action='store_true', default=default_arguments['low-vram'], help="Disables some optimizations that increases VRAM usage")
    parser.add_argument("--cond-latent-max-chunk-size", default=default_arguments['cond-latent-max-chunk-size'], type=int, help="Sets an upper limit to audio chunk size when computing conditioning latents")
    parser.add_argument("--sample-batch-size", default=default_arguments['sample-batch-size'], type=int, help="Sets an upper limit to audio chunk size when computing conditioning latents")
    parser.add_argument("--concurrency-count", type=int, default=default_arguments['concurrency-count'], help="How many Gradio events to process at once")
    args = parser.parse_args()

    print("Initializating TorToiSe...")
    tts = TextToSpeech(minor_optimizations=not args.low_vram)

    main()