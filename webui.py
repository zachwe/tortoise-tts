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
import gradio.utils

from datetime import datetime

import tortoise.api

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices, get_voice_dir
from tortoise.utils.text import split_and_recombine_text

voicefixer = None

def generate(
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
    cvvp_weight,
    top_p,
    diffusion_temperature,
    length_penalty,
    repetition_penalty,
    cond_free_k,
    experimental_checkboxes,
    progress=None
):
    global args
    global tts

    try:
        tts
    except NameError:
        raise gr.Error("TTS is still initializing...")

    if voice != "microphone":
        voices = [voice]
    else:
        voices = []

    if voice == "microphone":
        if mic_audio is None:
            raise gr.Error("Please provide audio from mic when choosing `microphone` as a voice input")
        mic = load_audio(mic_audio, tts.input_sample_rate)
        voice_samples, conditioning_latents = [mic], None
    else:
        progress(0, desc="Loading voice...")
        voice_samples, conditioning_latents = load_voice(voice)

    if voice_samples is not None:
        sample_voice = voice_samples[0].squeeze().cpu()

        conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=not args.latents_lean_and_mean, progress=progress, max_chunk_size=args.cond_latent_max_chunk_size)
        if len(conditioning_latents) == 4:
            conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
            
        if voice != "microphone":
            torch.save(conditioning_latents, f'{get_voice_dir()}/{voice}/cond_latents.pth')
        voice_samples = None
    else:
        sample_voice = None

    if seed == 0:
        seed = None

    if conditioning_latents is not None and len(conditioning_latents) == 2 and cvvp_weight > 0:
        print("Requesting weighing against CVVP weight, but voice latents are missing some extra data. Please regenerate your voice latents.")
        cvvp_weight = 0


    settings = {
        'temperature': float(temperature),

        'top_p': float(top_p),
        'diffusion_temperature': float(diffusion_temperature),
        'length_penalty': float(length_penalty),
        'repetition_penalty': float(repetition_penalty),
        'cond_free_k': float(cond_free_k),

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
        'half_p': "Half Precision" in experimental_checkboxes,
        'cond_free': "Conditioning-Free" in experimental_checkboxes,
        'cvvp_amount': cvvp_weight,
    }

    if delimiter == "\\n":
        delimiter = "\n"

    if delimiter != "" and delimiter in text:
        texts = text.split(delimiter)
    else:
        texts = split_and_recombine_text(text)
 
    full_start_time = time.time()
 
    outdir = f"./results/{voice}/"
    os.makedirs(outdir, exist_ok=True)

    audio_cache = {}

    resample = None
    # not a ternary in the event for some reason I want to rely on librosa's upsampling interpolator rather than torchaudio's, for some reason
    if tts.output_sample_rate != args.output_sample_rate:
        resampler = torchaudio.transforms.Resample(
            tts.output_sample_rate,
            args.output_sample_rate,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="kaiser_window",
            beta=8.555504641634386,
        )

    volume_adjust = torchaudio.transforms.Vol(gain=args.output_volume, gain_type="amplitude") if args.output_volume != 1 else None

    idx = 1
    for i, file in enumerate(os.listdir(outdir)):
        if file[-5:] == ".json":
            idx = idx + 1

    # reserve, if for whatever reason you manage to concurrently generate
    with open(f'{outdir}/input_{idx}.json', 'w', encoding="utf-8") as f:
        f.write(" ")

    def get_name(line=0, candidate=0, combined=False):
        name = f"{idx}"
        if combined:
            name = f"{name}_combined"
        elif len(texts) > 1:
            name = f"{name}_{line}"
        if candidates > 1:
            name = f"{name}_{candidate}"
        return name

    for line, cut_text in enumerate(texts):
        if emotion == "Custom":
            if prompt.strip() != "":
                cut_text = f"[{prompt},] {cut_text}"
        else:
            cut_text = f"[I am really {emotion.lower()},] {cut_text}"

        progress.msg_prefix = f'[{str(line+1)}/{str(len(texts))}]'
        print(f"{progress.msg_prefix} Generating line: {cut_text}")

        start_time = time.time()
        gen, additionals = tts.tts(cut_text, **settings )
        seed = additionals[0]
        run_time = time.time()-start_time
        print(f"Generating line took {run_time} seconds")
 
        if not isinstance(gen, list):
            gen = [gen]

        for j, g in enumerate(gen):
            audio = g.squeeze(0).cpu()
            name = get_name(line=line, candidate=j)
            audio_cache[name] = {
                'audio': audio,
                'text': cut_text,
                'time': run_time
            }
            # save here in case some error happens mid-batch
            torchaudio.save(f'{outdir}/{voice}_{name}.wav', audio, args.output_sample_rate)

    for k in audio_cache:
        audio = audio_cache[k]['audio']

        if resampler is not None:
            audio = resampler(audio)
        if volume_adjust is not None:
            audio = volume_adjust(audio)

        audio_cache[k]['audio'] = audio
        torchaudio.save(f'{outdir}/{voice}_{k}.wav', audio, args.output_sample_rate)
 
    output_voice = None
    output_voices = []
    for candidate in range(candidates):
        if len(texts) > 1:
            audio_clips = []
            for line in range(len(texts)):
                name = get_name(line=line, candidate=candidate)
                audio = audio_cache[name]['audio']
                audio_clips.append(audio)
            
            name = get_name(candidate=candidate, combined=True)
            audio = torch.cat(audio_clips, dim=-1)
            torchaudio.save(f'{outdir}/{voice}_{name}.wav', audio, args.output_sample_rate)

            audio = audio.squeeze(0).cpu()
            audio_cache[name] = {
                'audio': audio,
                'text': text,
                'time': time.time()-full_start_time
            }

            output_voices.append(f'{outdir}/{voice}_{name}.wav')
            if output_voice is None:
                output_voice = f'{outdir}/{voice}_{name}.wav'
        else:
            name = get_name(candidate=candidate)
            output_voices.append(f'{outdir}/{voice}_{name}.wav')

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
        'cvvp_weight': cvvp_weight,
        'top_p': top_p,
        'diffusion_temperature': diffusion_temperature,
        'length_penalty': length_penalty,
        'repetition_penalty': repetition_penalty,
        'cond_free_k': cond_free_k,
        'experimentals': experimental_checkboxes,
        'time': time.time()-full_start_time,
    }
    
    with open(f'{outdir}/input_{idx}.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(info, indent='\t') )

    if voice is not None and conditioning_latents is not None:
        with open(f'{get_voice_dir()}/{voice}/cond_latents.pth', 'rb') as f:
            info['latents'] = base64.b64encode(f.read()).decode("ascii")

    if args.voice_fixer and voicefixer:
        # we could do this on the pieces before they get stiched up anyways to save some compute
        # but the stitching would need to read back from disk, defeating the point of caching the waveform
        for path in progress.tqdm(audio_cache, desc="Running voicefix..."):
            voicefixer.restore(
                input=f'{outdir}/{voice}_{k}.wav',
                output=f'{outdir}/{voice}_{k}.wav',
                #cuda=False,
                #mode=mode,
            )

    if args.embed_output_metadata:
        for path in progress.tqdm(audio_cache, desc="Embedding metadata..."):
            info['text'] = audio_cache[path]['text']
            info['time'] = audio_cache[path]['time']

            metadata = music_tag.load_file(f"{outdir}/{voice}_{path}.wav")
            metadata['lyrics'] = json.dumps(info) 
            metadata.save()

    #if output_voice is not None:
    #    output_voice = (args.output_sample_rate, output_voice.numpy())
 
    if sample_voice is not None:
        sample_voice = (tts.input_sample_rate, sample_voice.numpy())

    print(f"Generation took {info['time']} seconds, saved to '{output_voices[0]}'\n")

    info['seed'] = settings['use_deterministic_seed']
    del info['latents']
    with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(info, indent='\t') )

    stats = [
        [ seed, "{:.3f}".format(info['time']) ]
    ]

    return (
        sample_voice,
        output_voices,
        stats,
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

def read_generate_settings(file, save_latents=True, save_as_temp=True):
    j = None
    latents = None

    if file is not None:
        if hasattr(file, 'name'):
            file = file.name

        if file[-4:] == ".wav":
            metadata = music_tag.load_file(file)
            if 'lyrics' in metadata:
                j = json.loads(str(metadata['lyrics']))
        elif file[-5:] == ".json":
            with open(file, 'r') as f:
                j = json.load(f)
    
    if 'latents' in j and save_latents:
        latents = base64.b64decode(j['latents'])
        del j['latents']

    if latents and save_latents:
        outdir=f'{get_voice_dir()}/{".temp" if save_as_temp else j["voice"]}/'
        os.makedirs(outdir, exist_ok=True)
        with open(f'{outdir}/cond_latents.pth', 'wb') as f:
            f.write(latents)
        latents = f'{outdir}/cond_latents.pth'

    if "time" in j:
        j["time"] = "{:.3f}".format(j["time"])

    return (
        j,
        latents
    )
def save_latents(file):
    read_generate_settings(file, save_latents=True, save_as_temp=False)

def import_generate_settings(file="./config/generate.json"):
    settings, _ = read_generate_settings(file, save_latents=False)
    
    if settings is None:
        return None

    return (
        None if 'text' not in settings else settings['text'],
        None if 'delimiter' not in settings else settings['delimiter'],
        None if 'emotion' not in settings else settings['emotion'],
        None if 'prompt' not in settings else settings['prompt'],
        None if 'voice' not in settings else settings['voice'],
        None if 'mic_audio' not in settings else settings['mic_audio'],
        None if 'seed' not in settings else settings['seed'],
        None if 'candidates' not in settings else settings['candidates'],
        None if 'num_autoregressive_samples' not in settings else settings['num_autoregressive_samples'],
        None if 'diffusion_iterations' not in settings else settings['diffusion_iterations'],
        0.8 if 'temperature' not in settings else settings['temperature'],
        "DDIM" if 'diffusion_sampler' not in settings else settings['diffusion_sampler'],
        8   if 'breathing_room' not in settings else settings['breathing_room'],
        0.0 if 'cvvp_weight' not in settings else settings['cvvp_weight'],
        0.8 if 'top_p' not in settings else settings['top_p'],
        1.0 if 'diffusion_temperature' not in settings else settings['diffusion_temperature'],
        1.0 if 'length_penalty' not in settings else settings['length_penalty'],
        2.0 if 'repetition_penalty' not in settings else settings['repetition_penalty'],
        2.0 if 'cond_free_k' not in settings else settings['cond_free_k'],
        None if 'experimentals' not in settings else settings['experimentals'],
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

def reload_tts():
    global tts
    del tts
    tts = setup_tortoise(restart=True)

def cancel_generate():
    tortoise.api.STOP_SIGNAL = True

def get_voice_list():
    voice_dir = get_voice_dir()
    return [d for d in os.listdir(voice_dir) if os.path.isdir(os.path.join(voice_dir, d))]

def update_voices():
    return gr.Dropdown.update(choices=sorted(get_voice_list()) + ["microphone"])

def export_exec_settings( share, listen, check_for_updates, models_from_local_only, low_vram, embed_output_metadata, latents_lean_and_mean, voice_fixer, cond_latent_max_chunk_size, sample_batch_size, concurrency_count, output_sample_rate, output_volume ):
    args.share = share
    args.listen = listen
    args.low_vram = low_vram
    args.check_for_updates = check_for_updates
    args.models_from_local_only = models_from_local_only
    args.cond_latent_max_chunk_size = cond_latent_max_chunk_size
    args.sample_batch_size = sample_batch_size
    args.embed_output_metadata = embed_output_metadata
    args.latents_lean_and_mean = latents_lean_and_mean
    args.voice_fixer = voice_fixer
    args.concurrency_count = concurrency_count
    args.output_sample_rate = output_sample_rate
    args.output_volume = output_volume

    settings = {
        'share': args.share,
        'listen': args.listen,
        'low-vram':args.low_vram,
        'check-for-updates':args.check_for_updates,
        'models-from-local-only':args.models_from_local_only,
        'cond-latent-max-chunk-size': args.cond_latent_max_chunk_size,
        'sample-batch-size': args.sample_batch_size,
        'embed-output-metadata': args.embed_output_metadata,
        'latents-lean-and-mean': args.latents_lean_and_mean,
        'voice-fixer': args.voice_fixer,
        'concurrency-count': args.concurrency_count,
        'output-sample-rate': args.output_sample_rate,
        'output-volume': args.output_volume,
    }

    with open(f'./config/exec.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(settings, indent='\t') )

def setup_args():
    default_arguments = {
        'share': False,
        'listen': None,
        'check-for-updates': False,
        'models-from-local-only': False,
        'low-vram': False,
        'sample-batch-size': None,
        'embed-output-metadata': True,
        'latents-lean-and-mean': True,
        'voice-fixer': True,
        'cond-latent-max-chunk-size': 1000000,
        'concurrency-count': 2,
        'output-sample-rate': 44100,
        'output-volume': 1,
    }

    if os.path.isfile('./config/exec.json'):
        with open(f'./config/exec.json', 'r', encoding="utf-8") as f:
            overrides = json.load(f)
            for k in overrides:
                default_arguments[k] = overrides[k]

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action='store_true', default=default_arguments['share'], help="Lets Gradio return a public URL to use anywhere")
    parser.add_argument("--listen", default=default_arguments['listen'], help="Path for Gradio to listen on")
    parser.add_argument("--check-for-updates", action='store_true', default=default_arguments['check-for-updates'], help="Checks for update on startup")
    parser.add_argument("--models-from-local-only", action='store_true', default=default_arguments['models-from-local-only'], help="Only loads models from disk, does not check for updates for models")
    parser.add_argument("--low-vram", action='store_true', default=default_arguments['low-vram'], help="Disables some optimizations that increases VRAM usage")
    parser.add_argument("--no-embed-output-metadata", action='store_false', default=not default_arguments['embed-output-metadata'], help="Disables embedding output metadata into resulting WAV files for easily fetching its settings used with the web UI (data is stored in the lyrics metadata tag)")
    parser.add_argument("--latents-lean-and-mean", action='store_true', default=default_arguments['latents-lean-and-mean'], help="Exports the bare essentials for latents.")
    parser.add_argument("--voice-fixer", action='store_true', default=default_arguments['voice-fixer'], help="Uses python module 'voicefixer' to improve audio quality, if available.")
    parser.add_argument("--cond-latent-max-chunk-size", default=default_arguments['cond-latent-max-chunk-size'], type=int, help="Sets an upper limit to audio chunk size when computing conditioning latents")
    parser.add_argument("--sample-batch-size", default=default_arguments['sample-batch-size'], type=int, help="Sets an upper limit to audio chunk size when computing conditioning latents")
    parser.add_argument("--concurrency-count", type=int, default=default_arguments['concurrency-count'], help="How many Gradio events to process at once")
    parser.add_argument("--output-sample-rate", type=int, default=default_arguments['output-sample-rate'], help="Sample rate to resample the output to (from 24KHz)")
    parser.add_argument("--output-volume", type=float, default=default_arguments['output-volume'], help="Adjusts volume of output")
    args = parser.parse_args()

    args.embed_output_metadata = not args.no_embed_output_metadata

    args.listen_host = None
    args.listen_port = None
    args.listen_path = None
    if args.listen:
        match = re.findall(r"^(?:(.+?):(\d+))?(\/.+?)?$", args.listen)[0]

        args.listen_host = match[0] if match[0] != "" else "127.0.0.1"
        args.listen_port = match[1] if match[1] != "" else None
        args.listen_path = match[2] if match[2] != "" else "/"

    if args.listen_port is not None:
        args.listen_port = int(args.listen_port)
    
    return args

def setup_tortoise(restart=False):
    global args
    global tts
    global voicefixer

    if args.voice_fixer and not restart:
        try:
            from voicefixer import VoiceFixer
            print("Initializating voice-fixer")
            voicefixer = VoiceFixer()
            print("initialized voice-fixer")
        except Exception as e:
            pass

    print("Initializating TorToiSe...")
    tts = TextToSpeech(minor_optimizations=not args.low_vram)
    print("TorToiSe initialized, ready for generation.")
    return tts

def setup_gradio():
    global args
    
    if not args.share:
        def noop(function, return_value=None):
            def wrapped(*args, **kwargs):
                return return_value
            return wrapped
        gradio.utils.version_check = noop(gradio.utils.version_check)
        gradio.utils.initiated_analytics = noop(gradio.utils.initiated_analytics)
        gradio.utils.launch_analytics = noop(gradio.utils.launch_analytics)
        gradio.utils.integration_analytics = noop(gradio.utils.integration_analytics)
        gradio.utils.error_analytics = noop(gradio.utils.error_analytics)
        gradio.utils.log_feature_analytics = noop(gradio.utils.log_feature_analytics)
        #gradio.utils.get_local_ip_address = noop(gradio.utils.get_local_ip_address, 'localhost')

    if args.models_from_local_only:
        os.environ['TRANSFORMERS_OFFLINE']='1'

    with gr.Blocks() as webui:
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(lines=4, label="Prompt")
            with gr.Row():
                with gr.Column():
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
                        sorted(get_voice_list()) + ["microphone"],
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

                    show_experimental_settings = gr.Checkbox(label="Show Experimental Settings")
                with gr.Column(visible=False) as col:
                    experimental_column = col

                    experimental_checkboxes = gr.CheckboxGroup(["Half Precision", "Conditioning-Free"], value=["Conditioning-Free"], label="Experimental Flags")
                    cvvp_weight = gr.Slider(value=0, minimum=0, maximum=1, label="CVVP Weight")
                    top_p = gr.Slider(value=0.8, minimum=0, maximum=1, label="Top P")
                    diffusion_temperature = gr.Slider(value=1.0, minimum=0, maximum=1, label="Diffusion Temperature")
                    length_penalty = gr.Slider(value=1.0, minimum=0, maximum=8, label="Length Penalty")
                    repetition_penalty = gr.Slider(value=2.0, minimum=0, maximum=8, label="Repetition Penalty")
                    cond_free_k = gr.Slider(value=2.0, minimum=0, maximum=4, label="Conditioning-Free K")

                    show_experimental_settings.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=show_experimental_settings,
                        outputs=experimental_column
                    )
                with gr.Column():
                    submit = gr.Button(value="Generate")
                    stop = gr.Button(value="Stop")

                    generation_results = gr.Dataframe(label="Results", headers=["Seed", "Time"], visible=False)
                    source_sample = gr.Audio(label="Source Sample", visible=False)
                    output_audio = gr.Audio(label="Output")
                    candidates_list = gr.Dropdown(label="Candidates", type="value", visible=False)
                    output_pick = gr.Button(value="Select Candidate", visible=False)
                    
        with gr.Tab("History"):
            with gr.Row():
                with gr.Column():
                    headers = {
                        "Name": "",
                        "Samples": "num_autoregressive_samples",
                        "Iterations": "diffusion_iterations",
                        "Temp.": "temperature",
                        "Sampler": "diffusion_sampler",
                        "CVVP": "cvvp_weight",
                        "Top P": "top_p",
                        "Diff. Temp.": "diffusion_temperature",
                        "Len Pen": "length_penalty",
                        "Rep Pen": "repetition_penalty",
                        "Cond-Free K": "cond_free_k",
                        "Time": "time",
                    }
                    history_info = gr.Dataframe(label="Results", headers=list(headers.keys()))
            with gr.Row():
                with gr.Column():
                    history_voices = gr.Dropdown(
                        sorted(get_voice_list()) + ["microphone"],
                        label="Voice",
                        type="value",
                    )

                    history_view_results_button = gr.Button(value="View Files")
                with gr.Column():
                    history_results_list = gr.Dropdown(label="Results",type="value", interactive=True)
                    history_view_result_button = gr.Button(value="View File")
                with gr.Column():
                    history_audio = gr.Audio()
                    history_copy_settings_button = gr.Button(value="Copy Settings")
                
                def history_view_results( voice ):
                    results = []
                    files = []
                    outdir = f"./results/{voice}/"
                    for i, file in enumerate(os.listdir(outdir)):
                        if file[-4:] != ".wav":
                            continue

                        metadata, _ = read_generate_settings(f"{outdir}/{file}", save_latents=False)
                        if metadata is None:
                            continue
                            
                        values = []
                        for k in headers:
                            v = file
                            if k != "Name":
                                v = metadata[headers[k]]
                            values.append(v)


                        files.append(file)
                        results.append(values)

                    return (
                        results,
                        gr.Dropdown.update(choices=sorted(files))
                    )

                history_view_results_button.click(
                    fn=history_view_results,
                    inputs=history_voices,
                    outputs=[
                        history_info,
                        history_results_list,
                    ]
                )
                history_view_result_button.click(
                    fn=lambda voice, file: f"./results/{voice}/{file}",
                    inputs=[
                        history_voices,
                        history_results_list,
                    ],
                    outputs=history_audio
                )
        with gr.Tab("Utilities"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.File(type="file", label="Audio Input", file_types=["audio"])
                    copy_button = gr.Button(value="Copy Settings")
                    import_voice = gr.Button(value="Import Voice")
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

                import_voice.click(
                    fn=save_latents,
                    inputs=audio_in,
                )
        with gr.Tab("Settings"):
            with gr.Row():
                exec_inputs = []
                with gr.Column():
                    exec_inputs = exec_inputs + [
                        gr.Textbox(label="Listen", value=args.listen, placeholder="127.0.0.1:7860/"),
                        gr.Checkbox(label="Public Share Gradio", value=args.share),
                        gr.Checkbox(label="Check For Updates", value=args.check_for_updates),
                        gr.Checkbox(label="Only Load Models Locally", value=args.models_from_local_only),
                        gr.Checkbox(label="Low VRAM", value=args.low_vram),
                        gr.Checkbox(label="Embed Output Metadata", value=args.embed_output_metadata),
                        gr.Checkbox(label="Slimmer Computed Latents", value=args.latents_lean_and_mean),
                        gr.Checkbox(label="Voice Fixer", value=args.voice_fixer),
                    ]
                    gr.Button(value="Check for Updates").click(check_for_updates)
                    gr.Button(value="Reload TTS").click(reload_tts)
                with gr.Column():
                    exec_inputs = exec_inputs + [
                        gr.Number(label="Voice Latents Max Chunk Size", precision=0, value=args.cond_latent_max_chunk_size),
                        gr.Number(label="Sample Batch Size", precision=0, value=args.sample_batch_size),
                        gr.Number(label="Concurrency Count", precision=0, value=args.concurrency_count),
                        gr.Number(label="Ouptut Sample Rate", precision=0, value=args.output_sample_rate),
                        gr.Slider(label="Ouptut Volume", minimum=0, maximum=2, value=args.output_volume),
                    ]

                for i in exec_inputs:
                    i.change(
                        fn=export_exec_settings,
                        inputs=exec_inputs
                    )

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
            cvvp_weight,
            top_p,
            diffusion_temperature,
            length_penalty,
            repetition_penalty,
            cond_free_k,
            experimental_checkboxes,
        ]

        # YUCK
        def run_generation(
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
            cvvp_weight,
            top_p,
            diffusion_temperature,
            length_penalty,
            repetition_penalty,
            cond_free_k,
            experimental_checkboxes,
            progress=gr.Progress(track_tqdm=True)
        ):
            try:
                sample, outputs, stats = generate(
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
                    cvvp_weight,
                    top_p,
                    diffusion_temperature,
                    length_penalty,
                    repetition_penalty,
                    cond_free_k,
                    experimental_checkboxes,
                    progress
                )
            except Exception as e:
                message = str(e)
                if message == "Kill signal detected":
                    reload_tts()

                raise gr.Error(message)
            

            return (
                outputs[0],
                gr.update(value=sample, visible=sample is not None),
                gr.update(choices=outputs, value=outputs[0], visible=len(outputs) > 1, interactive=True),
                gr.update(visible=len(outputs) > 1),
                gr.update(value=stats, visible=True),
            )

        output_pick.click(
            lambda x: x,
            inputs=candidates_list,
            outputs=output_audio,
        )

        submit.click(
            lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
            outputs=[source_sample, candidates_list, output_pick, generation_results],
        )

        submit_event = submit.click(run_generation,
            inputs=input_settings,
            outputs=[output_audio, source_sample, candidates_list, output_pick, generation_results],
        )


        copy_button.click(import_generate_settings,
            inputs=audio_in, # JSON elements cannot be used as inputs
            outputs=input_settings
        )

        def history_copy_settings( voice, file ):
            settings = import_generate_settings( f"./results/{voice}/{file}" )
            return settings

        history_copy_settings_button.click(history_copy_settings,
            inputs=[
                history_voices,
                history_results_list,
            ],
            outputs=input_settings
        )

        if os.path.isfile('./config/generate.json'):
            webui.load(import_generate_settings, inputs=None, outputs=input_settings)
        
        if args.check_for_updates:
            webui.load(check_for_updates)

        stop.click(fn=cancel_generate, inputs=None, outputs=None, cancels=[submit_event])


    webui.queue(concurrency_count=args.concurrency_count)

    return webui