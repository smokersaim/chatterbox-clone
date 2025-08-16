import os
import time
import torch
import random
import numpy as np
import gradio as gr
import re
from chatterbox.tts import ChatterboxTTS, Conditionals

torch.set_float32_matmul_precision('high')

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

MODEL = None

PRIMARY_AVATARS = os.path.join("src", "chatterbox", "data", "avatars")
VOICES_PATH = os.path.join("src", "chatterbox", "data", "voices")
os.makedirs(VOICES_PATH, exist_ok=True)
AVATAR = "Tobin"


def load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
            MODEL.to(DEVICE)
        print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
    return MODEL


def list_avatars(preferred=AVATAR):
    # Scan for .wav, .flac, .mp3 files in PRIMARY_AVATARS
    avatar_files = []
    if os.path.isdir(PRIMARY_AVATARS):
        avatar_files = [
            f for f in os.listdir(PRIMARY_AVATARS)
            if os.path.isfile(os.path.join(PRIMARY_AVATARS, f)) and f.lower().endswith(('.wav', '.flac', '.mp3'))
        ]

    # Scan for .pt files in VOICES_PATH
    voice_files = []
    if os.path.isdir(VOICES_PATH):
        voice_files = [
            f for f in os.listdir(VOICES_PATH)
            if os.path.isfile(os.path.join(VOICES_PATH, f)) and f.lower().endswith('.pt')
        ]

    # Create a map from avatar name to file path
    avatar_map = {os.path.splitext(f)[0].capitalize(): os.path.join(PRIMARY_AVATARS, f) for f in avatar_files}
    voice_map = {os.path.splitext(f)[0].capitalize(): os.path.join(VOICES_PATH, f) for f in voice_files}
    avatar_map.update(voice_map)

    # Sort options and set a default
    options = sorted(avatar_map.keys(), key=lambda x: x.lower())
    default = preferred if preferred in avatar_map else (options[0] if options else None)
    return avatar_map, options, default


def set_seed(seed: int):
    seed = int(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_text_into_chunks(text, max_chunk_length=750):
    sentences = re.split(r'([.!?]+)', text)
    chunks = []
    current_chunk = ""
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if not sentence:
            i += 1
            continue
        if i + 1 < len(sentences) and sentences[i + 1].strip() in '.!?':
            sentence += sentences[i + 1].strip()
            i += 2
        else:
            i += 1
        if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if chunk.strip()]


def add_silence_padding(audio, silence_duration=0.5, sample_rate=24000):
    silence_samples = int(silence_duration * sample_rate)
    silence = torch.zeros(audio.shape[0], silence_samples)
    return torch.cat([audio, silence], dim=1)


def generate_long_audio(model, text, audio_prompt_path, max_chunk_length=750, silence_duration=0.5,
                        exaggeration=0.5, temperature=0.8, cfg_weight=0.5):

    if audio_prompt_path:
        if audio_prompt_path.endswith('.pt'):
            model.conds = Conditionals.load(audio_prompt_path, map_location=DEVICE).to(DEVICE)
        else:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

    chunks = split_text_into_chunks(text, max_chunk_length)
    if not chunks:
        return None

    total_chunks = len(chunks)
    total_words = len(text.split())
    print(f"Total chunks: {total_chunks} ({total_words} words total)")

    audio_segments = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{total_chunks} ({len(chunk.split())} words, {len(chunk)} chars)")
        try:
            chunk_audio = model.generate(
                chunk,
                audio_prompt_path=None,
                temperature=temperature,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration
            )
            if i < total_chunks - 1:
                chunk_audio = add_silence_padding(chunk_audio, silence_duration, model.sr)
            audio_segments.append(chunk_audio.cpu())
        except Exception as e:
            print(f"Error generating chunk {i+1}: {e}")
            continue

    if not audio_segments:
        return None

    final_audio = torch.cat(audio_segments, dim=1)
    return final_audio


def generate_audio(
    text_input: str,
    avatar_name: str,
    exaggeration_input: float,
    cfgw_input: float,
    temperature_input: float,
    chunk_len_input: int,
    seed_num_input: float
) -> tuple[int, np.ndarray]:

    current_model = load_model()
    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if int(seed_num_input) != 0:
        set_seed(int(seed_num_input))

    avatar_map, _, _ = list_avatars()
    avatar_path_input = avatar_map.get(avatar_name) if avatar_name else None

    print(f"Generating audio for text: '{text_input[:30]}...'")

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        wav = generate_long_audio(
            model=current_model,
            text=text_input[:10000],
            audio_prompt_path=avatar_path_input,
            max_chunk_length=chunk_len_input,
            silence_duration=0.5,
            exaggeration=exaggeration_input,
            temperature=temperature_input,
            cfg_weight=cfgw_input
        )

    if wav is None:
        raise RuntimeError("Failed to generate audio.")

    print("Audio generation complete.")
    wav_np = wav.squeeze(0).cpu().numpy().astype(np.float32)
    return current_model.sr, wav_np


def prepare_voice(voice_name, audio_file):
    if not voice_name or not audio_file:
        return "Please provide a voice name and an audio file.", gr.update()

    model = load_model()
    voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '_')).rstrip()
    output_path = os.path.join(VOICES_PATH, f"{voice_name}.pt")

    try:
        model.prepare_conditionals(audio_file.name)
        model.conds.save(output_path)

        # Refresh the avatar list
        global avatar_map, avatar_list, default_avatar
        avatar_map, avatar_list, default_avatar = list_avatars()
        return f"Voice '{voice_name}' prepared and saved successfully.", gr.update(choices=avatar_list)
    except Exception as e:
        return f"Error preparing voice: {e}", gr.update()


avatar_map, avatar_list, default_avatar = list_avatars()

with gr.Blocks(
    title="Chatterbox TTS",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    body, .gradio-container {font-family: 'Poppins', sans-serif;}
    .gradio-container {padding: 2rem;}
    footer, #footer, .svelte-1ipelgc {display: none !important;}
    """
) as demo:

    gr.Markdown("""
    ## Chatterbox TTS
    **A streamlined, high-quality text-to-speech tool built for clarity, expressiveness, and control.**
    """)

    with gr.Tabs():
        with gr.TabItem("Generate Speech"):
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(
                        label="📝 Text to Synthesize",
                        placeholder="Enter text to synthesize...",
                        lines=8,
                        max_lines=14,
                    )

                    ref_avatar = gr.Dropdown(
                        choices=avatar_list,
                        label="🎭 Voice Avatar",
                        value=default_avatar,
                        info="Select the voice character"
                    )

                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        exaggeration = gr.Slider(0.25, 2, step=0.05, label="Exaggeration", value=0.5)
                        cfg_weight = gr.Slider(0.2, 1, step=0.05, label="CFG / Pace", value=0.5)
                        temp = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)
                        chunk_len = gr.Slider(500, 800, step=50, label="Chunk Length", value=750)
                        seed_num = gr.Number(value=0, label="Seed (0 = random)")

                    error_display = gr.Markdown(visible=False)
                    run_btn = gr.Button("Generate Speech", variant="primary")

                with gr.Column():
                    audio_output = gr.Audio(
                        label="Output Audio",
                        type="numpy",
                        interactive=False,
                        autoplay=False,
                        show_download_button=True
                    )

                    gr.Markdown("""
                    ### 📋 Model Summary
                    **Current Constraints:**
                    - **ROPE Scaling**: 8,192 positional tokens
                    - **Embeddings**: Text (2,048), Speech (4,096)
                    - **Vocab**: 704 compact tokens
                    - **Chunking**: Adaptive 500–800 characters with silence padding
                    - **Performance**: Optimized for CUDA / MPS with FP16 support
                    - **License**: MIT License – Enhanced by [Usama Arshad](https://github.com/usamaraajput)
                    """)

        with gr.TabItem("Prepare Voice"):
            with gr.Row():
                with gr.Column():
                    voice_name_input = gr.Textbox(label="Voice Name", placeholder="Enter a name for the new voice")
                    audio_file_input = gr.File(label="Reference Audio", file_types=["audio"])
                    prepare_voice_btn = gr.Button("Prepare Voice", variant="primary")
                with gr.Column():
                    prepare_voice_output = gr.Textbox(label="Status")

    gr.Markdown("""
    <div style='position: fixed; bottom: 10px; left: 0; width: 100%; text-align: center; font-size: 0.85em; color: #666; z-index: 999; '>
       Built with ❤️ by <a href='https://github.com/usamaraajput' target='_blank'>Usama Arshad</a> | Based on the open-source <a href='https://github.com/resemble-ai/chatterbox' target='_blank'>Chatterbox</a> project (MIT License)
    </div>
    """)

    def check_limit(txt):
        char_count = len(txt)
        if char_count > 10000:
            return (
                gr.update(visible=True, value=f"⚠️ Text exceeds 10,000 characters ({char_count}/10,000). Please shorten it."),
                gr.update(interactive=False)
            )
        return gr.update(visible=False), gr.update(interactive=True)

    text.change(fn=check_limit, inputs=text, outputs=[error_display, run_btn])

    run_btn.click(
        fn=generate_audio,
        inputs=[text, ref_avatar, exaggeration, cfg_weight, temp, chunk_len, seed_num],
        outputs=[audio_output],
    )

    prepare_voice_btn.click(
        fn=prepare_voice,
        inputs=[voice_name_input, audio_file_input],
        outputs=[prepare_voice_output, ref_avatar],
    )

demo.queue(max_size=5)
demo.launch(inbrowser=True, share=True)
