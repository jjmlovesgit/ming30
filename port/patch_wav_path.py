from pathlib import Path

# encoder.h: declare the WAV loader
h = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\encoder.h")
text = h.read_text(encoding="utf-8")
if "load_wav_mono" not in text:
    text = text.replace(
        "// Probe: encode a raw f32 waveform file",
        "// Load a RIFF/WAVE file as mono float samples. Supports 16-bit PCM and 32-bit float,\n"
        "// averaging channels. Requires 44100 Hz (no resampling here).\n"
        "std::vector<float> load_wav_mono(const std::filesystem::path & path);\n\n"
        "// Probe: encode a raw f32 waveform file")
    h.write_text(text, encoding="utf-8")
    print("header updated")

c = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\encoder.cpp")
text = c.read_text(encoding="utf-8")
anchor = "void encode_reference(const MingTinyAssets & assets, const std::vector<float> & waveform,"
impl = '''std::vector<float> load_wav_mono(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open WAV: " + path.string());
    const auto read_u32 = [&](const char * expected) {
        char tag[4] = {0, 0, 0, 0};
        in.read(tag, 4);
        if (std::memcmp(tag, expected, 4) != 0) {
            throw std::runtime_error("WAV: expected '" + std::string(expected, 4) + "' in " + path.string());
        }
    };
    const auto read_le32 = [&]() {
        unsigned char bytes[4];
        in.read(reinterpret_cast<char *>(bytes), 4);
        return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
               (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    };
    const auto read_le16 = [&]() {
        unsigned char bytes[2];
        in.read(reinterpret_cast<char *>(bytes), 2);
        return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
    };

    read_u32("RIFF");
    (void) read_le32();                     // riff size
    read_u32("WAVE");

    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<char> data;
    while (in) {
        char tag[4];
        in.read(tag, 4);
        if (!in) break;
        const uint32_t size = read_le32();
        if (std::memcmp(tag, "fmt ", 4) == 0) {
            format = read_le16();
            channels = read_le16();
            rate = read_le32();
            (void) read_le32();                 // byte rate
            (void) read_le16();                 // block align
            bits = read_le16();
            for (uint32_t skip = 16; skip < size; ++skip) in.get();
        } else if (std::memcmp(tag, "data", 4) == 0) {
            data.resize(size);
            in.read(data.data(), static_cast<std::streamsize>(size));
        } else {
            for (uint32_t skip = 0; skip < size; ++skip) in.get();
        }
    }

    if (rate != 44100) {
        throw std::runtime_error("WAV must be 44100 Hz; got " + std::to_string(rate) + " (" + path.string() + ")");
    }
    if (channels == 0) throw std::runtime_error("WAV has no channels: " + path.string());

    std::vector<float> samples;
    if (format == 1 && bits == 16) {
        const auto * raw = reinterpret_cast<const int16_t *>(data.data());
        const size_t frames = data.size() / (sizeof(int16_t) * channels);
        samples.resize(frames);
        for (size_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0f;
            for (size_t channel = 0; channel < channels; ++channel) {
                sum += static_cast<float>(raw[frame * channels + channel]) / 32768.0f;
            }
            samples[frame] = sum / static_cast<float>(channels);
        }
    } else if (format == 3 && bits == 32) {
        const auto * raw = reinterpret_cast<const float *>(data.data());
        const size_t frames = data.size() / (sizeof(float) * channels);
        samples.resize(frames);
        for (size_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0f;
            for (size_t channel = 0; channel < channels; ++channel) sum += raw[frame * channels + channel];
            samples[frame] = sum / static_cast<float>(channels);
        }
    } else {
        throw std::runtime_error("WAV: unsupported format " + std::to_string(format) + " / " +
                                 std::to_string(bits) + " bits (" + path.string() + ")");
    }
    return samples;
}

void encode_reference(const MingTinyAssets & assets, const std::vector<float> & waveform,'''
assert anchor in text
text = text.replace(anchor, impl, 1)
if "#include <stdexcept>" not in text:
    text = text.replace("#include <string>", "#include <stdexcept>\n#include <string>")
c.write_text(text, encoding="utf-8")
print("wav loader added")

a = Path(r"C:\Projects\ming30\port\ming3_tts_tiny\assets.cpp")
src = a.read_text(encoding="utf-8")
anchor2 = '''        // Encoder probe: waveform file -> reference latents (math port, temporary env).'''
glue = '''        // WAV-backed podcast prompt: reference clips -> concatenated waveform -> encoder ->
        // prompt matrix. Clips are concatenated in the order given, matching the checkpoint.
        if (const char * wavs = std::getenv("MING3_TINY_PODCAST_WAVS"); wavs != nullptr && *wavs != '\\0') {
            std::vector<std::string> paths;
            std::string current;
            for (const char * cursor = wavs; *cursor != '\\0'; ++cursor) {
                if (*cursor == ';') {
                    if (!current.empty()) paths.push_back(current);
                    current.clear();
                } else {
                    current.push_back(*cursor);
                }
            }
            if (!current.empty()) paths.push_back(current);

            std::vector<float> waveform;
            for (const auto & path : paths) {
                const auto samples = load_wav_mono(path);
                std::printf("  wav reference   : %s -> %zu samples (%.2f s)\\n", path.c_str(), samples.size(),
                    static_cast<double>(samples.size()) / 44100.0);
                waveform.insert(waveform.end(), samples.begin(), samples.end());
            }
            std::vector<float> latents;
            size_t frames = 0;
            encode_reference(*assets, waveform, latents, frames);
            std::printf("  wav encode      : %zu samples -> %zu frames -> %zu latents\\n", waveform.size(), frames,
                latents.size() / 64);

            const char * script_env = std::getenv("MING3_TINY_PODCAST_SCRIPT");
            const char * reference_env = std::getenv("MING3_TINY_PODCAST_REFERENCE");
            const char * out_env = std::getenv("MING3_TINY_PODCAST_ASSEMBLE_DIR");
            dump_podcast_prompt_probe(*assets, script_env != nullptr ? script_env : "",
                reference_env != nullptr ? reference_env : "", latents, latents.size() / 64,
                std::filesystem::path{out_env != nullptr ? out_env : "podcast-wav-embeds"});
        }

''' + anchor2
assert anchor2 in src
a.write_text(src.replace(anchor2, glue, 1), encoding="utf-8")
print("assets.cpp wires the wav path")
