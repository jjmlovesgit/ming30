#pragma once

#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::ming3_tts_tiny {

// The token ids the converted checkpoint was trained with
// (conversion_manifest.json -> tokenizer.tts_token_ids / prompt_eos_token_id).
struct MingTinySpecialTokens {
    int32_t speaker_start = 157175;   // <spk>
    int32_t audio_patch = 157164;     // <audioPatch>
    int32_t speaker_end = 157176;     // </spk>
    int32_t audio_start = 157165;     // <audio>
    int32_t prompt_eos = 156895;      // <|role_end|>
};

struct MingTinyPromptIds {
    std::vector<int32_t> ids;
    size_t speaker_embedding_position = 0;  // index of the <audioPatch> slot
};

// Stage A surface: the prompt is assembled from separately encoded segments, which
// is what the reference does. Tokenising one joined string can differ at segment
// boundaries, so the segmentation is kept visible for comparison.
class MingTinyPromptBuilder {
public:
    MingTinyPromptBuilder(tokenizers::LlamaBpeTokenizerSpec spec, MingTinySpecialTokens tokens);

    [[nodiscard]] const tokenizers::LlamaBpeTokenizer & tokenizer() const { return tokenizer_; }

    [[nodiscard]] std::vector<std::pair<std::string, std::vector<int32_t>>> build_segments(
        const std::string & text) const;

    [[nodiscard]] MingTinyPromptIds build(const std::string & text) const;

private:
    tokenizers::LlamaBpeTokenizer tokenizer_;
    MingTinySpecialTokens tokens_;
};

}  // namespace engine::models::ming3_tts_tiny
